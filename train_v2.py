import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import copy
import sys
from model.model_v2 import ChaosWaveFSL, create_enhanced_model, create_enhanced_loss
from utils import enhanced_metric_calculation, result_print, EarlyStopping, MetricsTracker


def _default_minibatch_size() -> int:
    return 2048 if torch.cuda.is_available() else 4096


def _slice_data(data: Any, start: int, end: int) -> Any:
    sliced = Data()
    sliced.x          = data.x[start:end]
    sliced.y          = data.y[start:end]
    sliced.node_num   = data.node_num
    sliced.edge_index = data.edge_index
    if hasattr(data, 'spatial_features'):
        sliced.spatial_features  = data.spatial_features
        sliced.temporal_features = data.temporal_features
    return sliced


class EnhancedMetaLearner(nn.Module):
    def __init__(self, model: ChaosWaveFSL, config: Dict):
        super(EnhancedMetaLearner, self).__init__()
        self.model  = model
        self.config = config

        self.meta_lr          = float(config['model'].get('meta_lr', 0.001))
        self.inner_lr         = float(config.get('few_shot', {}).get('inner_lr', 0.01))
        self.adaptation_steps = int(config.get('few_shot', {}).get('adaptation_steps', 5))
        self.num_tasks        = int(config.get('few_shot', {}).get('num_tasks', 2))
        self.support_size     = int(config.get('few_shot', {}).get('support_size', 8))
        self.query_size       = int(config.get('few_shot', {}).get('query_size', 12))
        self.minibatch_size   = int(config.get('training', {}).get('minibatch_size', _default_minibatch_size()))

        self.loss_fn = create_enhanced_loss(config)

        self.meta_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.meta_lr,
            weight_decay=float(config.get('training', {}).get('weight_decay', 1e-4)),
            betas=(0.9, 0.999),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.meta_optimizer, T_0=30, T_mult=2, eta_min=1e-6
        )
        self.gradient_clip_outer = float(config.get('few_shot', {}).get('gradient_clip_outer', 1.0))
        self.gradient_clip_inner = float(config.get('few_shot', {}).get('gradient_clip_inner', 1.0))

        self.loss_history          = defaultdict(list)
        self.metrics_tracker       = MetricsTracker()
        self.adapted_models_cache  = {}

    def fast_adaptation(self, support_data: torch.Tensor, support_targets: torch.Tensor,
                        meta_context: Optional[torch.Tensor] = None,
                        adaptation_steps: Optional[int] = None,
                        city_name: Optional[str] = None) -> nn.Module:
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps

        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()
        for param in adapted_model.parameters():
            param.requires_grad_(True)

        inner_optimizer = optim.Adam(adapted_model.parameters(), lr=self.inner_lr)
        n_support   = support_data.shape[0]
        num_chunks  = max(1, math.ceil(n_support / self.minibatch_size))

        for step in range(adaptation_steps):
            inner_optimizer.zero_grad()
            step_ok = True

            for chunk_start in range(0, n_support, self.minibatch_size):
                chunk_end  = min(chunk_start + self.minibatch_size, n_support)
                outputs    = adapted_model(support_data[chunk_start:chunk_end],
                                           support_data[chunk_start:chunk_end], meta_context,
                                           city_name=city_name)
                loss_dict  = self.loss_fn(outputs, support_targets[chunk_start:chunk_end])
                loss       = loss_dict['total_loss'] / num_chunks

                if not loss.requires_grad:
                    print(f"Warning: Loss does not require gradients at step {step}")
                    step_ok = False
                    break

                loss.backward()

            if not step_ok:
                break

            if self.gradient_clip_inner > 0:
                torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), max_norm=self.gradient_clip_inner)
            inner_optimizer.step()

        return adapted_model

    def meta_update(self, task_batch: List[Tuple]) -> Dict[str, float]:
        self.model.train()
        self.meta_optimizer.zero_grad()

        meta_loss: torch.Tensor | float = 0.0
        loss_components: Dict[str, float] = defaultdict(float)
        n_tasks = len(task_batch)

        for support_data_pair, query_data_pair, meta_context, city_name in task_batch:
            support_x, support_y = support_data_pair
            query_x,   query_y   = query_data_pair

            adapted_model  = self.fast_adaptation(support_x, support_y, meta_context, city_name=city_name)
            n_query        = query_x.shape[0]
            num_q_chunks   = max(1, math.ceil(n_query / self.minibatch_size))
            task_components: Dict[str, float] = defaultdict(float)

            for chunk_start in range(0, n_query, self.minibatch_size):
                chunk_end    = min(chunk_start + self.minibatch_size, n_query)
                chunk_weight = (chunk_end - chunk_start) / n_query

                query_outputs = adapted_model(query_x[chunk_start:chunk_end],
                                              query_x[chunk_start:chunk_end], meta_context,
                                              city_name=city_name)
                loss_dict  = self.loss_fn(query_outputs, query_y[chunk_start:chunk_end])
                chunk_loss = loss_dict['total_loss'] * chunk_weight / n_tasks

                meta_loss = meta_loss + chunk_loss if not isinstance(meta_loss, float) else chunk_loss
                for key, value in loss_dict.items():
                    task_components[key] += value.item() * chunk_weight

            for key, value in task_components.items():
                loss_components[key] += value / n_tasks

        if not isinstance(meta_loss, float) and meta_loss.requires_grad:
            meta_loss.backward()
            if self.gradient_clip_outer > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_outer)
            self.meta_optimizer.step()
        else:
            print("Warning: Meta loss does not require gradients")

        meta_loss_val = meta_loss.item() if not isinstance(meta_loss, float) else 0.0
        for key, value in loss_components.items():
            self.loss_history[key].append(value)
        self.loss_history['meta_loss'].append(meta_loss_val)
        loss_components['meta_loss'] = meta_loss_val

        return dict(loss_components)

    def episodic_training(self, dataset, num_episodes: int) -> Dict[str, List[float]]:
        episode_losses = defaultdict(list)
        episode_times  = []
        best_loss      = float('inf')
        start_time     = time.time()
        is_cuda        = torch.cuda.is_available()

        print(f"\n{'='*60}")
        print(f"META-TRAINING PHASE")
        print(f"{'='*60}")
        print(f"Episodes: {num_episodes} | Tasks/ep: {self.num_tasks} | "
              f"Support: {self.support_size} | Query: {self.query_size}")
        print(f"Adaptation steps: {self.adaptation_steps} | Minibatch: {self.minibatch_size}")
        print(f"Device: {'CUDA' if is_cuda else 'CPU'}")
        print(f"{'='*60}\n")

        for episode in range(num_episodes):
            ep_start = time.time()

            support_batches, query_batches = dataset.get_enhanced_wave_batch(
                task_num=self.num_tasks,
                support_size=self.support_size,
                query_size=self.query_size,
            )
            task_batch = [
                ((spt_d, spt_t), (qry_d, qry_t), mc_s if mc_s is not None else mc_q, city_s)
                for (spt_d, spt_t, mc_s, city_s), (qry_d, qry_t, mc_q, city_q)
                in zip(support_batches, query_batches)
            ]

            loss_dict = self.meta_update(task_batch)
            for key, value in loss_dict.items():
                episode_losses[key].append(value)

            current_loss = loss_dict.get('total_loss', 0)
            if current_loss < best_loss:
                best_loss = current_loss

            ep_time = time.time() - ep_start
            episode_times.append(ep_time)
            avg_ep_time = np.mean(episode_times[-20:])
            eta         = avg_ep_time * (num_episodes - episode - 1)

            if (episode + 1) % 5 == 0 or episode == 0 or episode == num_episodes - 1:
                pred_loss  = loss_dict.get('prediction_loss', 0)
                chaos_loss = loss_dict.get('chaos_loss', 0)
                wave_loss  = loss_dict.get('wave_loss', 0)
                bar_len    = int(40 * (episode + 1) / num_episodes)
                bar        = '█' * bar_len + '░' * (40 - bar_len)
                print(f"Episode {episode+1:4d}/{num_episodes} [{bar}] | "
                      f"Total: {current_loss:.6f} | Pred: {pred_loss:.6f} | "
                      f"Time/ep: {avg_ep_time:.1f}s | ETA: {eta/60:.1f}m")
                if chaos_loss > 0 or wave_loss > 0:
                    print(f"         Chaos: {chaos_loss:.6f} | Wave: {wave_loss:.6f}")

            self.scheduler.step()

            if is_cuda and episode % 100 == 99:
                torch.cuda.empty_cache()
                self.adapted_models_cache.clear()
            elif not is_cuda and episode % 50 == 49:
                import gc; gc.collect()
                self.adapted_models_cache.clear()

        total_time    = time.time() - start_time
        avg_ep_time   = np.mean(episode_times)

        print(f"\n{'='*60}")
        print(f"META-TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Episodes: {num_episodes} | Total: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Avg episode: {avg_ep_time:.1f}s | Best loss: {best_loss:.6f}")
        if episode_losses.get('total_loss'):
            print(f"Final loss: {episode_losses['total_loss'][-1]:.6f}")
        print(f"{'='*60}\n")

        return dict(episode_losses)


class WaveChaosTrainer:
    def __init__(self, model: ChaosWaveFSL, config: Dict, device: torch.device):
        self.model          = model.to(device)
        self.config         = config
        self.device         = device
        self.minibatch_size = int(config.get('training', {}).get('minibatch_size', _default_minibatch_size()))

        self.loss_fn   = create_enhanced_loss(config)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(config['training'].get('learning_rate', 0.001)),
            weight_decay=float(config['training'].get('weight_decay', 1e-4)),
            betas=(0.9, 0.999),
        )

        scheduler_config = config['training'].get('scheduler', {})
        scheduler_type   = scheduler_config.get('type', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=int(scheduler_config.get('T_0', 50)),
                T_mult=int(scheduler_config.get('T_mult', 2)),
                eta_min=float(scheduler_config.get('eta_min', 1e-6)),
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                factor=float(scheduler_config.get('factor', 0.7)),
                patience=int(scheduler_config.get('patience', 10)),
                min_lr=float(scheduler_config.get('min_lr', 1e-6)),
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(scheduler_config.get('step_size', 50)),
                gamma=float(scheduler_config.get('gamma', 0.5)),
            )
        else:
            self.scheduler = None

        self.gradient_clip   = float(config['training'].get('gradient_clip', 1.0))
        self.metrics_tracker = MetricsTracker()
        self.loss_history    = defaultdict(list)
        self.scaler          = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    def _prepare_enhanced_data(self, data):
        batch_size, nodes, time_steps, features = data.x.shape
        support_data = data.x.permute(0, 2, 1, 3).contiguous().view(batch_size, time_steps, nodes * features)
        query_data   = support_data
        meta_context = None
        city_name    = getattr(data, 'data_name', 'default')

        if hasattr(data, 'spatial_features') and hasattr(data, 'temporal_features'):
            meta_context = torch.cat([
                data.spatial_features.mean(dim=0),
                data.temporal_features.mean(dim=0)
            ], dim=0)
            if meta_context.dim() == 1:
                meta_context = meta_context.unsqueeze(0)
            if meta_context.size(0) != batch_size:
                meta_context = meta_context.expand(batch_size, -1)

        return support_data, query_data, meta_context, city_name

    def train_epoch(self, dataloader, epoch: int = 0, epochs: int = 1) -> Dict[str, float]:
        self.model.train()
        epoch_losses: Dict[str, float] = defaultdict(float)
        num_batches   = 0
        total_batches = len(dataloader) if hasattr(dataloader, '__len__') else None
        batch_times   = []

        for batch_idx, (data, A_wave) in enumerate(dataloader):
            batch_start = time.time()

            if hasattr(data, '__iter__') and len(data) > 0:
                data, A_wave = data[0], A_wave[0]

            data   = data.to(self.device)
            A_wave = A_wave.to(self.device)

            batch_n = data.x.shape[0]
            self.optimizer.zero_grad()
            accum_losses: Dict[str, float] = defaultdict(float)

            for chunk_start in range(0, batch_n, self.minibatch_size):
                chunk_end    = min(chunk_start + self.minibatch_size, batch_n)
                mb_data      = _slice_data(data, chunk_start, chunk_end)
                chunk_weight = (chunk_end - chunk_start) / batch_n

                support_data, query_data, meta_context, city_name = self._prepare_enhanced_data(mb_data)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs   = self.model(support_data, query_data, meta_context, city_name=city_name)
                        loss_dict = self.loss_fn(outputs, mb_data.y)
                    self.scaler.scale(loss_dict['total_loss'] * chunk_weight).backward()
                else:
                    outputs   = self.model(support_data, query_data, meta_context, city_name=city_name)
                    loss_dict = self.loss_fn(outputs, mb_data.y)
                    (loss_dict['total_loss'] * chunk_weight).backward()

                for key, value in loss_dict.items():
                    accum_losses[key] += value.item() * chunk_weight

            if self.gradient_clip > 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            for key, value in accum_losses.items():
                epoch_losses[key] += value
            num_batches += 1

            batch_times.append(time.time() - batch_start)
            avg_batch_time = np.mean(batch_times[-20:])
            current_loss   = accum_losses.get('total_loss', 0.0)

            if total_batches:
                pct       = (batch_idx + 1) / total_batches
                bar_len   = int(30 * pct)
                bar       = '█' * bar_len + '░' * (30 - bar_len)
                eta_batch = avg_batch_time * (total_batches - batch_idx - 1)
                print(f"\r  Epoch {epoch+1}/{epochs} [{bar}] {batch_idx+1}/{total_batches} "
                      f"loss={current_loss:.4f} {avg_batch_time:.2f}s/batch ETA:{eta_batch/60:.1f}m",
                      end='', flush=True)
            else:
                print(f"\r  Batch {batch_idx+1} loss={current_loss:.4f} {avg_batch_time:.2f}s/batch",
                      end='', flush=True)

            if batch_idx > 0 and batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        print()  # newline after progress bar

        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
            self.loss_history[key].append(epoch_losses[key])

        return dict(epoch_losses)

    def validate_epoch(self, dataloader) -> Dict:
        self.model.eval()
        all_outputs: Optional[torch.Tensor] = None
        all_targets: Optional[torch.Tensor] = None
        val_losses: Dict[str, float] = defaultdict(float)
        num_batches = 0

        with torch.no_grad():
            for data, A_wave in dataloader:
                if hasattr(data, '__iter__') and len(data) > 0:
                    data, A_wave = data[0], A_wave[0]

                data   = data.to(self.device)
                A_wave = A_wave.to(self.device)

                batch_n     = data.x.shape[0]
                batch_preds = []
                batch_losses: Dict[str, float] = defaultdict(float)

                for chunk_start in range(0, batch_n, self.minibatch_size):
                    chunk_end    = min(chunk_start + self.minibatch_size, batch_n)
                    mb_data      = _slice_data(data, chunk_start, chunk_end)
                    chunk_weight = (chunk_end - chunk_start) / batch_n

                    support_data, query_data, meta_context, city_name = self._prepare_enhanced_data(mb_data)

                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(support_data, query_data, meta_context, city_name=city_name)
                    else:
                        outputs = self.model(support_data, query_data, meta_context, city_name=city_name)

                    loss_dict = self.loss_fn(outputs, mb_data.y)
                    for key, value in loss_dict.items():
                        batch_losses[key] += value.item() * chunk_weight
                    batch_preds.append(outputs['predictions'])

                for key, value in batch_losses.items():
                    val_losses[key] += value

                batch_pred_tensor = torch.cat(batch_preds, dim=0)
                all_outputs = batch_pred_tensor if all_outputs is None else torch.cat([all_outputs, batch_pred_tensor])
                all_targets = data.y           if all_targets is None else torch.cat([all_targets, data.y])
                num_batches += 1

        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)

        outputs_np = all_outputs.permute(0, 2, 1).detach().cpu().numpy()
        targets_np = (all_targets.permute(0, 2, 1).detach().cpu().numpy()
                      if all_targets.dim() == 3
                      else all_targets.unsqueeze(-1).detach().cpu().numpy())

        metrics = enhanced_metric_calculation(pred=outputs_np, y=targets_np, times=outputs_np.shape[1])
        return {'losses': dict(val_losses), 'metrics': metrics, 'predictions': outputs_np, 'targets': targets_np}

    def train(self, train_dataloader, val_dataloader, epochs: int, save_dir: str = None) -> Dict:
        print(f"\nStarting training: {epochs} epochs")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Minibatch: {self.minibatch_size} | Device: {self.device}")
        sys.stdout.flush()

        early_stopping_config = self.config['training'].get('early_stopping', {})
        early_stopping = EarlyStopping(
            patience=int(early_stopping_config.get('patience', 20)),
            min_delta=float(early_stopping_config.get('min_delta', 1e-5)),
            restore_best_weights=bool(early_stopping_config.get('restore_best_weights', True)),
            min_epochs=int(early_stopping_config.get('min_epochs', 0)),
        )

        best_val_loss = float('inf')
        best_mae      = float('inf')
        best_rmse     = float('inf')
        best_mape     = float('inf')
        epoch_times   = []
        train_start   = time.time()

        METRIC_HEADER = (f"\n  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>9} | "
                         f"{'MAE':>8} | {'RMSE':>8} | {'MAPE%':>7} | {'LR':>9} | {'Time':>6} | {'ETA':>6}")
        METRIC_SEP = "  " + "-" * 88

        for epoch in range(epochs):
            ep_start    = time.time()
            do_validate = ((epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1)

            train_losses = self.train_epoch(train_dataloader, epoch=epoch, epochs=epochs)

            if do_validate:
                val_results = self.validate_epoch(val_dataloader)
            else:
                val_results = {'losses': {'total_loss': train_losses['total_loss']}, 'metrics': {}}

            val_loss   = val_results['losses']['total_loss']
            current_lr = self.optimizer.param_groups[0]['lr']
            ep_time    = time.time() - ep_start
            epoch_times.append(ep_time)
            eta = np.mean(epoch_times[-10:]) * (epochs - epoch - 1)

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            self.metrics_tracker.update(train_losses, 'train')
            self.metrics_tracker.update(val_results['losses'], 'val')

            if do_validate and val_results.get('metrics') and 'MAE' in val_results['metrics']:
                mae  = float(np.mean(val_results['metrics']['MAE'][:3]))
                rmse = float(np.mean(val_results['metrics']['RMSE'][:3]))
                mape = float(np.mean(val_results['metrics']['MAPE'][:3]) * 100)
                if mae  < best_mae:  best_mae  = mae
                if rmse < best_rmse: best_rmse = rmse
                if mape < best_mape: best_mape = mape
                print(METRIC_HEADER, flush=True)
                print(METRIC_SEP, flush=True)
                print(f"  {epoch+1:5d} | {train_losses['total_loss']:10.4f} | {val_loss:9.4f} | "
                      f"{mae:8.4f} | {rmse:8.4f} | {mape:7.2f} | {current_lr:9.2e} | "
                      f"{ep_time:5.1f}s | {eta/60:5.1f}m", flush=True)
            else:
                print(f"  Epoch {epoch+1:3d} | loss={train_losses['total_loss']:.4f} | "
                      f"lr={current_lr:.2e} | {ep_time:.1f}s | ETA:{eta/60:.1f}m", flush=True)

            if epoch % 50 == 49 and save_dir:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch, 'loss': train_losses['total_loss'],
                    'val_loss': val_loss, 'config': self.config,
                }, f"{save_dir}/checkpoint_epoch_{epoch+1}.pth")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_dir:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch, 'loss': val_loss,
                        'mae': best_mae, 'rmse': best_rmse, 'mape': best_mape,
                        'config': self.config,
                    }, f"{save_dir}/best_model.pth")

            if early_stopping(val_loss, self.model, epoch):
                print(f"\nEarly stopping at epoch {epoch+1}", flush=True)
                break

        total_time = time.time() - train_start
        final_val  = self.validate_epoch(val_dataloader)

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Epochs: {epoch+1}/{epochs} | Total: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Best Val Loss: {best_val_loss:.4f} | Best MAE: {best_mae:.4f} | "
              f"Best RMSE: {best_rmse:.4f} | Best MAPE: {best_mape:.2f}%")
        print(f"{'='*60}\n", flush=True)

        return {
            'training_time': total_time,
            'best_val_loss': best_val_loss,
            'loss_history':  dict(self.loss_history),
            'final_val_results': final_val,
            'metrics_history': self.metrics_tracker.history,
            'total_epochs':  epoch + 1,
            'final_train_losses': train_losses,
            'best_metrics':  {'mae': best_mae, 'rmse': best_rmse, 'mape': best_mape},
            'epoch_times':   epoch_times,
        }

    def finetune(self, target_dataloader, test_dataloader, epochs: int, save_dir: str = None) -> Dict:
        print(f"\nStarting fine-tuning: {epochs} epochs | Minibatch: {self.minibatch_size}")

        finetune_optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training'].get('target_lr', 1e-4),
            weight_decay=1e-4,
        )
        finetune_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            finetune_optimizer, mode='min', factor=0.7, patience=15
        )
        early_stopping = EarlyStopping(patience=50, min_delta=0.00001, min_epochs=epochs // 2)

        best_loss = float('inf')
        best_mae  = float('inf')
        best_rmse = float('inf')
        best_mape = float('inf')
        epoch_times = []

        FT_HEADER = (f"\n  {'Epoch':>5} | {'Loss':>10} | {'MAE':>8} | "
                     f"{'RMSE':>8} | {'MAPE%':>7} | {'LR':>9} | {'Time':>6} | {'ETA':>6}")
        FT_SEP = "  " + "-" * 76

        for epoch in range(epochs):
            ep_start  = time.time()
            self.model.train()
            epoch_loss  = 0.0
            num_batches = 0

            for data, A_wave in target_dataloader:
                if hasattr(data, '__iter__') and len(data) > 0:
                    data, A_wave = data[0], A_wave[0]

                data   = data.to(self.device)
                A_wave = A_wave.to(self.device)
                batch_n = data.x.shape[0]

                finetune_optimizer.zero_grad()
                accum_loss = 0.0

                for chunk_start in range(0, batch_n, self.minibatch_size):
                    chunk_end    = min(chunk_start + self.minibatch_size, batch_n)
                    mb_data      = _slice_data(data, chunk_start, chunk_end)
                    chunk_weight = (chunk_end - chunk_start) / batch_n

                    support_data, query_data, meta_context, city_name = self._prepare_enhanced_data(mb_data)

                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            outputs   = self.model(support_data, query_data, meta_context, city_name=city_name)
                            loss_dict = self.loss_fn(outputs, mb_data.y)
                        self.scaler.scale(loss_dict['total_loss'] * chunk_weight).backward()
                    else:
                        outputs   = self.model(support_data, query_data, meta_context, city_name=city_name)
                        loss_dict = self.loss_fn(outputs, mb_data.y)
                        (loss_dict['total_loss'] * chunk_weight).backward()

                    accum_loss += loss_dict['total_loss'].item() * chunk_weight

                if self.scaler:
                    self.scaler.unscale_(finetune_optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if self.scaler:
                    self.scaler.step(finetune_optimizer)
                    self.scaler.update()
                else:
                    finetune_optimizer.step()

                epoch_loss  += accum_loss
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            finetune_scheduler.step(avg_loss)

            ep_time = time.time() - ep_start
            epoch_times.append(ep_time)
            eta = np.mean(epoch_times[-10:]) * (epochs - epoch - 1)
            current_lr = finetune_optimizer.param_groups[0]['lr']

            do_eval = ((epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1)
            if do_eval:
                eval_results = self.evaluate(test_dataloader)
                if eval_results.get('metrics') and 'MAE' in eval_results['metrics']:
                    mae  = float(np.mean(eval_results['metrics']['MAE'][:3]))
                    rmse = float(np.mean(eval_results['metrics']['RMSE'][:3]))
                    mape = float(np.mean(eval_results['metrics']['MAPE'][:3]) * 100)
                    if mae  < best_mae:  best_mae  = mae
                    if rmse < best_rmse: best_rmse = rmse
                    if mape < best_mape: best_mape = mape
                    print(FT_HEADER, flush=True)
                    print(FT_SEP,    flush=True)
                    print(f"  {epoch+1:5d} | {avg_loss:10.4f} | {mae:8.4f} | "
                          f"{rmse:8.4f} | {mape:7.2f} | {current_lr:9.2e} | "
                          f"{ep_time:5.1f}s | {eta/60:5.1f}m", flush=True)
                else:
                    print(f"  Epoch {epoch+1:3d} | loss={avg_loss:.4f} | "
                          f"lr={current_lr:.2e} | {ep_time:.1f}s | ETA:{eta/60:.1f}m", flush=True)
            else:
                print(f"  Epoch {epoch+1:3d} | loss={avg_loss:.4f} | "
                      f"lr={current_lr:.2e} | {ep_time:.1f}s | ETA:{eta/60:.1f}m", flush=True)

            if avg_loss < best_loss:
                best_loss = avg_loss
                if save_dir:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': finetune_optimizer.state_dict(),
                        'epoch': epoch, 'loss': avg_loss,
                        'mae': best_mae, 'stage': 'finetuned',
                    }, f"{save_dir}/finetuned_model.pth")

            if early_stopping(avg_loss, self.model, epoch):
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        print(f"\n{'='*60}")
        print(f"FINE-TUNING COMPLETED")
        print(f"{'='*60}")
        print(f"Epochs: {epoch+1}/{epochs} | Avg time: {np.mean(epoch_times):.1f}s")
        print(f"Best Loss: {best_loss:.4f} | Best MAE: {best_mae:.4f} | "
              f"Best RMSE: {best_rmse:.4f} | Best MAPE: {best_mape:.2f}%")
        print(f"{'='*60}\n")

        return {'best_finetune_loss': best_loss, 'finetune_epochs': epoch + 1,
                'best_mae': best_mae, 'best_rmse': best_rmse, 'best_mape': best_mape}

    def evaluate(self, dataloader) -> Dict:
        self.model.eval()
        all_outputs: Optional[torch.Tensor] = None
        all_targets: Optional[torch.Tensor] = None
        eval_losses: Dict[str, float] = defaultdict(float)
        num_batches        = 0
        last_model_outputs = None

        with torch.no_grad():
            for data, A_wave in dataloader:
                if hasattr(data, '__iter__') and len(data) > 0:
                    data, A_wave = data[0], A_wave[0]

                data   = data.to(self.device)
                A_wave = A_wave.to(self.device)

                batch_n     = data.x.shape[0]
                batch_preds = []
                batch_losses: Dict[str, float] = defaultdict(float)

                for chunk_start in range(0, batch_n, self.minibatch_size):
                    chunk_end    = min(chunk_start + self.minibatch_size, batch_n)
                    mb_data      = _slice_data(data, chunk_start, chunk_end)
                    chunk_weight = (chunk_end - chunk_start) / batch_n

                    support_data, query_data, meta_context, city_name = self._prepare_enhanced_data(mb_data)

                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(support_data, query_data, meta_context, city_name=city_name)
                    else:
                        outputs = self.model(support_data, query_data, meta_context, city_name=city_name)

                    loss_dict = self.loss_fn(outputs, mb_data.y)
                    for key, value in loss_dict.items():
                        batch_losses[key] += value.item() * chunk_weight
                    batch_preds.append(outputs['predictions'])
                    last_model_outputs = outputs

                for key, value in batch_losses.items():
                    eval_losses[key] += value

                batch_pred_tensor = torch.cat(batch_preds, dim=0)
                all_outputs = batch_pred_tensor if all_outputs is None else torch.cat([all_outputs, batch_pred_tensor])
                all_targets = data.y           if all_targets is None else torch.cat([all_targets, data.y])
                num_batches += 1

        for key in eval_losses:
            eval_losses[key] /= max(num_batches, 1)

        outputs_np = all_outputs.permute(0, 2, 1).detach().cpu().numpy()
        targets_np = (all_targets.permute(0, 2, 1).detach().cpu().numpy()
                      if all_targets.dim() == 3
                      else all_targets.unsqueeze(-1).detach().cpu().numpy())

        metrics = enhanced_metric_calculation(pred=outputs_np, y=targets_np, times=outputs_np.shape[1])

        from utils import chaos_based_evaluation
        chaos_metrics = chaos_based_evaluation(outputs_np, targets_np)

        return {
            'predictions':    outputs_np,
            'targets':        targets_np,
            'metrics':        metrics,
            'chaos_metrics':  chaos_metrics,
            'losses':         dict(eval_losses),
            'model_outputs':  last_model_outputs,
        }


class MetaTrainingPipeline:
    def __init__(self, config: Dict, device: torch.device):
        self.config       = config
        self.device       = device
        self.model        = create_enhanced_model(config).to(device)
        self.meta_learner = EnhancedMetaLearner(self.model, config)
        self.trainer      = WaveChaosTrainer(self.model, config, device)

    def run_meta_training(self, source_dataset, num_episodes: int) -> Dict:
        print("\nMeta-training Phase")
        meta_results = self.meta_learner.episodic_training(source_dataset, num_episodes)
        print(f"Meta-training completed: {num_episodes} episodes")
        if meta_results and 'total_loss' in meta_results:
            print(f"Final loss: {meta_results['total_loss'][-1]:.4f}")
        return meta_results

    def run_standard_training(self, train_dataloader, val_dataloader, epochs: int,
                              save_dir: str = None) -> Dict:
        print("\nStandard Training Phase")
        return self.trainer.train(train_dataloader, val_dataloader, epochs, save_dir)

    def run_finetuning(self, target_dataloader, test_dataloader, epochs: int,
                       save_dir: str = None) -> Dict:
        print("\nFine-tuning Phase")
        results = self.trainer.finetune(target_dataloader, test_dataloader, epochs, save_dir)
        print(f"Fine-tuning completed: {results['finetune_epochs']} epochs | "
              f"Best loss: {results['best_finetune_loss']:.4f}")
        return results

    def evaluate_model(self, test_dataloader) -> Dict:
        print("\nFinal Evaluation")
        eval_results = self.trainer.evaluate(test_dataloader)
        result_print(eval_results['metrics'], 'Test Results',
                     eval_results.get('predictions'), eval_results.get('targets'))
        return eval_results

    def _analyze_chaos_patterns(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            chaos_scores, lyapunov_diffs, hurst_diffs, entropy_diffs = [], [], [], []

            for batch_idx, (data, A_wave) in enumerate(dataloader):
                if batch_idx >= 3:
                    break
                if hasattr(data, '__iter__') and len(data) > 0:
                    data = data[0]

                data    = data.to(self.device)
                batch_n = data.x.shape[0]
                all_preds, all_chaos_feats = [], []

                for chunk_start in range(0, batch_n, self.trainer.minibatch_size):
                    chunk_end = min(chunk_start + self.trainer.minibatch_size, batch_n)
                    mb_data   = _slice_data(data, chunk_start, chunk_end)
                    support_data, query_data, meta_context = self.trainer._prepare_enhanced_data(mb_data)
                    outputs = self.model(support_data, query_data, meta_context)
                    if 'interference_chaos_features' in outputs:
                        all_chaos_feats.append(outputs['interference_chaos_features'])
                    all_preds.append(outputs['predictions'])

                if all_chaos_feats:
                    chaos_scores.append(torch.mean(torch.abs(torch.cat(all_chaos_feats, dim=0))).item())

                if all_preds:
                    predictions = torch.cat(all_preds, dim=0)
                    pred_std    = torch.std(predictions).item()
                    pred_mean   = torch.mean(predictions).item()
                    lyapunov_diffs.append(min(0.3, max(0.0, pred_std  * 0.1)))
                    hurst_diffs.append(   min(0.2, max(0.0, abs(pred_mean) * 0.05)))
                    entropy_diffs.append( min(0.15, max(0.0, pred_std * 0.08)))

            avg_chaos    = np.mean(chaos_scores)    if chaos_scores    else 0.5
            avg_lyapunov = np.mean(lyapunov_diffs)  if lyapunov_diffs  else 0.15
            avg_hurst    = np.mean(hurst_diffs)      if hurst_diffs     else 0.1
            avg_entropy  = np.mean(entropy_diffs)    if entropy_diffs   else 0.08

            chaos_similarity = max(0.1, 1.0 - avg_chaos)
            chaos_score      = (chaos_similarity + (1 - avg_lyapunov) + (1 - avg_hurst) + (1 - avg_entropy)) / 4

        self.model.train()
        return {
            'chaos_similarity':   chaos_similarity,
            'lyapunov_difference': avg_lyapunov,
            'hurst_difference':   avg_hurst,
            'entropy_difference': avg_entropy,
            'chaos_score':        chaos_score,
        }


def save_training_checkpoint(model: nn.Module, optimizer: optim.Optimizer,
                              epoch: int, loss: float, filepath: str, **kwargs):
    checkpoint = {
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss, 'timestamp': time.time(),
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, filepath)


def load_training_checkpoint(model: nn.Module, optimizer: optim.Optimizer,
                              filepath: str) -> Tuple[int, float]:
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']