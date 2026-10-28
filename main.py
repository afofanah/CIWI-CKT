import os
import argparse
import time
import torch
import numpy as np
from typing import Dict, Tuple, Any

from datasets import DataManager
from train import MetaTrainingPipeline
from utils import (
    load_config, save_config, set_random_seed, create_experiment_dir,
    get_device, validate_config, count_parameters,
    create_comprehensive_plots, save_predictions_and_targets
)


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    if seconds < 86400:
        return f"{seconds/3600:.1f}h"
    return f"{seconds/86400:.1f}d"


def parse_arguments():
    parser = argparse.ArgumentParser(description='CWIST-KT: Chaotic Wave-Aware Interference Spatio-Temporal with Knowledge Transfer based on Few-Shot Learning for Traffic Flow Forecasting')
    
    parser.add_argument('--config', default='config.yaml', type=str, help='Path to configuration file')
    parser.add_argument('--test_dataset', default='shenzhen', type=str, help='Target dataset for testing')
    parser.add_argument('--target_days', default=3, type=int, help='Number of target days for few-shot learning')
    parser.add_argument('--meta_epochs', default=100, type=int, help='Number of meta-learning episodes')
    parser.add_argument('--source_epochs', default=200, type=int, help='Number of source domain training epochs')
    parser.add_argument('--target_epochs', default=300, type=int, help='Number of target domain fine-tuning epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='Training batch size')
    parser.add_argument('--test_batch_size', default=1, type=int, help='Testing batch size')
    parser.add_argument('--meta_batch_size', default=1, type=int, help='Number of tasks per meta-batch')
    parser.add_argument('--gradient_accumulation_steps', default=5, type=int, help='Accumulate gradients over N steps')
    parser.add_argument('--hidden_dim', default=64, type=int, help='Hidden dimension size')
    parser.add_argument('--wave_components', default=12, type=int, help='Number of wave components')
    parser.add_argument('--frequency_bands', default=16, type=int, help='Number of frequency bands')
    parser.add_argument('--chaos_dim', default=32, type=int, help='Chaos dimension size')
    parser.add_argument('--meta_dim', default=16, type=int, help='Meta-learning dimension size')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Base learning rate')
    parser.add_argument('--meta_lr', default=0.0005, type=float, help='Meta-learning rate')
    parser.add_argument('--inner_lr', default=0.01, type=float, help='Inner loop learning rate for meta-learning')
    parser.add_argument('--enable_meta_learning', default=True, type=bool, help='Enable meta-learning phase')
    parser.add_argument('--enable_standard_training', default=True, type=bool, help='Enable standard training phase')
    parser.add_argument('--enable_finetuning', default=True, type=bool, help='Enable fine-tuning phase')
    parser.add_argument('--support_size', default=5, type=int, help='Support set size for few-shot learning')
    parser.add_argument('--query_size', default=10, type=int, help='Query set size for few-shot learning')
    parser.add_argument('--n_way', default=5, type=int, help='Number of traffic pattern classes per episode')
    parser.add_argument('--k_shot', default=3, type=int, help='Number of support examples per class')
    parser.add_argument('--adaptation_steps', default=3, type=int, help='Number of adaptation steps in meta-learning')
    parser.add_argument('--use_lyapunov_attention', default=True, type=bool, help='Enable Lyapunov-guided attention mechanism')
    parser.add_argument('--use_recurrence_plot', default=False, type=bool, help='Convert time series to recurrence plots')
    parser.add_argument('--chaos_embedding_type', default='phase_space', choices=['phase_space', 'recurrence', 'wavelet_chaos', 'hybrid'], help='Type of chaos-aware embedding')
    parser.add_argument('--delay_tau', default=5, type=int, help='Time delay for phase space reconstruction')
    parser.add_argument('--embedding_dim_chaos', default=20, type=int, help='Embedding dimension for chaos reconstruction')
    parser.add_argument('--lyapunov_loss_weight', default=0.1, type=float, help='Weight for Lyapunov consistency in loss')
    parser.add_argument('--chaotic_invariant_transfer', default=True, type=bool, help='Transfer knowledge via chaotic invariants')
    parser.add_argument('--loss_type', default='hybrid', choices=['mse', 'mae', 'quantile', 'hybrid', 'chaos_aware'], help='Loss function type')
    parser.add_argument('--quantile_levels', default=[0.1, 0.5, 0.9], type=float, nargs='+', help='Quantile levels for probabilistic forecasting')
    parser.add_argument('--curriculum_chaos', default=False, type=bool, help='Use curriculum learning from low to high chaos regimes')
    parser.add_argument('--chaos_thresholds', default=[0.1, 0.5, 1.0], type=float, nargs='+', help='Lyapunov exponent thresholds for curriculum stages')
    parser.add_argument('--experiment_name', default='enhanced_wave_traffic', type=str, help='Name of the experiment')
    parser.add_argument('--save_dir', default='./experiments', type=str, help='Directory to save results')
    parser.add_argument('--device', default='auto', type=str, help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--resume_from', default=None, type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', default=False, type=bool, help='Only run evaluation')
    
    _default_minibatch = 2048 if torch.cuda.is_available() else 4096
    parser.add_argument('--minibatch_size', type=int, default=_default_minibatch, help='Mini-batch size for inner-loop gradient computation')
    
    return parser.parse_args()



def setup_experiment(args) -> Tuple[Dict, str, torch.device]:
    set_random_seed(args.seed)
    device = get_device(args.device)

    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    if not validate_config(config):
        raise ValueError("Invalid configuration file")

    config['model']['hidden_dim']      = args.hidden_dim
    config['model']['wave_components'] = args.wave_components
    config['model']['frequency_bands'] = args.frequency_bands
    config['model']['chaos_dim']       = args.chaos_dim
    config['model']['meta_dim']        = args.meta_dim
    config['model']['meta_lr']         = args.meta_lr

    config['training']['learning_rate']   = args.learning_rate
    config['training']['batch_size']      = args.batch_size
    config['training']['test_batch_size'] = args.test_batch_size
    config['training']['minibatch_size']  = args.minibatch_size

    config['task']['batch_size']      = args.batch_size
    config['task']['test_batch_size'] = args.test_batch_size

    if 'few_shot' not in config:
        config['few_shot'] = {}
    config['few_shot']['inner_lr']         = args.inner_lr
    config['few_shot']['support_size']     = args.support_size
    config['few_shot']['query_size']       = args.query_size
    config['few_shot']['adaptation_steps'] = args.adaptation_steps

    exp_dir = create_experiment_dir(args.save_dir, f"{args.experiment_name}_{args.test_dataset}")
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)

    save_config(config, os.path.join(exp_dir, 'config.yaml'))
    save_config(vars(args), os.path.join(exp_dir, 'args.yaml'))

    print(f"Experiment dir : {exp_dir}")
    print(f"Device         : {device}")
    print(f"Test dataset   : {args.test_dataset}")
    print(f"Seed           : {args.seed}")
    print(f"Minibatch      : {args.minibatch_size}")

    return config, exp_dir, device


def create_pipeline(config: Dict, device: torch.device) -> MetaTrainingPipeline:
    """
    MetaTrainingPipeline creates the model internally.
    We do NOT create a separate standalone model to avoid double initialisation
    and double parameter count.
    """
    print("\nCreating model and pipeline...")
    pipeline     = MetaTrainingPipeline(config, device)
    total_params = count_parameters(pipeline.model)
    print(f"Parameters     : {total_params:,}")
    print(f"Wave components: {config['model']['wave_components']}")
    print(f"Frequency bands: {config['model']['frequency_bands']}")
    print(f"Hidden dim     : {config['model']['hidden_dim']}")
    print(f"Chaos dim      : {config['model']['chaos_dim']}")
    print(f"Meta dim       : {config['model']['meta_dim']}")
    print(f"City bank      : {list(config['data'].get('data_keys', []))}")
    return pipeline


def create_data_loaders(config: Dict, args) -> Tuple[Any, Dict]:
    print(f"\nCreating data loaders for: {args.test_dataset}")
    data_manager = DataManager(config['data'], config['task'])
    dataloaders  = data_manager.create_all_dataloaders(
        test_data=args.test_dataset,
        target_days=args.target_days,
        enable_augmentation=config.get('training', {}).get('enable_augmentation', False),
        minibatch_size=args.minibatch_size,
    )
    source_dataset = data_manager.get_meta_learning_data(
        test_data=args.test_dataset,
        support_size=args.support_size,
        query_size=args.query_size,
        minibatch_size=args.minibatch_size,
    )
    print(f"Target days    : {args.target_days}")
    for key, info in data_manager.get_dataset_statistics().items():
        print(f"  {key}: {len(info)} datasets")
    return source_dataset, dataloaders


def run_meta_learning_phase(pipeline: MetaTrainingPipeline, source_dataset,
                            args, exp_dir: str) -> Dict:
    if not args.enable_meta_learning:
        print("\nSkipping meta-learning phase (disabled)")
        return {}

    print(f"\n{'='*70}")
    print("META-LEARNING PHASE")
    print(f"{'='*70}")
    print(f"Episodes: {args.meta_epochs} | Support: {args.support_size} | "
          f"Query: {args.query_size} | Adapt steps: {args.adaptation_steps}")

    t0           = time.time()
    meta_results = pipeline.run_meta_training(source_dataset, args.meta_epochs)
    elapsed      = time.time() - t0

    print(f"Completed in {_fmt_time(elapsed)}")
    if meta_results and 'total_loss' in meta_results:
        print(f"Final meta loss: {meta_results['total_loss'][-1]:.6f}")

    torch.save({
        'model_state_dict': pipeline.model.state_dict(),
        'meta_results':     meta_results,
        'args':             vars(args),
    }, os.path.join(exp_dir, 'models', 'meta_trained_model.pth'))

    return meta_results


def run_standard_training_phase(pipeline: MetaTrainingPipeline, dataloaders: Dict,
                                args, exp_dir: str) -> Dict:
    if not args.enable_standard_training:
        print("\nSkipping standard training phase (disabled)")
        return {}

    print(f"\n{'='*70}")
    print("STANDARD TRAINING PHASE")
    print(f"{'='*70}")

    t0      = time.time()
    results = pipeline.run_standard_training(
        dataloaders['source'],
        dataloaders['validation'],
        args.source_epochs,
        os.path.join(exp_dir, 'models'),
    )
    print(f"Completed in {_fmt_time(time.time() - t0)}")
    return results


def run_finetuning_phase(pipeline: MetaTrainingPipeline, dataloaders: Dict,
                         args, exp_dir: str) -> Dict:
    if not args.enable_finetuning:
        print("\nSkipping fine-tuning phase (disabled)")
        return {}

    print(f"\n{'='*70}")
    print("FINE-TUNING PHASE")
    print(f"{'='*70}")
    print(f"Epochs: {args.target_epochs} | Target days: {args.target_days}")

    t0      = time.time()
    results = pipeline.run_finetuning(
        dataloaders['target'],
        dataloaders['test'],
        args.target_epochs,
        os.path.join(exp_dir, 'models'),
    )
    elapsed = time.time() - t0
    print(f"Completed in {_fmt_time(elapsed)} | "
          f"Epochs: {results['finetune_epochs']} | "
          f"Best loss: {results['best_finetune_loss']:.4f}")
    return results


def run_evaluation_phase(pipeline: MetaTrainingPipeline, dataloaders: Dict,
                         exp_dir: str) -> Dict:
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")

    t0           = time.time()
    eval_results = pipeline.evaluate_model(dataloaders['test'])
    print(f"Completed in {_fmt_time(time.time() - t0)}")

    if eval_results.get('metrics'):
        m = eval_results['metrics']
        n = min(6, len(m.get('MAE', [])))
        print(f"\n  {'Step':>4} | {'MAE':>8} | {'RMSE':>8} | {'MAPE%':>7}")
        print(f"  {'-'*44}")
        for i in range(n):
            print(f"  {i+1:4d} | {m['MAE'][i]:8.4f} | {m['RMSE'][i]:8.4f} | "
                  f"{m['MAPE'][i]*100:7.2f}")
        print(f"  {'Avg':>4} | {np.mean(m['MAE'][:n]):8.4f} | {np.mean(m['RMSE'][:n]):8.4f} | "
              f"{np.mean(m['MAPE'][:n])*100:7.2f}")

    if eval_results.get('chaos_metrics', {}).get('chaos_score', 0) > 0:
        c = eval_results['chaos_metrics']
        print(f"\n  Chaos score   : {c['chaos_score']:.4f}")
        print(f"  Lyapunov diff : {c['lyapunov_difference']:.4f}")
        print(f"  Hurst diff    : {c['hurst_difference']:.4f}")
        print(f"  Entropy diff  : {c['entropy_difference']:.4f}")

    return eval_results


def save_experiment_results(exp_dir: str, args, config: Dict,
                            meta_results: Dict, training_results: Dict,
                            finetune_results: Dict, eval_results: Dict,
                            total_params: int):
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    results_summary = {
        'experiment_info': {
            'name':         args.experiment_name,
            'test_dataset': args.test_dataset,
            'target_days':  args.target_days,
            'timestamp':    time.strftime("%Y-%m-%d %H:%M:%S"),
            'device':       str(args.device),
            'seed':         args.seed,
        },
        'model_config': {
            'hidden_dim':       args.hidden_dim,
            'wave_components':  args.wave_components,
            'frequency_bands':  args.frequency_bands,
            'chaos_dim':        args.chaos_dim,
            'meta_dim':         args.meta_dim,
            'total_parameters': total_params,
        },
        'training_config': {
            'meta_epochs':      args.meta_epochs,
            'source_epochs':    args.source_epochs,
            'target_epochs':    args.target_epochs,
            'learning_rate':    args.learning_rate,
            'meta_lr':          args.meta_lr,
            'inner_lr':         args.inner_lr,
            'support_size':     args.support_size,
            'query_size':       args.query_size,
            'adaptation_steps': args.adaptation_steps,
            'minibatch_size':   args.minibatch_size,
        },
    }

    if eval_results and 'metrics' in eval_results:
        m = eval_results['metrics']
        results_summary['performance_summary'] = {
            'avg_mae_6_steps':  float(np.mean(m['MAE'][:6])),
            'avg_rmse_6_steps': float(np.mean(m['RMSE'][:6])),
            'avg_mape_6_steps': float(np.mean(m['MAPE'][:6]) * 100),
            'best_mae':         float(np.min(m['MAE'])),
            'best_rmse':        float(np.min(m['RMSE'])),
            'best_mape':        float(np.min(m['MAPE']) * 100)
        }

    results_file = os.path.join(exp_dir, 'experiment_results.yaml')
    save_config(results_summary, results_file)

    if eval_results and 'predictions' in eval_results:
        save_predictions_and_targets(
            eval_results['predictions'],
            eval_results['targets'],
            eval_results['metrics'],
            os.path.join(exp_dir, 'predictions'),
        )

    # Only generate plots when training loss history exists (not in eval_only mode)
    train_losses = training_results.get('loss_history', {}).get('total_loss', [])
    if eval_results and train_losses:
        val_losses = None
        fvr = training_results.get('final_val_results', {})
        if fvr.get('losses'):
            val_losses = [fvr['losses']['total_loss']]
        create_comprehensive_plots(
            eval_results.get('predictions', np.array([])),
            eval_results.get('targets',     np.array([])),
            eval_results.get('metrics',     {}),
            train_losses,
            val_losses,
            os.path.join(exp_dir, 'plots'),
        )

    summary_file = os.path.join(exp_dir, 'experiment_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"{'='*70}\nCIWI-CKT EXPERIMENT SUMMARY\n{'='*70}\n\n")
        f.write(f"Experiment  : {args.experiment_name}\n")
        f.write(f"Dataset     : {args.test_dataset}\n")
        f.write(f"Target Days : {args.target_days}\n")
        f.write(f"Timestamp   : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"MODEL\n{'-'*40}\n")
        f.write(f"Hidden Dim      : {args.hidden_dim}\n")
        f.write(f"Wave Components : {args.wave_components}\n")
        f.write(f"Frequency Bands : {args.frequency_bands}\n")
        f.write(f"Chaos Dim       : {args.chaos_dim}\n")
        f.write(f"Meta Dim        : {args.meta_dim}\n")
        f.write(f"Parameters      : {total_params:,}\n\n")
        f.write(f"TRAINING\n{'-'*40}\n")
        f.write(f"Meta Episodes   : {args.meta_epochs}\n")
        f.write(f"Source Epochs   : {args.source_epochs}\n")
        f.write(f"Target Epochs   : {args.target_epochs}\n")
        f.write(f"Learning Rate   : {args.learning_rate}\n")
        f.write(f"Meta LR         : {args.meta_lr}\n")
        f.write(f"Inner LR        : {args.inner_lr}\n")
        f.write(f"Minibatch Size  : {args.minibatch_size}\n\n")
        if 'performance_summary' in results_summary:
            p = results_summary['performance_summary']
            f.write(f"PERFORMANCE (avg first 6 steps)\n{'-'*40}\n")
            f.write(f"MAE  : {p['avg_mae_6_steps']:.4f}\n")
            f.write(f"RMSE : {p['avg_rmse_6_steps']:.4f}\n")
            f.write(f"MAPE : {p['avg_mape_6_steps']:.2f}%\n")

    print(f"Results : {results_file}")
    print(f"Summary : {summary_file}")


def main():
    print(f"{'='*70}")
    print("CIWI-CKT: Chaos-Informed Wave Interference + Cross-City Knowledge Transfer")
    print(f"{'='*70}")

    args                        = parse_arguments()
    config, exp_dir, device     = setup_experiment(args)
    pipeline                    = create_pipeline(config, device)
    source_dataset, dataloaders = create_data_loaders(config, args)

    total_params = count_parameters(pipeline.model)
    t0           = time.time()

    meta_results     = {}
    training_results = {}
    finetune_results = {}

    if args.eval_only:
        print("\nEval-only mode.")
        if args.resume_from:
            checkpoint = torch.load(args.resume_from, map_location=device)
            pipeline.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded weights from {args.resume_from}")
    else:
        meta_results     = run_meta_learning_phase(pipeline, source_dataset, args, exp_dir)
        training_results = run_standard_training_phase(pipeline, dataloaders, args, exp_dir)
        finetune_results = run_finetuning_phase(pipeline, dataloaders, args, exp_dir)

    eval_results = run_evaluation_phase(pipeline, dataloaders, exp_dir)

    save_experiment_results(
        exp_dir, args, config,
        meta_results, training_results, finetune_results, eval_results,
        total_params,
    )

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"COMPLETED  |  Total time: {_fmt_time(total_time)}  |  Results: {exp_dir}")
    if eval_results and 'metrics' in eval_results:
        m = eval_results['metrics']
        print(f"MAE: {np.mean(m['MAE'][:6]):.4f}  "
              f"RMSE: {np.mean(m['RMSE'][:6]):.4f}  "
              f"MAPE: {np.mean(m['MAPE'][:6])*100:.2f}%  "
)
    print(f"{'='*70}")


if __name__ == '__main__':
    main()



