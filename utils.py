import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import json
import yaml
import time
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_model_memory(model: torch.nn.Module) -> Dict[str, float]:
    param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return {
        'param_count':    sum(p.nelement() for p in model.parameters()),
        'buffer_count':   sum(b.nelement() for b in model.buffers()),
        'param_size_mb':  param_size  / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024,
        'total_size_mb':  (param_size + buffer_size) / 1024 / 1024,
    }


def chaos_based_evaluation(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    from model.model import ChaosAnalyzer
    chaos_analyzer = ChaosAnalyzer()
    pred_chaos   = chaos_analyzer.extract_chaos_features(predictions.flatten())
    target_chaos = chaos_analyzer.extract_chaos_features(targets.flatten())
    similarity   = 1.0 - np.mean(np.abs(pred_chaos - target_chaos))
    return {
        'chaos_similarity':                float(similarity),
        'lyapunov_difference':             float(abs(pred_chaos[0] - target_chaos[0])),
        'hurst_difference':                float(abs(pred_chaos[1] - target_chaos[1])),
        'entropy_difference':              float(abs(pred_chaos[2] - target_chaos[2])),
        'correlation_dimension_difference': float(abs(pred_chaos[3] - target_chaos[3])),
        'chaos_score': float(
            1.0 - (abs(pred_chaos[0] - target_chaos[0]) + abs(pred_chaos[1] - target_chaos[1])
                   + abs(pred_chaos[2] - target_chaos[2]) + abs(pred_chaos[3] - target_chaos[3])) / 4.0
        ),
    }


def enhanced_metric_calculation(pred: np.ndarray, y: np.ndarray, times: int) -> Dict[str, np.ndarray]:
    result = {k: np.zeros(times) for k in ['MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'WAPE']}

    def _mape(p, t):
        m = t != 0
        return float(np.mean(np.abs(t[m] - p[m]) / np.abs(t[m]))) if m.any() else 0.0

    def _smape(p, t):
        d = (np.abs(p) + np.abs(t)) / 2
        m = d != 0
        return float(np.mean(np.abs(p[m] - t[m]) / d[m])) if m.any() else 0.0

    def _wape(p, t):
        s = np.sum(np.abs(t))
        return float(np.sum(np.abs(p - t)) / s) if s != 0 else 0.0

    # shape alignment
    if pred.shape != y.shape:
        n = min(pred.shape[0], y.shape[0])
        if pred.ndim == 3 and y.ndim == 3:
            c = min(pred.shape[2], y.shape[2])
            pred, y = pred[:n, :, :c], y[:n, :, :c]
        elif pred.ndim == 2 and y.ndim == 2:
            pred, y = pred[:n], y[:n]
        else:
            if pred.ndim == 2: pred = pred[:, :, np.newaxis]
            if y.ndim   == 2: y    = y[:, :, np.newaxis]

    max_t = min(times, 12)
    ndim  = pred.ndim

    for i in range(max_t if ndim == 3 else min(max_t, pred.shape[1] if ndim >= 2 else 1)):
        yi = (y[:, i, :].flatten() if ndim == 3 else y[:, i])
        pi = (pred[:, i, :].flatten() if ndim == 3 else pred[:, i])
        mn = min(len(yi), len(pi))
        yi, pi = yi[:mn], pi[:mn]
        mask = np.isfinite(yi) & np.isfinite(pi)
        yi, pi = yi[mask], pi[mask]
        if len(yi) == 0:
            continue
        result['MSE'][i]   = mean_squared_error(yi, pi)
        result['RMSE'][i]  = np.sqrt(result['MSE'][i])
        result['MAE'][i]   = mean_absolute_error(yi, pi)
        result['MAPE'][i]  = _mape(pi, yi)
        result['SMAPE'][i] = _smape(pi, yi)
        result['WAPE'][i]  = _wape(pi, yi)

    return result


def result_print(result: Dict, info_name: str = 'Evaluate',
                 predictions: np.ndarray = None, targets: np.ndarray = None):
    n = min(6, len(result['MAE']))
    print(f"========== {info_name} results ==========")
    print(" MAE: " + " / ".join(f"{result['MAE'][i]:.3f}"       for i in range(n)))
    print("MAPE: " + " / ".join(f"{result['MAPE'][i]*100:.3f}"  for i in range(n)))
    print("RMSE: " + " / ".join(f"{result['RMSE'][i]:.3f}"      for i in range(n)))
    print(f"Average - MAE: {np.mean(result['MAE'][:n]):.3f}, "
          f"MAPE: {np.mean(result['MAPE'][:n])*100:.3f}%, "
          f"RMSE: {np.mean(result['RMSE'][:n]):.3f}, "
)
    if predictions is not None and targets is not None:
        cm = chaos_based_evaluation(predictions, targets)
        if cm['chaos_score'] > 0:
            print("--- Chaos Analysis ---")
            for k, v in cm.items():
                print(f"  {k}: {v:.3f}")
    print("---------------------------------------")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str = 'config.yaml') -> Dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict, filepath: str):
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    if filepath.endswith(('.yaml', '.yml')):
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(config, f)


def set_random_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    import datetime
    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{ts}")
    os.makedirs(exp_dir, exist_ok=True)
    for sub in ['plots', 'checkpoints', 'logs', 'models', 'data', 'predictions']:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    print(f"Created experiment directory: {exp_dir}")
    return exp_dir


def log_metrics(metrics: Dict, epoch: int, stage: str,
                log_file: str = None, verbose: bool = True):
    msg = f"Epoch {epoch:4d} [{stage:^12}] - " + ", ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
    if verbose:
        print(msg)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(msg + "\n")


def get_device(device_str: str = 'auto') -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def validate_config(config: Dict) -> bool:
    for k in ['data', 'task', 'model']:
        if k not in config:
            return False
    if 'data_keys' not in config['data']:
        return False
    for k in ['his_num', 'pred_num', 'batch_size']:
        if k not in config['task']:
            return False
    for k in ['hidden_dim', 'output_dim']:
        if k not in config['model']:
            return False
    return True


def save_checkpoint(model, optimizer, epoch, loss, filepath, **kwargs):
    ck = {'epoch': epoch, 'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': loss, 'timestamp': time.time()}
    ck.update(kwargs)
    torch.save(ck, filepath)


def load_checkpoint(model, optimizer, filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    ck = torch.load(filepath)
    model.load_state_dict(ck['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ck['optimizer_state_dict'])
    return ck['epoch'], ck['loss']


def save_predictions_and_targets(predictions, targets, metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(save_dir, 'targets.npy'),     targets)
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in metrics.items()}, f, indent=2)
    print(f"Saved predictions and metrics to: {save_dir}")


# ---------------------------------------------------------------------------
# EarlyStopping / MetricsTracker
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.0001, restore_best_weights=True,
                 min_epochs=0, mode='min'):
        self.patience             = patience
        self.min_delta            = min_delta
        self.restore_best_weights = restore_best_weights
        self.min_epochs           = min_epochs
        self.mode                 = mode
        self.best_score           = float('inf') if mode == 'min' else float('-inf')
        self.counter              = 0
        self.best_weights         = None
        self.current_epoch        = 0

    def __call__(self, score, model, epoch=None):
        self.current_epoch = epoch if epoch is not None else self.current_epoch + 1
        if self.current_epoch < self.min_epochs:
            if self._is_better(score):
                self.best_score   = score
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()} \
                                    if self.restore_best_weights else None
            return False
        if self._is_better(score):
            self.best_score   = score
            self.counter      = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()} \
                                if self.restore_best_weights else None
        else:
            self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                dev = next(model.parameters()).device
                model.load_state_dict({k: v.to(dev) for k, v in self.best_weights.items()})
            return True
        return False

    def _is_better(self, score):
        return (score < self.best_score - self.min_delta if self.mode == 'min'
                else score > self.best_score + self.min_delta)


class MetricsTracker:
    def __init__(self):
        self.metrics = {}
        self.history = {}

    def update(self, new_metrics, stage='train'):
        self.history.setdefault(stage, {})
        for k, v in new_metrics.items():
            self.history[stage].setdefault(k, []).append(v)
        self.metrics[stage] = new_metrics

    def get_best(self, metric, stage='train', mode='min'):
        vals = self.history.get(stage, {}).get(metric, [])
        if not vals:
            return float('inf') if mode == 'min' else float('-inf')
        return min(vals) if mode == 'min' else max(vals)

    def get_last(self, metric, stage='train'):
        vals = self.history.get(stage, {}).get(metric, [0.0])
        return vals[-1]

    def get_average(self, metric, stage='train', last_n=None):
        vals = self.history.get(stage, {}).get(metric, [0.0])
        return float(np.mean(vals[-last_n:] if last_n else vals))

    def save_metrics(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_metrics(self, filepath):
        with open(filepath, 'r') as f:
            self.history = json.load(f)


# ---------------------------------------------------------------------------
# Research-question visualisations
# ---------------------------------------------------------------------------

def _mpl():
    """Lazy import of matplotlib so the module loads even without it."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Global font sizes — applied to every figure produced by this module
    plt.rcParams.update({
        'font.size':          15,
        'axes.titlesize':     16,
        'axes.labelsize':     15,
        'xtick.labelsize':    13,
        'ytick.labelsize':    13,
        'legend.fontsize':    13,
        'figure.titlesize':   17,
    })

    return plt, gridspec


_PALETTE = {
    'blue':   '#2563EB',
    'orange': '#EA580C',
    'green':  '#16A34A',
    'red':    '#DC2626',
    'purple': '#7C3AED',
    'teal':   '#0891B2',
    'gray':   '#6B7280',
    'light':  '#F1F5F9',
    'bg':     '#FFFFFF',
    'text':   '#1E293B',
    'grid':   '#E2E8F0',
    'spine':  '#CBD5E1',
}
_STEP_LABELS = ['5 min', '10 min', '15 min', '20 min', '25 min', '30 min',
                '35 min', '40 min', '45 min', '50 min', '55 min', '60 min']


def _style(ax, plt, title='', xlabel='', ylabel='', grid=True):
    ax.set_facecolor('white')
    ax.tick_params(colors=_PALETTE['text'], labelsize=13)
    for spine in ax.spines.values():
        spine.set_edgecolor(_PALETTE.get('spine', '#CBD5E1'))
    if title:  ax.set_title(title, color=_PALETTE['text'], fontweight='bold', pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=_PALETTE['text'])
    if ylabel: ax.set_ylabel(ylabel, color=_PALETTE['text'])
    if grid:
        ax.grid(True, color=_PALETTE.get('grid', '#E2E8F0'), linewidth=0.8)
        ax.set_axisbelow(True)


# RQ1 — Prediction accuracy across time horizons
def plot_rq1_horizon_metrics(metrics: Dict[str, np.ndarray], save_dir: str):
    """
    RQ1: Does CIWI-CKT maintain accuracy across all prediction horizons?
    Shows MAE, RMSE, and MAPE per step as a grouped bar chart.
    """
    plt, gs = _mpl()
    n  = min(12, len(metrics.get('MAE', [])))
    if n == 0:
        return
    steps  = np.arange(n)
    labels = _STEP_LABELS[:n]
    mae    = metrics['MAE'][:n]
    rmse   = metrics['RMSE'][:n]
    mape   = metrics['MAPE'][:n] * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('white')
    fig.suptitle('RQ1 — Prediction Accuracy Across Time Horizons',
                 color=_PALETTE['text'], fontweight='bold', y=1.02)

    for ax, vals, color, ylabel, title in zip(
        axes,
        [mae, rmse, mape],
        [_PALETTE['blue'], _PALETTE['orange'], _PALETTE['green']],
        ['MAE (mph)', 'RMSE (mph)', 'MAPE (%)'],
        ['Mean Absolute Error', 'Root Mean Squared Error', 'Mean Absolute % Error'],
    ):
        bars = ax.bar(steps, vals, color=color, alpha=0.85, width=0.6, edgecolor='#CBD5E1')
        # annotate every bar
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    f'{v:.2f}', ha='center', va='bottom', color=_PALETTE['text'])
        # trend line — guard against NaN/Inf/constant arrays
        _v = np.array(vals, dtype=float)
        if np.all(np.isfinite(_v)) and len(np.unique(_v)) > 1:
            z = np.polyfit(steps, _v, 1)
            ax.plot(steps, np.poly1d(z)(steps), '--', color='white', lw=1.2, alpha=0.6, label='Trend')
        ax.set_xticks(steps)
        ax.set_xticklabels(labels, rotation=35, ha='right')
        _style(ax, plt, title=title, ylabel=ylabel)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rq1_horizon_metrics.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Plot] RQ1 saved → rq1_horizon_metrics.png")


# RQ2 — Training convergence and LR schedule effect
def plot_rq2_convergence(train_losses: List[float], val_losses: List[float],
                         lr_history: List[float], save_dir: str):
    """
    RQ2: Does meta-pretraining accelerate convergence in standard training?
    Shows train/val loss curves with LR schedule overlaid.
    """
    plt, _ = _mpl()
    if not train_losses:
        return

    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')

    ax1.plot(epochs, train_losses, color=_PALETTE['blue'],   lw=2,   label='Train Loss', alpha=0.9)
    if val_losses and len(val_losses) == len(train_losses):
        ax1.plot(epochs, val_losses, color=_PALETTE['orange'], lw=2, label='Val Loss',   alpha=0.9, linestyle='--')
    ax1.fill_between(epochs, train_losses,
                     alpha=0.08, color=_PALETTE['blue'])

    if lr_history and len(lr_history) == len(train_losses):
        ax2 = ax1.twinx()
        ax2.plot(epochs, lr_history, color=_PALETTE['teal'], lw=1.2,
                 linestyle=':', alpha=0.7, label='LR')
        ax2.set_ylabel('Learning Rate', color=_PALETTE['teal'])
        ax2.tick_params(colors=_PALETTE['teal'], labelsize=8)
        ax2.set_facecolor('white')
        for sp in ax2.spines.values():
            sp.set_edgecolor('#CBD5E1')
        ax2.legend(loc='upper right', framealpha=0.85, labelcolor=_PALETTE['text'])

    _style(ax1, plt, title='RQ2 — Training Convergence',
           xlabel='Epoch', ylabel='Loss')
    ax1.legend(loc='upper left', framealpha=0.85, labelcolor=_PALETTE['text'])

    fig.suptitle('Does meta-pretraining lead to faster, more stable convergence?',
                 color='#64748B', style='italic')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rq2_convergence.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Plot] RQ2 saved → rq2_convergence.png")


# RQ3 — Cross-city prototype similarity
def plot_rq3_city_transfer(city_names: List[str], prototype_matrix: np.ndarray,
                           city_metrics: Dict[str, Dict], save_dir: str):
    """
    RQ3: Does the CityPrototypeBank learn meaningful city-specific representations?
    Shows (a) cosine similarity heatmap between prototypes and
          (b) per-city MAE comparison.
    """
    plt, gs_mod = _mpl()
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('RQ3 — Cross-City Knowledge Transfer Quality',
                 color=_PALETTE['text'], fontweight='bold')
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # (a) prototype cosine similarity matrix
    ax1 = fig.add_subplot(gs[0])
    if prototype_matrix is not None and len(city_names) > 1:
        from numpy.linalg import norm
        n  = len(city_names)
        pm = prototype_matrix
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ni, nj = norm(pm[i]), norm(pm[j])
                sim[i, j] = float(np.dot(pm[i], pm[j]) / (ni * nj + 1e-8))
        im = ax1.imshow(sim, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        ax1.set_xticks(range(n)); ax1.set_xticklabels(city_names, rotation=30, ha='right')
        ax1.set_yticks(range(n)); ax1.set_yticklabels(city_names)
        for i in range(n):
            for j in range(n):
                ax1.text(j, i, f'{sim[i,j]:.2f}', ha='center', va='center',
                         color='black' if abs(sim[i,j]) > 0.5 else 'white')
        cb = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=_PALETTE['text'], labelsize=12)
        cb.ax.yaxis.label.set_color(_PALETTE['text'])
    else:
        ax1.text(0.5, 0.5, 'Prototype data\nnot available',
                 ha='center', va='center', color=_PALETTE['gray'],
                 transform=ax1.transAxes)
    _style(ax1, plt, title='City Prototype Cosine Similarity', grid=False)

    # (b) per-city MAE bar chart
    ax2 = fig.add_subplot(gs[1])
    colors = [_PALETTE['blue'], _PALETTE['orange'], _PALETTE['green'],
              _PALETTE['purple'], _PALETTE['teal'], _PALETTE['red']]
    if city_metrics:
        names = list(city_metrics.keys())
        maes  = [np.mean(city_metrics[c].get('MAE', [0])[:3]) for c in names]
        bars  = ax2.bar(names, maes,
                        color=[colors[i % len(colors)] for i in range(len(names))],
                        alpha=0.85, edgecolor='#CBD5E1')
        for bar, v in zip(bars, maes):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(maes) * 0.01,
                     f'{v:.2f}', ha='center', va='bottom', color=_PALETTE['text'])
        ax2.set_xticklabels(names, rotation=20, ha='right')
    else:
        ax2.text(0.5, 0.5, 'Per-city metrics\nnot available',
                 ha='center', va='center', color=_PALETTE['gray'],
                 transform=ax2.transAxes)
    _style(ax2, plt, title='Per-City Avg MAE (first 3 steps)', ylabel='MAE (mph)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rq3_city_transfer.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Plot] RQ3 saved → rq3_city_transfer.png")


# RQ4 — Chaos feature correlation with prediction error
def plot_rq4_chaos_importance(predictions: np.ndarray, targets: np.ndarray,
                               save_dir: str):
    """
    RQ4: Do high-chaos traffic segments produce higher prediction error?
    Computes rolling window chaos score and correlates with MAE per window.
    """
    plt, _ = _mpl()
    if predictions.size == 0 or targets.size == 0:
        return

    from model.model import ChaosAnalyzer
    analyzer = ChaosAnalyzer()

    # flatten to (N, T)
    if predictions.ndim == 3:
        p = predictions[:, :, 0]
        t = targets[:, :, 0]
    else:
        p = predictions
        t = targets

    window = 50
    n_windows = len(p) // window
    if n_windows < 4:
        return

    chaos_scores, maes = [], []
    for i in range(n_windows):
        seg_p = p[i*window:(i+1)*window].flatten()
        seg_t = t[i*window:(i+1)*window].flatten()
        if not (np.all(np.isfinite(seg_p)) and np.all(np.isfinite(seg_t))):
            continue
        feats = analyzer.extract_chaos_features(seg_t)
        cs = float(abs(feats[0]) + (1 - feats[1]))
        me = float(np.mean(np.abs(seg_p - seg_t)))
        if np.isfinite(cs) and np.isfinite(me):
            chaos_scores.append(cs)
            maes.append(me)

    if len(chaos_scores) < 4:
        return

    chaos_scores = np.array(chaos_scores)
    maes         = np.array(maes)
    corr         = float(np.corrcoef(chaos_scores, maes)[0, 1]) if len(np.unique(chaos_scores)) > 1 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'RQ4 — Chaos Complexity vs. Prediction Error  (r = {corr:.3f})',
                 color=_PALETTE['text'], fontweight='bold')

    # scatter
    axes[0].scatter(chaos_scores, maes, c=maes, cmap='plasma',
                    s=30, alpha=0.7, edgecolors='none')
    if (np.all(np.isfinite(chaos_scores)) and np.all(np.isfinite(maes))
            and len(np.unique(chaos_scores)) > 1):
        z  = np.polyfit(chaos_scores, maes, 1)
        xs = np.linspace(chaos_scores.min(), chaos_scores.max(), 100)
        axes[0].plot(xs, np.poly1d(z)(xs), '--', color=_PALETTE['orange'], lw=1.5)
    _style(axes[0], plt, title='Chaos Score vs MAE',
           xlabel='Chaos Score (|λ| + 1−H)', ylabel='MAE (mph)')

    # distribution split: low vs high chaos
    med = np.median(chaos_scores)
    low_mae  = maes[chaos_scores <= med]
    high_mae = maes[chaos_scores >  med]
    axes[1].hist(low_mae,  bins=15, alpha=0.7, color=_PALETTE['blue'],
                 label=f'Low chaos  (n={len(low_mae)})',  edgecolor='#CBD5E1')
    axes[1].hist(high_mae, bins=15, alpha=0.7, color=_PALETTE['red'],
                 label=f'High chaos (n={len(high_mae)})', edgecolor='#CBD5E1')
    axes[1].axvline(low_mae.mean(),  color=_PALETTE['blue'],   lw=1.5, linestyle='--')
    axes[1].axvline(high_mae.mean(), color=_PALETTE['red'],    lw=1.5, linestyle='--')
    axes[1].legend(framealpha=0.85, labelcolor=_PALETTE['text'])
    _style(axes[1], plt, title='MAE Distribution: Low vs High Chaos',
           xlabel='MAE (mph)', ylabel='Count')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rq4_chaos_importance.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Plot] RQ4 saved → rq4_chaos_importance.png")


# RQ5 — Wave interference pattern analysis
def plot_rq5_wave_patterns(predictions: np.ndarray, targets: np.ndarray,
                            save_dir: str):
    """
    RQ5: Does wave interference capture diurnal and weekly traffic rhythms?
    Applies FFT to both predictions and targets; compares dominant frequencies.
    """
    plt, _ = _mpl()
    if predictions.size == 0 or targets.size == 0:
        return

    p = predictions[:, :, 0].flatten() if predictions.ndim == 3 else predictions.flatten()
    t = targets[:, :, 0].flatten()     if targets.ndim    == 3 else targets.flatten()

    n   = min(len(p), len(t))
    p, t = p[:n], t[:n]
    freq  = np.fft.rfftfreq(n, d=1/12)   # 12 steps/hour → cycles/hour
    Pp    = np.abs(np.fft.rfft(p)) ** 2
    Pt    = np.abs(np.fft.rfft(t)) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('RQ5 — Wave Interference: Frequency Domain Analysis',
                 color=_PALETTE['text'], fontweight='bold')

    max_freq = 2.5   # cycles per hour (up to ~40-min period)
    mask     = freq <= max_freq

    axes[0].plot(freq[mask], Pt[mask], color=_PALETTE['teal'],   lw=1.5, label='Target',     alpha=0.8)
    axes[0].plot(freq[mask], Pp[mask], color=_PALETTE['orange'], lw=1.5, label='Prediction', alpha=0.8, linestyle='--')
    axes[0].fill_between(freq[mask], Pt[mask], alpha=0.1, color=_PALETTE['teal'])
    axes[0].legend(framealpha=0.85, labelcolor=_PALETTE['text'])
    _style(axes[0], plt, title='Power Spectral Density',
           xlabel='Frequency (cycles / hour)', ylabel='Power')

    # Spectral coherence (normalised cross-power)
    cross   = np.fft.rfft(p) * np.conj(np.fft.rfft(t))
    coherence = np.abs(cross) / (np.sqrt(Pp * Pt) + 1e-12)
    axes[1].plot(freq[mask], coherence[mask], color=_PALETTE['green'], lw=1.5, alpha=0.85)
    axes[1].axhline(0.7, color=_PALETTE['red'], lw=1, linestyle='--', alpha=0.6, label='0.7 threshold')
    axes[1].set_ylim(0, 1.05)
    avg_coh = float(coherence[mask].mean())
    axes[1].set_title(f'Spectral Coherence  (avg = {avg_coh:.3f})',
                      color=_PALETTE['text'], fontweight='bold', pad=8)
    axes[1].legend(framealpha=0.85, labelcolor=_PALETTE['text'])
    _style(axes[1], plt, xlabel='Frequency (cycles / hour)', ylabel='Coherence')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rq5_wave_patterns.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Plot] RQ5 saved → rq5_wave_patterns.png")


# RQ6 — Few-shot adaptation: before vs after fine-tuning
def plot_rq6_fewshot_adaptation(pretrain_metrics: Dict, finetune_metrics: Dict,
                                 save_dir: str):
    """
    RQ6: How much does few-shot fine-tuning improve over the pretrained model?
    Compares MAE / RMSE / MAPE before and after fine-tuning per horizon.
    """
    plt, _ = _mpl()
    if not pretrain_metrics or not finetune_metrics:
        return

    n     = min(12, len(pretrain_metrics.get('MAE', [])), len(finetune_metrics.get('MAE', [])))
    if n == 0:
        return
    steps = np.arange(n)
    x_labels = _STEP_LABELS[:n]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('RQ6 — Few-Shot Adaptation: Before vs After Fine-Tuning',
                 color=_PALETTE['text'], fontweight='bold')

    metrics_info = [
        ('MAE',  'MAE (mph)',  1),
        ('RMSE', 'RMSE (mph)', 1),
        ('MAPE', 'MAPE (%)',   100),
    ]
    w = 0.35
    for ax, (key, ylabel, scale) in zip(axes, metrics_info):
        pre  = np.array(pretrain_metrics[key][:n]) * scale
        post = np.array(finetune_metrics[key][:n])  * scale
        ax.bar(steps - w/2, pre,  width=w, color=_PALETTE['gray'],   alpha=0.8, label='Pretrained', edgecolor='#CBD5E1')
        ax.bar(steps + w/2, post, width=w, color=_PALETTE['blue'],   alpha=0.9, label='Fine-tuned', edgecolor='#CBD5E1')
        # improvement arrows
        for s, (pv, fv) in enumerate(zip(pre, post)):
            if fv < pv:
                ax.annotate('', xy=(s + w/2, fv), xytext=(s + w/2, pv),
                            arrowprops=dict(arrowstyle='->', color=_PALETTE['green'], lw=1.2))
        ax.set_xticks(steps)
        ax.set_xticklabels(x_labels, rotation=35, ha='right')
        ax.legend(framealpha=0.85, labelcolor=_PALETTE['text'])
        _style(ax, plt, title=ylabel, ylabel=ylabel)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rq6_fewshot_adaptation.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Plot] RQ6 saved → rq6_fewshot_adaptation.png")


# RQ7 — Spatial error distribution across sensor nodes
def plot_rq7_spatial_error(predictions: np.ndarray, targets: np.ndarray,
                            save_dir: str):
    """
    RQ7: Are prediction errors uniformly distributed across sensor nodes,
         or do specific locations consistently produce higher errors?
    """
    plt, _ = _mpl()
    if predictions.size == 0 or targets.size == 0:
        return

    def _to_2d(arr):
        """Collapse any (N,T,nodes) or (N,T,1) or (N,1,T) array to (N,T)."""
        arr = np.array(arr)
        if arr.ndim == 3:
            if arr.shape[2] == 1:
                return arr[:, :, 0]           # (N, T, 1) → (N, T)
            elif arr.shape[1] == 1:
                return arr[:, 0, :]           # (N, 1, T) old-permute → (N, T)
            else:
                return arr.mean(axis=2)       # (N, T, nodes) → (N, T)
        elif arr.ndim == 2:
            return arr
        return arr.reshape(arr.shape[0], -1)

    p = _to_2d(predictions)
    t = _to_2d(targets)
    min_n = min(p.shape[0], t.shape[0])
    min_t = min(p.shape[1], t.shape[1])
    p, t = p[:min_n, :min_t], t[:min_n, :min_t]
    if p.size == 0 or p.shape != t.shape:
        return
    node_mae = np.mean(np.abs(p - t), axis=0)   # (T,) — per-step MAE

    n_nodes = len(node_mae)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('RQ7 — Spatial Distribution of Prediction Error',
                 color=_PALETTE['text'], fontweight='bold')

    # Sorted error bar
    sorted_idx = np.argsort(node_mae)
    colors_bar = plt.cm.plasma(np.linspace(0.2, 0.9, n_nodes))
    axes[0].bar(np.arange(n_nodes), node_mae[sorted_idx], color=colors_bar,
                alpha=0.85, edgecolor='none', width=1.0)
    axes[0].axhline(node_mae.mean(), color=_PALETTE['orange'], lw=1.5, linestyle='--',
                    label=f'Mean: {node_mae.mean():.2f}')
    axes[0].axhline(np.percentile(node_mae, 90), color=_PALETTE['red'], lw=1, linestyle=':',
                    label=f'P90: {np.percentile(node_mae, 90):.2f}')
    axes[0].legend(framealpha=0.85, labelcolor=_PALETTE['text'])
    _style(axes[0], plt, title='Sorted Node MAE',
           xlabel='Node rank (ascending)', ylabel='MAE (mph)')

    # CDF of node errors
    sorted_mae = np.sort(node_mae)
    cdf        = np.arange(1, n_nodes + 1) / n_nodes
    axes[1].plot(sorted_mae, cdf, color=_PALETTE['blue'], lw=2)
    axes[1].fill_between(sorted_mae, cdf, alpha=0.1, color=_PALETTE['blue'])
    axes[1].axvline(node_mae.mean(), color=_PALETTE['orange'], lw=1.5, linestyle='--',
                    label=f'Mean: {node_mae.mean():.2f}')
    p50 = np.percentile(node_mae, 50)
    p90 = np.percentile(node_mae, 90)
    axes[1].axvline(p90, color=_PALETTE['red'], lw=1, linestyle=':',
                    label=f'P90: {p90:.2f}')
    axes[1].text(p50, 0.52, f'P50={p50:.2f}', color=_PALETTE['text'])
    axes[1].legend(framealpha=0.85, labelcolor=_PALETTE['text'])
    _style(axes[1], plt, title='Cumulative Distribution of Node MAE',
           xlabel='MAE (mph)', ylabel='Cumulative fraction')

    # pct of nodes within 10% of mean
    pct_close = float(np.mean(node_mae <= node_mae.mean() * 1.1) * 100)
    fig.text(0.5, -0.02,
             f'{pct_close:.1f}% of nodes are within 10% of the mean error — '
             f'{"uniform" if pct_close > 60 else "non-uniform"} spatial error distribution',
             ha='center', color='#64748B', style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rq7_spatial_error.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Plot] RQ7 saved → rq7_spatial_error.png")


# RQ8 — Overall summary dashboard
def plot_rq8_summary_dashboard(metrics: Dict, train_losses: List[float],
                                val_losses: List[float], predictions: np.ndarray,
                                targets: np.ndarray, save_dir: str):
    """
    Summary dashboard: one-page overview of all key results for the paper.
    """
    plt, gridspec = _mpl()
    import matplotlib.gridspec as gs_lib

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor('white')
    fig.suptitle('CIWI-CKT — Research Summary Dashboard',
                 color=_PALETTE['text'], fontweight='bold', y=0.98)

    gs = gs_lib.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38)

    # Panel A: MAE / RMSE per horizon
    ax_a = fig.add_subplot(gs[0, :2])
    n = min(12, len(metrics.get('MAE', [])))
    if n > 0:
        steps = np.arange(n)
        ax_a.plot(steps, metrics['MAE'][:n],  color=_PALETTE['blue'],   lw=2, marker='o', ms=4, label='MAE')
        ax_a.plot(steps, metrics['RMSE'][:n], color=_PALETTE['orange'], lw=2, marker='s', ms=4, label='RMSE')
        ax_a.fill_between(steps, metrics['MAE'][:n], metrics['RMSE'][:n],
                           alpha=0.08, color=_PALETTE['blue'])
        ax_a.set_xticks(steps)
        ax_a.set_xticklabels(_STEP_LABELS[:n], rotation=30, ha='right')
        ax_a.legend(framealpha=0.85, labelcolor=_PALETTE['text'])
    _style(ax_a, plt, title='A — MAE & RMSE per Horizon', ylabel='mph')

    # Panel B: MAPE per horizon
    ax_b = fig.add_subplot(gs[0, 2:])
    if n > 0:
        mape = metrics['MAPE'][:n] * 100
        bars = ax_b.bar(steps, mape,
                        color=[plt.cm.RdYlGn_r(v / max(mape)) for v in mape],
                        alpha=0.85, edgecolor='#CBD5E1')
        for bar, v in zip(bars, mape):
            ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mape) * 0.02,
                      f'{v:.1f}%', ha='center', va='bottom', color=_PALETTE['text'])
        ax_b.set_xticks(steps)
        ax_b.set_xticklabels(_STEP_LABELS[:n], rotation=30, ha='right')
    _style(ax_b, plt, title='B — MAPE per Horizon', ylabel='MAPE (%)')

    # Panel C: Training curve
    ax_c = fig.add_subplot(gs[1, :2])
    if train_losses:
        ep = np.arange(1, len(train_losses) + 1)
        ax_c.plot(ep, train_losses, color=_PALETTE['blue'],   lw=1.8, label='Train', alpha=0.9)
        if val_losses and len(val_losses) == len(train_losses):
            ax_c.plot(ep, val_losses, color=_PALETTE['orange'], lw=1.8, label='Val', linestyle='--')
        ax_c.legend(framealpha=0.85, labelcolor=_PALETTE['text'])
    _style(ax_c, plt, title='C — Training Convergence', xlabel='Epoch', ylabel='Loss')

    # Panel D: Prediction sample (first 100 time steps, node 0)
    ax_d = fig.add_subplot(gs[1, 2:])
    if predictions.size > 0:
        p_seq = predictions[:100, 0, 0] if predictions.ndim == 3 else predictions[:100, 0]
        t_seq = targets[:100, 0, 0]     if targets.ndim    == 3 else targets[:100, 0]
        ts    = np.arange(len(p_seq))
        ax_d.plot(ts, t_seq, color=_PALETTE['teal'],   lw=1.5, label='Target',     alpha=0.9)
        ax_d.plot(ts, p_seq, color=_PALETTE['orange'], lw=1.5, label='Prediction', linestyle='--', alpha=0.85)
        ax_d.fill_between(ts, t_seq, p_seq, alpha=0.07, color=_PALETTE['red'])
        ax_d.legend(framealpha=0.85, labelcolor=_PALETTE['text'])
    _style(ax_d, plt, title='D — Prediction vs Target (first 100 samples)',
           xlabel='Sample', ylabel='Speed (mph)')

    # Panel E: Key metrics summary table
    ax_e = fig.add_subplot(gs[2, :2])
    ax_e.axis('off')
    if n > 0:
        horizons = ['5 min', '15 min', '30 min', '60 min']
        h_idx    = [0, 2, 5, 11]
        h_idx    = [i for i in h_idx if i < n]
        h_labels = [horizons[h_idx.index(i)] for i in h_idx]
        table_data = [[f"{metrics['MAE'][i]:.3f}",
                       f"{metrics['RMSE'][i]:.3f}",
                       f"{metrics['MAPE'][i]*100:.2f}%",
]
                      for i in h_idx]
        tbl = ax_e.table(
            cellText=table_data,
            rowLabels=h_labels,
            colLabels=['MAE', 'RMSE', 'MAPE'],
            cellLoc='center', loc='center',
            bbox=[0, 0, 1, 1],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor('#DBEAFE' if r == 0 else 'white')
            cell.set_text_props(color='#1E293B')
            cell.set_edgecolor('#CBD5E1')
    ax_e.set_title('E — Key Metrics by Horizon', color=_PALETTE['text'], fontweight='bold', pad=6)

    # Panel F: Error distribution histogram
    ax_f = fig.add_subplot(gs[2, 2:])
    if predictions.size > 0:
        p_flat = predictions.flatten()
        t_flat = targets.flatten()
        errors = p_flat[:len(t_flat)] - t_flat[:len(p_flat)]
        ax_f.hist(errors, bins=60, color=_PALETTE['blue'], alpha=0.75,
                  edgecolor='#CBD5E1', density=True)
        ax_f.axvline(0, color=_PALETTE['orange'], lw=1.5, linestyle='--', label='Zero error')
        ax_f.axvline(errors.mean(), color=_PALETTE['red'], lw=1.2, linestyle=':',
                     label=f'Bias={errors.mean():.3f}')
        ax_f.legend(framealpha=0.85, labelcolor=_PALETTE['text'])
    _style(ax_f, plt, title='F — Prediction Error Distribution',
           xlabel='Error (mph)', ylabel='Density')

    plt.savefig(os.path.join(save_dir, 'rq8_summary_dashboard.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Plot] RQ8 saved → rq8_summary_dashboard.png")


# ---------------------------------------------------------------------------
# Master entry-point called from train.py / main.py
# ---------------------------------------------------------------------------

def create_comprehensive_plots(predictions: np.ndarray, targets: np.ndarray,
                               metrics: Dict, train_losses: List[float],
                               val_losses: List[float], save_dir: str,
                               # optional extras
                               lr_history: List[float] = None,
                               city_names: List[str] = None,
                               prototype_matrix: np.ndarray = None,
                               city_metrics: Dict = None,
                               pretrain_metrics: Dict = None):
    """
    Generate all research-question plots and the summary dashboard.

    Called automatically at the end of training via save_experiment_results().
    All plots are saved as PNG files in `save_dir`.

    Parameters
    ----------
    predictions, targets  : numpy arrays from evaluate()
    metrics               : dict from enhanced_metric_calculation()
    train_losses          : list of per-epoch train losses
    val_losses            : list of per-epoch val losses (may be None / shorter)
    save_dir              : directory to write PNG files
    lr_history            : per-epoch learning rate (optional)
    city_names            : list of city names for RQ3 (optional)
    prototype_matrix      : (N_cities, D) prototype embedding matrix (optional)
    city_metrics          : {city_name: metrics_dict} for RQ3 (optional)
    pretrain_metrics      : metrics before fine-tuning for RQ6 (optional)
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[Plots] Generating research visualisations → {save_dir}")

    if metrics and metrics.get('MAE') is not None and np.any(metrics['MAE'] > 0):
        plot_rq1_horizon_metrics(metrics, save_dir)

    if train_losses:
        _vl = val_losses if (val_losses and len(val_losses) == len(train_losses)) else None
        plot_rq2_convergence(train_losses, _vl, lr_history or [], save_dir)

    if city_names or city_metrics:
        plot_rq3_city_transfer(
            city_names or [],
            prototype_matrix,
            city_metrics or {},
            save_dir,
        )

    if predictions is not None and predictions.size > 0:
        plot_rq4_chaos_importance(predictions, targets, save_dir)
        plot_rq5_wave_patterns(predictions, targets, save_dir)
        plot_rq7_spatial_error(predictions, targets, save_dir)

    if pretrain_metrics and metrics:
        plot_rq6_fewshot_adaptation(pretrain_metrics, metrics, save_dir)

    if metrics and predictions is not None and predictions.size > 0:
        plot_rq8_summary_dashboard(metrics, train_losses or [],
                                   val_losses or [], predictions, targets, save_dir)

    print(f"[Plots] All plots saved to: {save_dir}\n")


# ---------------------------------------------------------------------------
# Legacy stubs (kept for backward-compat imports)
# ---------------------------------------------------------------------------

def plot_training_curves(train_losses, val_losses=None, save_path=None, title="Training Curves"):
    if train_losses and save_path:
        create_comprehensive_plots(np.array([]), np.array([]), {}, train_losses, val_losses or [],
                                   os.path.dirname(save_path) or '.')

def plot_predictions_comparison(predictions, actuals, save_path=None, **kw):
    pass

def plot_error_analysis(predictions, actuals, save_path=None):
    pass

def plot_metrics_comparison(metrics_dict, save_path=None):
    pass