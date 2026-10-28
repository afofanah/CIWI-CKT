"""Save the real model outputs that make_figures.py reads.

Call these from the evaluation code after training. Dataset keys are the
config keys: metr-la, pems-bay, chengdu_m, shenzhen. Torch tensors are accepted.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(os.environ.get("CIWI_ARTIFACTS", "/Users/s5273738/WAVEFSL_Journal/results/artifacts"))


def _np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _out(name):
    ART.mkdir(parents=True, exist_ok=True)
    return ART / name


def append_csv(name, rows):
    """rows: list of dicts. Appends to ART/name (creates it with a header if new)."""
    df = pd.DataFrame(rows)
    p = _out(name)
    df.to_csv(p, mode="a", header=not p.exists(), index=False)


# ---------------------------------------------------------------- statistics
def save_errors(key, per_model):
    """per_model: {model_name: (N_test, 4) absolute error per test window, averaged over
    sensors, de-normalised, columns = the four table horizons}. All models must use the
    same test windows in the same order. Include "CIWI-CKT" and the baselines."""
    np.savez_compressed(_out(f"errors_{key}.npz"),
                        **{m: _np(v).astype(np.float32) for m, v in per_model.items()})


# ---------------------------------------------------------------- calibration
def sample_error(y_hat, y):
    """Per-sample L2 error, same definition as s_target in training."""
    y_hat, y = _np(y_hat), _np(y)
    return np.linalg.norm((y_hat - y).reshape(len(y), -1), axis=1)


def save_predictability(key, s_cal, e_cal, s_test, e_test, kappa=1.0):
    """s_*: predicted scores; e_*: sample_error(...) in the normalised units used in training.
    Calibration split = validation set, test split = test set."""
    np.savez_compressed(_out(f"predictability_{key}.npz"), s_cal=_np(s_cal).ravel(),
                        e_cal=_np(e_cal).ravel(), s_test=_np(s_test).ravel(),
                        e_test=_np(e_test).ravel(), kappa=float(kappa))


# ---------------------------------------------------------------- chaos
def save_chaos(key, hour, indicators, errors=None):
    """hour: (N,) hour of day of each test window.
    indicators: {"lyapunov": (N,), "sample_entropy": (N,), ...} offline indicators of Definition 2.
    errors: {"CIWI-CKT": (N,), "TSKNET": (N,)} per-window MAE at 30 min."""
    d = {"hour": _np(hour).astype(np.int16)}
    d.update({f"ind_{k}": _np(v).astype(np.float32) for k, v in indicators.items()})
    d.update({f"err_{k}": _np(v).astype(np.float32) for k, v in (errors or {}).items()})
    np.savez_compressed(_out(f"chaos_{key}.npz"), **d)


# ---------------------------------------------------------------- transfer
def save_embeddings(emb, city):
    """emb: (N, d) learned chaos embeddings C; city: (N,) names (METR-LA, PEMS-BAY, Chengdu, Shenzhen)."""
    np.savez_compressed(_out("embeddings.npz"), emb=_np(emb).astype(np.float32),
                        city=np.asarray(city).astype(str))


def log_transfer(source, target, seed, mae_transfer, mae_scratch, horizon=30):
    """Single-source transfer run vs. target-only training with the same support budget."""
    append_csv("transfer_pairs.csv", [dict(source=source, target=target, seed=seed, horizon=horizon,
                                           mae_transfer=mae_transfer, mae_scratch=mae_scratch)])


def log_adaptation(model, target, n_samples, seed, mae):
    append_csv("adaptation.csv", [dict(model=model, target=target, n_samples=n_samples, seed=seed, mae=mae)])


# ---------------------------------------------------------------- wave analysis
def save_wave_example(key, t_h, y, recon, components=None, period_h=None):
    """One sensor-day: t_h hours (T,), y measured speed (T,), recon superposition (T,) in the same
    units, components (N_w, T) individual waves, period_h (N_w,) dominant period of each wave."""
    d = dict(t_h=_np(t_h), y=_np(y), recon=_np(recon))
    if components is not None:
        d["components"] = _np(components)
    if period_h is not None:
        d["period_h"] = _np(period_h)
    np.savez_compressed(_out(f"wave_example_{key}.npz"), **d)


def save_wave_freqs(freqs_cycles_per_day):
    """{dataset_key: array of learned wave frequencies in cycles/day}."""
    np.savez_compressed(_out("wave_freqs.npz"), **{k: _np(v) for k, v in freqs_cycles_per_day.items()})


def save_interference(key, gain, speed):
    """gain: (T,) or (T, nodes) signed interference gain on the test split, e.g.
    ||I~|| - ||sum_n beta_n (W_s + W_q)||  (>0 constructive, <0 destructive);
    speed: same shape, measured speed (de-normalised)."""
    np.savez_compressed(_out(f"interference_{key}.npz"), gain=_np(gain).astype(np.float32),
                        speed=_np(speed).astype(np.float32))


def log_alternative(variant, dataset, horizon, seed, mae, rmse=None, params_m=None):
    """variant in {full, additive, concat, fourier, wavelet, seasonal}; dataset = display name."""
    append_csv("alternatives.csv", [dict(variant=variant, dataset=dataset, horizon=horizon, seed=seed,
                                         mae=mae, rmse=rmse, params_m=params_m)])


# ---------------------------------------------------------------- theory checks
def tau_T(t):
    t = _np(t).astype(float)
    return float(np.linalg.norm(t) / np.sqrt(t.size))


def spectral_norm(W):
    return float(np.linalg.norm(_np(W), 2))


def gamma_bound(T, tau, W_A1, W_A2, nu_A, lam_k, k0, W_K, L_g=1.13):
    """Lemma 2(ii): sqrt(T) * (L_A + tau_T * L_k) for one wave."""
    L_A = 0.25 * L_g * nu_A * spectral_norm(W_A2) * spectral_norm(W_A1)
    L_k = lam_k * float(np.abs(_np(k0)).max()) * spectral_norm(W_K)
    return float(np.sqrt(T) * (L_A + tau * L_k))


def fd_lipschitz(fn, x, eps=1e-3, n_dir=16, seed=0):
    """Max over random unit directions of ||fn(x+eps u) - fn(x)|| / eps, per sample (lower estimate).
    fn: torch callable in eval mode mapping (B, ...) -> (B, ...)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        fx = fn(x)
        best = torch.zeros(x.shape[0], device=x.device)
        for _ in range(n_dir):
            u = torch.randn(x.shape, generator=g).to(x)
            u = u / u.flatten(1).norm(dim=1).view(-1, *[1] * (x.dim() - 1))
            r = (fn(x + eps * u) - fx).flatten(1).norm(dim=1) / eps
            best = torch.maximum(best, r)
    return best.cpu().numpy()


def log_gamma(T, index, bound, fd):
    """index: 'raw' (t_i = i) or 'normalised' (t_i = i/T). fd: max fd_lipschitz over a batch."""
    append_csv("gamma.csv", [dict(T=T, index=index, bound=bound, fd=float(np.max(fd)))])


def superposition_stats(W):
    """W: (N_w, B, ...) wave outputs over B samples. Returns the quantities of Lemma 5,
    using v_max = max_n Var(W^(n)) (the constant the proof actually needs)."""
    W = _np(W).reshape(W.shape[0], W.shape[1], -1).astype(float)
    Nw = W.shape[0]
    Wc = W - W.mean(1, keepdims=True)
    cov = np.einsum("ibk,jbk->ij", Wc, Wc) / W.shape[1]
    v = np.diag(cov)
    rho = cov / np.sqrt(np.outer(v, v) + 1e-30)
    off = ~np.eye(Nw, dtype=bool)
    rho_plus = float(np.clip(rho[off], 0, None).mean()) if Nw > 1 else 0.0
    S = Wc.mean(0)
    var_emp = float((S ** 2).sum(1).mean())
    bound = float(v.max() / Nw * (1 + (Nw - 1) * rho_plus))
    return dict(N_w=Nw, var_emp=var_emp, v_max=float(v.max()), rho_plus=rho_plus, bound=bound)


def log_superposition(W):
    append_csv("superposition.csv", [superposition_stats(W)])


def prototype_stats(target, P):
    """P: (|C|, D_m) source-city prototypes used for this target."""
    P = _np(P).astype(float)
    U = P / np.linalg.norm(P, axis=1, keepdims=True)
    C = len(U)
    G = U @ U.T
    d_div = float((G.sum() - np.trace(G)) / (C * (C - 1)))
    rel = float(np.linalg.norm(P.mean(0)) / np.linalg.norm(P, axis=1).mean())
    append_csv("prototypes.csv", [dict(target=target, n_sources=C, d_div=d_div, mean_norm_rel=rel)])


def episode_stats(target, losses):
    """losses: (R, n_tau) evaluation-episode losses over R repetitions (seeds/resamples)."""
    L = _np(losses).astype(float)
    n = L.shape[1]
    R = np.corrcoef(L.T)
    off = ~np.eye(n, dtype=bool)
    rho_plus = float(np.clip(np.nan_to_num(R[off]), 0, None).mean())
    append_csv("episodes.csv", [dict(target=target, n_tau=n, rho_plus=rho_plus,
                                     n_eff=n / (1 + (n - 1) * rho_plus))])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# ---------------------------------------------------------------- input stability (Proposition 1)
def run_noise_robustness(model, loader, dataset, model_name, sigmas=(0, .05, .1, .2, .3, .5),
                         seed=0, inverse=None, device="cpu", horizon_step=None):
    """Adds N(0, sigma^2) to the normalised inputs and logs test MAE per sigma.
    loader yields (x, y); model(x) -> y_hat. inverse: optional callable to de-normalise.
    horizon_step: index along the horizon axis to score (None = all horizons)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    model.eval()
    rows = []
    with torch.no_grad():
        for s in sigmas:
            err, n = 0.0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                xn = x + s * torch.randn(x.shape, generator=g).to(x)
                yh = model(xn)
                if inverse is not None:
                    yh, y = inverse(yh), inverse(y)
                if horizon_step is not None:
                    yh, y = yh[:, horizon_step], y[:, horizon_step]
                err += (yh - y).abs().sum().item()
                n += y.numel()
            rows.append(dict(model=model_name, dataset=dataset, sigma=s, seed=seed, mae=err / n))
    append_csv("robustness.csv", rows)
    return rows


def log_lipschitz(model, x, dataset, model_name, bound=None, eps=1e-3, n_dir=16):
    """Finite-difference Lipschitz ratios of the full forward map F_Theta on a test batch x
    (inference mode). bound: L_F from Proposition 1 (CIWI-CKT only)."""
    model.eval()
    r = fd_lipschitz(model, x, eps=eps, n_dir=n_dir)
    append_csv("lipschitz.csv", [dict(model=model_name, dataset=dataset, ratio=float(v),
                                      bound=bound if bound is not None else np.nan) for v in r])
    return r


# ---------------------------------------------------------------- ablation embeddings
def save_ablation_embeddings(per_variant, city, difficulty):
    """per_variant: {"Full": (N, d), "w/o CAE": (N, d), ...} learned representations (e.g. H_att pooled
    over time) of the SAME N test windows for each trained ablation variant, "Full" first.
    city: (N,) city names; difficulty: (N,) 30-min-ahead |speed change| of each window."""
    d = {f"emb_{k}": _np(v).astype(np.float32) for k, v in per_variant.items()}
    np.savez_compressed(_out("ablation_embeddings.npz"), city=np.asarray(city).astype(str),
                        difficulty=_np(difficulty), **d)
