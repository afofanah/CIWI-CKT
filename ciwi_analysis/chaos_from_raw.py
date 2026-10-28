#!/usr/bin/env python3
"""Offline chaos indicators computed directly from raw speed windows.

Windows have the model's input length T (12 steps) and are drawn from the
training split only. Indicators (vectorised over windows):
  lyapunov             largest Lyapunov exponent, Rosenstein method, m=3, tau=1,
                       Theiler window 1, divergence fitted over k=0..3 (per step)
  sample_entropy       m=2, r=0.2*sd; reported only (undefined on most 12-step windows)
  permutation_entropy  order 3, normalised to [0, 1]
  fractal_dim          Higuchi fractal dimension, k_max=4
Writes artifacts/chaos_<key>.npz through export_artifacts.save_chaos.
Add per-window model errors later with errors={"CIWI-CKT": ..., "TSKNET": ...}.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_artifacts as ea  # noqa: E402

ROOT = Path(os.environ.get("CIWI_ROOT", "/Users/s5273738/CIWI-CKT"))
CITIES = {"metr-la": 34272, "pems-bay": 52116, "chengdu_m": 17280, "shenzhen": 17280}
T, N_WIN, TRAIN_FRAC, SEED = 12, 40000, 0.7, 0


def sample_windows(data, n, rng):
    tr = int(round(TRAIN_FRAC * data.shape[0]))
    t0 = rng.integers(0, tr - T, n)
    nd = rng.integers(0, data.shape[1], n)
    W = np.stack([data[t0 + k, nd, 0] for k in range(T)], 1).astype(float)
    hour = (np.floor(data[t0 + T - 1, nd, 1] * 24).astype(int)) % 24
    ok = np.all(W > 0, 1) & (W.std(1) > 1e-6)
    return W[ok], hour[ok]


def embed(W, m=3):
    return np.stack([W[:, i:W.shape[1] - m + 1 + i] for i in range(m)], -1)


def lyapunov_rosenstein(W, m=3, theiler=1, kmax=3):
    Y = embed(W, m)
    M = Y.shape[1]
    D = np.linalg.norm(Y[:, :, None, :] - Y[:, None, :, :], axis=-1)
    idx = np.arange(M)
    D[:, np.abs(idx[:, None] - idx[None, :]) <= theiler] = np.inf
    nn = D.argmin(2)
    logs = np.full((W.shape[0], kmax + 1), np.nan)
    rows = np.arange(W.shape[0])[:, None]
    for k in range(kmax + 1):
        i = idx[None, :] + k
        j = nn + k
        valid = (i < M) & (j < M)
        ic, jc = np.minimum(i, M - 1), np.minimum(j, M - 1)
        d = np.linalg.norm(Y[rows, np.broadcast_to(ic, nn.shape)] - Y[rows, jc], axis=-1)
        d = np.where(valid & (d > 0), d, np.nan)
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            logs[:, k] = np.nanmean(np.log(d), 1)
    k = np.arange(kmax + 1, dtype=float)
    kc = k - k.mean()
    good = np.all(np.isfinite(logs), 1)
    lam = np.full(W.shape[0], np.nan)
    lam[good] = (logs[good] - logs[good].mean(1, keepdims=True)) @ kc / (kc @ kc)
    return lam


def sample_entropy(W, m=2, rfac=0.2):
    r = rfac * W.std(1)
    n = W.shape[1] - m

    def count(mm):
        Y = embed(W, mm)[:, :n]
        C = np.abs(Y[:, :, None, :] - Y[:, None, :, :]).max(-1)
        iu = np.triu_indices(n, 1)
        return (C[:, iu[0], iu[1]] <= r[:, None]).sum(1)

    B, A = count(m), count(m + 1)
    with np.errstate(all="ignore"):
        se = -np.log(A / B)
    se[(A == 0) | (B == 0)] = np.nan
    return se


def permutation_entropy(W, order=3):
    Y = embed(W, order)
    ranks = np.argsort(np.argsort(Y, -1), -1)
    code = (ranks * (order ** np.arange(order))).sum(-1)
    H = np.zeros(W.shape[0])
    for c in np.unique(code):
        p = (code == c).mean(1)
        H -= np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)
    return H / math.log(math.factorial(order))


def higuchi_fd(W, kmax=4):
    n = W.shape[1]
    ks, Ls = np.arange(1, kmax + 1), []
    for k in ks:
        Lk = []
        for m0 in range(k):
            idx = np.arange(m0, n, k)
            if len(idx) < 2:
                continue
            seg = np.abs(np.diff(W[:, idx], axis=1)).sum(1)
            Lk.append(seg * (n - 1) / ((len(idx) - 1) * k) / k)
        Ls.append(np.mean(Lk, 0))
    L = np.log(np.stack(Ls, 1) + 1e-12)
    x = np.log(1.0 / ks)
    xc = x - x.mean()
    return (L - L.mean(1, keepdims=True)) @ xc / (xc @ xc)


def main():
    rng = np.random.default_rng(SEED)
    for key in CITIES:
        p = ROOT / "data" / key / "dataset.npy"
        if not p.exists():
            print(f"[skip] {p}")
            continue
        data = np.load(p, mmap_mode="r")
        W, hour = sample_windows(data, N_WIN, rng)
        se = sample_entropy(W)
        ind = {"lyapunov": lyapunov_rosenstein(W), "permutation_entropy": permutation_entropy(W),
               "fractal_dim": higuchi_fd(W)}
        ea.save_chaos(key, hour, ind)
        print(f"       sample entropy (m=2) undefined in {100 * np.isnan(se).mean():.0f}% of windows; not used")
        msg = ", ".join(f"{k}: median {np.nanmedian(v):.3f} ({100 * np.isnan(v).mean():.0f}% undefined)"
                        for k, v in ind.items())
        print(f"[ok]   {key}: {len(W)} windows; {msg}")


if __name__ == "__main__":
    main()
