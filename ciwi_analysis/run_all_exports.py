#!/usr/bin/env python3
"""Run every export that make_figures.py needs from trained models, in one go.

Only the Adapter class is project-specific: fill in its methods with calls into
your code. Any method left as NotImplementedError skips the exports that depend on
it, so you can complete it step by step. Then run, in order:
    python chaos_from_raw.py      (training-window indicators)
    python run_all_exports.py     (overwrites them with test windows + model errors when available)
    python make_figures.py
    python theory_checks.py
    python ablation_embedding.py
"""
from __future__ import annotations

import traceback

import numpy as np

import export_artifacts as ea

CITIES = {"metr-la": "METR-LA", "pems-bay": "PEMS-BAY", "chengdu_m": "Chengdu", "shenzhen": "Shenzhen"}
STEP_MIN = {"metr-la": 5, "pems-bay": 5, "chengdu_m": 10, "shenzhen": 10}
MODELS = ["CIWI-CKT", "TSKNET", "STGP", "TransGTR", "TPB", "CHAMFormer", "ST-GCN"]
SEEDS = [0, 1, 2, 3, 4]
# Output steps scored per dataset; MUST match the horizon columns of Tables II-III
# and HORIZONS in make_figures.py (here: 5/15/30/60 min and 10/30/60 min).
HORIZON_STEPS = {"metr-la": [1, 3, 6, 12], "pems-bay": [1, 3, 6, 12], "chengdu_m": [1, 3, 6], "shenzhen": [1, 3, 6]}
ALTERNATIVES = ["full", "additive", "concat", "fourier", "wavelet", "seasonal"]
ABLATIONS = ["Full", "w/o CAE", "w/o AWG", "w/o MIP", "w/o WAM", "w/o MP"]
KAPPA = 1.0


class Adapter:
    """Connect the exports to your code. Arrays may be numpy or torch."""

    def model(self, name, target, seed):
        """Trained model `name` adapted to `target` (dataset key) with `seed`, in eval mode."""
        raise NotImplementedError

    def batches(self, target, split):
        """Yield (x, y, hour) for split in {"val", "test"}: x (B, N, T, F) normalised input,
        y (B, N, H) normalised target, hour (B,) hour of day of the last input step."""
        raise NotImplementedError

    def predict(self, model, x):
        """Normalised prediction (B, N, H)."""
        raise NotImplementedError

    def denorm(self, target, y):
        """Normalised -> speed units."""
        raise NotImplementedError

    # CIWI-CKT internals
    def score(self, model, x):
        """Predictability score s_pred, one value per prediction sample (same unit as s_target)."""
        raise NotImplementedError

    def internals(self, model, x):
        """dict with 'C' (S, d_c) chaos embeddings, 'W' (N_w, S, T, D) query waves,
        'gain' (S,) signed interference gain, 'recon' (S, T) superposition in speed units,
        where S = B*N windows (sensor-major order matching x)."""
        raise NotImplementedError

    def wave_freqs(self, model):
        """Learned wave frequencies in cycles/day."""
        raise NotImplementedError

    def prototypes(self, model):
        """(|C|, D_m) source prototypes."""
        raise NotImplementedError

    # Extra experiments (each returns test MAE per horizon in HORIZON_STEPS, speed units)
    def single_source(self, source, target, seed):
        """(mae_transfer, mae_scratch) at 30 min for one source -> target run."""
        raise NotImplementedError

    def adaptation_mae(self, name, target, n_samples, seed):
        """30-min test MAE after adapting with n_samples target samples."""
        raise NotImplementedError

    def alternative_mae(self, variant, target, seed):
        """List of MAE per horizon for a CIWI-CKT variant whose interference operator is replaced."""
        raise NotImplementedError

    def ablation_model(self, variant, target, seed):
        """Trained ablation variant (see ABLATIONS)."""
        raise NotImplementedError


def errors_and_scores(ad, key):
    per_model, s_val, e_val, s_te, e_te = {}, [], [], [], []
    for name in MODELS:
        m = ad.model(name, key, SEEDS[0])
        rows = []
        for x, y, _ in ad.batches(key, "test"):
            yh = ad.predict(m, x)
            err = np.abs(np.asarray(ad.denorm(key, yh)) - np.asarray(ad.denorm(key, y)))
            rows.append(err.mean(1)[:, [h - 1 for h in HORIZON_STEPS[key]]])
            if name == "CIWI-CKT":
                s_te.append(np.asarray(ad.score(m, x)).ravel())
                e_te.append(ea.sample_error(yh, y))
        per_model[name] = np.concatenate(rows)
        if name == "CIWI-CKT":
            for x, y, _ in ad.batches(key, "val"):
                s_val.append(np.asarray(ad.score(m, x)).ravel())
                e_val.append(ea.sample_error(ad.predict(m, x), y))
    ea.save_errors(key, per_model)
    ea.save_predictability(key, np.concatenate(s_val), np.concatenate(e_val),
                           np.concatenate(s_te), np.concatenate(e_te), KAPPA)


def chaos_gain(ad, key):
    """Offline indicators and 30-min errors of CIWI-CKT and TSKNET on the SAME test windows
    (overwrites the training-window file written by chaos_from_raw.py)."""
    import chaos_from_raw as cr
    h30 = 30 // STEP_MIN[key] - 1
    models = {n: ad.model(n, key, SEEDS[0]) for n in ("CIWI-CKT", "TSKNET")}
    W, hours, errs = [], [], {n: [] for n in models}
    for x, y, hour in ad.batches(key, "test"):
        xs = np.asarray(x)[..., 0]                                   # (B, N, T) normalised speed
        W.append(xs.reshape(-1, xs.shape[-1]))
        hours.append(np.repeat(np.asarray(hour), xs.shape[1]))
        for n, m in models.items():
            d = np.abs(np.asarray(ad.denorm(key, ad.predict(m, x))) - np.asarray(ad.denorm(key, y)))
            errs[n].append(d[..., h30].ravel())
    W = np.concatenate(W)
    ind = {"lyapunov": cr.lyapunov_rosenstein(W), "permutation_entropy": cr.permutation_entropy(W),
           "fractal_dim": cr.higuchi_fd(W)}
    ea.save_chaos(key, np.concatenate(hours), ind, {n: np.concatenate(v) for n, v in errs.items()})


def ablation_embeddings(ad):
    per, cities, diff = {v: [] for v in ABLATIONS}, [], []
    for key, city in CITIES.items():
        h30 = 30 // STEP_MIN[key] - 1
        x, y, _ = next(iter(ad.batches(key, "test")))
        for v in ABLATIONS:
            per[v].append(np.asarray(ad.internals(ad.ablation_model(v, key, SEEDS[0]), x)["C"]))
        n = len(per[ABLATIONS[0]][-1])
        cities += [city] * n
        diff.append(np.abs(np.asarray(y)[..., h30] - np.asarray(x)[..., -1, 0]).ravel()[:n])
    ea.save_ablation_embeddings({v: np.concatenate(e) for v, e in per.items()}, cities, np.concatenate(diff))


def waves_and_theory(ad, key):
    m = ad.model("CIWI-CKT", key, SEEDS[0])
    x, y, hour = next(iter(ad.batches(key, "test")))
    it = ad.internals(m, x)
    t_h = np.arange(it["recon"].shape[1]) * STEP_MIN[key] / 60
    yy = np.asarray(ad.denorm(key, x[..., 0])).reshape(-1, it["recon"].shape[1])[0]
    W = np.asarray(it["W"])
    ea.save_wave_example(key, t_h, yy, it["recon"][0], W[:, 0, :, 0])
    gains, speeds = [], []
    for x, _, _ in ad.batches(key, "test"):
        g = ad.internals(m, x)["gain"]
        gains.append(np.asarray(g))
        speeds.append(np.asarray(ad.denorm(key, x[..., -1, 0])).ravel())
    ea.save_interference(key, np.concatenate(gains), np.concatenate(speeds))
    ea.log_superposition(W)
    ea.prototype_stats(key, ad.prototypes(m))
    return it["C"]


def main():
    ad = Adapter()
    freqs, embs, cities = {}, [], []
    for key, city in CITIES.items():
        for label, fn in (("errors + calibration", lambda: errors_and_scores(ad, key)),
                          ("waves + theory", lambda: embs.append(waves_and_theory(ad, key)) or
                           cities.extend([city] * len(embs[-1]))),
                          ("wave freqs", lambda: freqs.__setitem__(key, ad.wave_freqs(ad.model("CIWI-CKT", key, 0)))),
                          ("chaos gain", lambda: chaos_gain(ad, key))):
            try:
                fn()
                print(f"[ok]   {city}: {label}")
            except NotImplementedError:
                print(f"[skip] {city}: {label} (adapter method not implemented)")
            except Exception:
                print(f"[fail] {city}: {label}")
                traceback.print_exc()
        for seed in SEEDS:
            for src_key, src in CITIES.items():
                if src_key == key:
                    continue
                try:
                    ea.log_transfer(src, city, seed, *ad.single_source(src_key, key, seed))
                except NotImplementedError:
                    break
            for name in MODELS[:4]:
                for n in (10, 50, 100, 250, 500, 1000, 1750, 3000):
                    try:
                        ea.log_adaptation(name, city, n, seed, ad.adaptation_mae(name, key, n, seed))
                    except NotImplementedError:
                        break
            for v in ALTERNATIVES:
                try:
                    for h, mae in zip(HORIZON_STEPS[key][1:], ad.alternative_mae(v, key, seed)[1:]):
                        ea.log_alternative(v, city, h * STEP_MIN[key], seed, mae)
                except NotImplementedError:
                    break
    try:
        ablation_embeddings(ad)
        print("[ok]   ablation embeddings")
    except NotImplementedError:
        print("[skip] ablation embeddings (adapter method not implemented)")
    if freqs:
        ea.save_wave_freqs(freqs)
    if embs:
        ea.save_embeddings(np.concatenate([np.asarray(e) for e in embs]), cities)
    print("Done. Now run make_figures.py, theory_checks.py and ablation_embedding.py.")


if __name__ == "__main__":
    main()
