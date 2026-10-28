#!/usr/bin/env python3
"""CIWI-CKT figures sized for IEEE two-column papers.

Panels built from the manuscript tables are always drawn. Every other panel is
computed from artifacts written by export_artifacts.py (real model outputs) or
from the raw dataset.npy files. If an input is missing the panel is skipped and
listed at the end; nothing is simulated.

Outputs in RESULTS: <figure>.pdf/.png, figure_numbers.tex, dataset_table.tex,
alternatives_table.tex.
"""
from __future__ import annotations

import math
import os
import traceback
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import signal, stats

ROOT = Path(os.environ.get("CIWI_ROOT", "/Users/s5273738/WAVEFSL_Journal"))
RESULTS = Path(os.environ.get("CIWI_RESULTS", ROOT / "results"))
ART = Path(os.environ.get("CIWI_ARTIFACTS", RESULTS / "artifacts"))
CONFIG = Path(os.environ.get("CIWI_CONFIG", ROOT / "config.yaml"))

COL_W, TEXT_W = 3.5, 7.16
N_SEEDS = 5
OURS = "CIWI-CKT"

CITIES = ["METR-LA", "PEMS-BAY", "Chengdu", "Shenzhen"]
KEY = {"METR-LA": "metr-la", "PEMS-BAY": "pems-bay", "Chengdu": "chengdu_m", "Shenzhen": "shenzhen"}
TAG = {"METR-LA": "Metrla", "PEMS-BAY": "Pemsbay", "Chengdu": "Chengdu", "Shenzhen": "Shenzhen"}
SHORT = {"METR-LA": "LA", "PEMS-BAY": "BAY", "Chengdu": "CD", "Shenzhen": "SZ"}
ROAD = {"METR-LA": "freeway", "PEMS-BAY": "freeway", "Chengdu": "urban", "Shenzhen": "urban"}
STEP_MIN = {"METR-LA": 5, "PEMS-BAY": 5, "Chengdu": 10, "Shenzhen": 10}
N_STEPS = {"METR-LA": 34272, "PEMS-BAY": 52116, "Chengdu": 17280, "Shenzhen": 17280}
N_NODES = {"METR-LA": 207, "PEMS-BAY": 325, "Chengdu": 524, "Shenzhen": 627}
SPLIT = (0.7, 0.1, 0.2)
FEWSHOT_AT_END_OF_TRAIN = True
RUSH_H = [(7, 10), (16, 20)]


def _load_cfg():
    try:
        import yaml
        return yaml.safe_load(CONFIG.read_text()) or {}
    except Exception:
        return {}


CFG = _load_cfg()
T_IN = CFG.get("task", {}).get("his_num", 12)
H_OUT = CFG.get("task", {}).get("pred_num", 12)
TARGET_DAYS = CFG.get("training", {}).get("target_days", 3)

# Horizon labels exactly as printed in the performance tables (single place to edit).
HORIZONS = {"METR-LA": [5, 15, 30, 60], "PEMS-BAY": [5, 15, 30, 60],
            "Chengdu": [10, 15, 30, 60], "Shenzhen": [10, 15, 30, 60]}

# ---------------------------------------------------------------- table values
MAE = {
    "METR-LA": {
        "CIWI-CKT": ([1.2784, 2.2581, 2.8619, 3.1178], [.06, .09, .12, .14]),
        "TSKNET": ([1.5738, 2.4882, 3.0273, 3.2956], [.16, .26, .37, .44]),
        "STGP": ([2.2983, 2.9736, 3.5418, 4.2329], [.19, .30, .43, .57]),
        "TransGTR": ([2.3859, 3.0123, 3.6428, 4.4426], [.20, .30, .44, .59]),
        "TPB": ([2.3927, 2.9118, 3.6943, 4.5126], [.20, .29, .44, .60]),
        "CHAMFormer": ([2.5122, 3.2411, 3.9979, 4.9217], [.20, .32, .47, .66]),
        "ST-GCN": ([2.7018, 3.3216, 4.2119, 5.1024], [.23, .33, .51, .71]),
    },
    "PEMS-BAY": {
        "CIWI-CKT": ([1.0128, 1.3580, 1.8029, 2.1014], [.05, .07, .10, .11]),
        "TSKNET": ([1.0838, 1.4463, 1.9022, 2.2014], [.13, .19, .29, .37]),
        "STGP": ([1.1725, 1.7453, 2.1358, 2.7036], [.14, .22, .32, .44]),
        "TransGTR": ([1.1658, 1.7053, 2.1348, 2.7913], [.14, .22, .32, .46]),
        "TPB": ([1.1839, 1.7326, 2.2254, 2.6027], [.14, .22, .33, .43]),
        "CHAMFormer": ([1.4981, 1.9548, 2.4012, 2.9059], [.16, .25, .35, .48]),
        "ST-GCN": ([1.4772, 1.7575, 2.3493, 2.8128], [.16, .22, .35, .47]),
    },
    "Chengdu": {
        "CIWI-CKT": ([1.2089, 1.5258, 2.0427, 2.2458], [.05, .06, .09, .10]),
        "TSKNET": ([1.4783, 1.6585, 2.2203, 2.4411], [.12, .13, .20, .24]),
        "STGP": ([1.8978, 1.9847, 2.7456, 2.8659], [.15, .16, .24, .29]),
        "TransGTR": ([2.2814, 2.5127, 2.6589, 2.8073], [.18, .20, .23, .28]),
        "TPB": ([2.2843, 2.5436, 2.8637, 3.2829], [.18, .21, .25, .33]),
        "CHAMFormer": ([2.2913, 2.5962, 2.8889, 3.3378], [.18, .21, .25, .33]),
        "ST-GCN": ([2.3185, 2.5437, 2.8953, 3.3658], [.19, .21, .26, .34]),
    },
    "Shenzhen": {
        "CIWI-CKT": ([1.2403, 1.4811, 1.7298, 1.9866], [.05, .06, .08, .09]),
        "TSKNET": ([1.3481, 1.6099, 1.8802, 2.1593], [.11, .13, .17, .22]),
        "STGP": ([1.7658, 1.8247, 2.2749, 2.4276], [.14, .15, .21, .24]),
        "TransGTR": ([1.6547, 1.8953, 2.3058, 2.4763], [.13, .15, .21, .25]),
        "TPB": ([1.8039, 1.9678, 2.2243, 2.5137], [.14, .16, .20, .25]),
        "CHAMFormer": ([1.9073, 2.1129, 2.5687, 2.9789], [.15, .17, .23, .30]),
        "ST-GCN": ([1.9813, 2.0618, 2.3759, 2.8963], [.16, .17, .21, .29]),
    },
}
BASELINES = ["TSKNET", "STGP", "TransGTR", "TPB", "CHAMFormer", "ST-GCN"]

EFF = pd.DataFrame([
    ("ST-DTNN", "Reptile", 142.3, 3.2, 200, 15.2, 1.8, 68.4, 10000, 2.1, 4.0917, 2.4116),
    ("ST-GCN", "Reptile", 128.7, 2.9, 180, 13.8, 1.6, 61.9, 8500, 1.9, 4.2119, 2.3493),
    ("DDGCRN", "Reptile", 156.2, 3.5, 220, 17.4, 2.1, 78.2, 12000, 2.4, 4.2097, 2.4839),
    ("FOGS", "Reptile", 135.8, 3.1, 190, 14.6, 1.7, 66.3, 9500, 2.0, 3.9958, 2.3837),
    ("DTAN", "Reptile", 148.9, 3.3, 210, 16.1, 1.9, 72.8, 11000, 2.2, 4.0915, 2.3917),
    ("ST-GFSL", "Transfer", 167.4, 3.8, 160, 18.3, 2.2, 42.7, 5000, 2.8, 3.8728, 2.2217),
    ("TPB", "Transfer", 172.1, 3.9, 170, 19.1, 2.3, 38.5, 4200, 2.9, 3.6943, 2.2254),
    ("AdaRNN", "Transfer", 158.3, 3.6, 150, 17.8, 2.0, 35.2, 3800, 2.5, 3.9015, 2.3815),
    ("TransGTR", "Transfer", 189.2, 4.2, 140, 21.4, 2.6, 31.8, 3200, 3.1, 3.6428, 2.1348),
    ("Cross-IDR", "Transfer", 182.6, 4.1, 145, 20.8, 2.5, 33.4, 3500, 3.0, 3.8198, 2.1746),
    ("STGP", "Prompt", 194.7, 4.4, 130, 22.3, 2.7, 28.6, 2800, 3.3, 3.5418, 2.1358),
    ("DynAGS", "Prompt", 201.3, 4.6, 135, 23.1, 2.8, 29.8, 3000, 3.4, 3.5769, 2.1569),
    ("PromptST", "Prompt", 197.8, 4.5, 132, 22.7, 2.7, 29.2, 2900, 3.3, 3.6113, 2.1773),
    ("ProST", "Prompt", 203.6, 4.7, 138, 23.4, 2.8, 30.5, 3100, 3.5, 3.6479, 2.1996),
    ("FlashST", "Prompt", 209.4, 4.8, 142, 24.0, 2.9, 31.2, 3300, 3.6, 3.6821, 2.2208),
    ("CIWI-CKT", "Ours", 176.8, 3.4, 120, 18.9, 2.4, 8.2, 1750, 3.2, 2.8619, 1.8029),
], columns=["model", "family", "t_ep", "mem_tr", "epochs", "inf_ms", "mem_inf",
            "ad_min", "ad_samples", "ad_mem", "mae30_la", "mae30_bay"])
EFF["mae30"] = (EFF.mae30_la + EFF.mae30_bay) / 2

COMPLEXITY = pd.DataFrame([
    ("DCRNN", "Classical", 3.8, 2.1), ("ST-GCN", "Classical", 4.2, 1.8),
    ("Graph WaveNet", "Classical", 5.1, 2.4), ("ST-GFSL", "Transfer", 8.4, 4.2),
    ("TransGTR", "Transfer", 12.6, 6.8), ("TSKNET", "Transfer", 7.1, 3.9),
    ("STGP", "Prompt", 7.5, 4.1), ("FlashST", "Prompt", 6.2, 3.5), ("CIWI-CKT", "Ours", 5.3, 2.6),
], columns=["model", "family", "params_m", "gflops"])

ALT_ORDER = ["full", "additive", "concat", "fourier", "wavelet", "seasonal"]
ALT_NAME = {"full": "CIWI-CKT (interference)", "additive": r"Additive only ($\gamma$=0)",
            "concat": "Concat fusion", "fourier": "Fourier features",
            "wavelet": "Wavelet (DWT)", "seasonal": "Seasonal embedding"}

# ------------------------------------------------------------------ styling
MSTYLE = {
    "CIWI-CKT": ("#D55E00", "o", "-"), "TSKNET": ("#0072B2", "s", "--"),
    "STGP": ("#009E73", "^", "-."), "TransGTR": ("#CC79A7", "v", ":"),
    "TPB": ("#E69F00", "D", "--"), "CHAMFormer": ("#56B4E9", "P", "-."),
    "ST-GCN": ("#7F7F7F", "X", ":"),
}
CSTYLE = {"METR-LA": ("#0072B2", "o"), "PEMS-BAY": ("#E69F00", "s"),
          "Chengdu": ("#009E73", "^"), "Shenzhen": ("#CC79A7", "D")}
FAM_C = {"Reptile": "#9E9E9E", "Classical": "#9E9E9E", "Transfer": "#0072B2",
         "Prompt": "#009E73", "Ours": "#D55E00"}
FAM_M = {"Reptile": "s", "Classical": "s", "Transfer": "^", "Prompt": "D", "Ours": "o"}


FONT_SCALE = 1.0


def fs(x):
    return x * FONT_SCALE


def set_style(scale=None):
    global FONT_SCALE
    if scale is not None:
        FONT_SCALE = scale
    k = FONT_SCALE
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8 * k, "axes.titlesize": 8 * k, "axes.labelsize": 8 * k,
        "xtick.labelsize": 7 * k, "ytick.labelsize": 7 * k, "legend.fontsize": 6.5 * k,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
        "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
        "legend.labelspacing": 0.3, "legend.columnspacing": 0.9,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# ------------------------------------------------------------------ helpers
MISSING: set[str] = set()
MACROS: dict[str, str] = {}
EXPECTED = [
    "ECEMean", "ECEMax", "CovNinetyMean", "WidthRatioMean", "SpearmanMean",
    *[f"{p}{TAG[c]}" for p in ("ECE", "CovNinety", "WidthRatio", "SpearmanScore") for c in CITIES],
    "MinNegLogP", "DzMin", "DzMax", "NTests", *[f"DzTsk{TAG[c]}" for c in CITIES],
    *[f"FracPosTsk{TAG[c]}" for c in CITIES],
    "ChaosGainLow", "ChaosGainHigh", "RushLyapRatio", "SpearmanLyapErr",
    "TransferWithin", "TransferCross", "AdaptSamplesOurs", "AdaptSamplesTsk",
    "WaveCorr", "PeakDropCon", "PeakDropDes", "PeakDropRand",
    "AltAdditive", "AltConcat", "AltFourier", "AltWavelet", "AltSeasonal",
    "LphiOrdMax", "LphiMomSlope", "GammaSlopeRaw", "GammaSlopeNorm", "GammaTightness",
    "SuperTight", "DdivMean", "MeanProtoNorm", "NeffMin", "NeffMax", "RhoEpisodes",
    "RobustOurs", "RobustBestBase", "RobustBestBaseName", "RobustSigmaMax",
    "LfMedOurs", "LfPninetynineOurs", "LfBound", "LfTightness", "ShiftWithin", "ShiftCross",
    *[f"{p}{TAG[c]}" for p in ("AcfSixty", "TailFour", "FewShotGap", "FewShotGapFull") for c in CITIES],
]


def setm(name, val, fmt="{:.2f}"):
    MACROS[name] = val if isinstance(val, str) else fmt.format(val)


def art(name):
    p = ART / name
    if p.exists():
        return p
    MISSING.add(name)
    return None


def tci(sd, n=N_SEEDS):
    return stats.t.ppf(0.975, n - 1) * np.asarray(sd, float) / math.sqrt(n)


def save(fig, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / f"{name}.pdf")
    fig.savefig(RESULTS / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"[ok]   {name}")


def panel(ax, i, title):
    """Bold title on top; panel letter under the x-axis label."""
    ax.set_title(title, loc="center", pad=3, fontweight="bold")
    letter = r"$\mathbf{(" + chr(97 + i) + r")}$"
    xl = ax.get_xlabel()
    ax.set_xlabel(f"{xl}\n{letter}" if xl else letter)


def compose(name, panels, ncols, width=TEXT_W, row_h=1.8, shared_legend=None,
            legend_ncol=None, width_ratios=None):
    panels = [p for p in panels if p]
    if not panels:
        print(f"[skip] {name}: no inputs")
        return
    if len(panels) < ncols:
        width *= len(panels) / ncols if width == TEXT_W else 1
        if width < COL_W:
            width = COL_W
    ncols = min(ncols, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    extra = 0.22 if shared_legend is not None else 0
    fig = plt.figure(figsize=(width, row_h * nrows + extra), layout="constrained")
    wr = width_ratios if width_ratios and len(width_ratios) == ncols else None
    gs = fig.add_gridspec(nrows, ncols, width_ratios=wr)
    axes = []
    for i, (title, fn) in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        fn(ax)
        panel(ax, i, title)
        axes.append(ax)
    fig.align_xlabels(axes)
    if shared_legend is not None:
        h, l = axes[shared_legend].get_legend_handles_labels()
        fig.legend(h, l, loc="outside upper center", ncol=legend_ncol or len(l),
                   handlelength=2.2)
    save(fig, name)


def heat(ax, M, rows, cols, fmt, cmap, vmin, vmax, clabel, fs=6):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cm = plt.get_cmap(cmap)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=fs, color="0.45")
                continue
            r, g, b, _ = cm((np.clip(v, vmin, vmax) - vmin) / (vmax - vmin))
            dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=fs,
                    color="white" if dark else "black")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(clabel)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.4)


def clean_log(ax, axis="both", decades_only=False):
    fmt = matplotlib.ticker.FuncFormatter(
        lambda v, _: (f"$10^{{{int(round(np.log10(v)))}}}$" if decades_only else f"{v:g}") if v > 0 else "")
    for a in ([ax.xaxis, ax.yaxis] if axis == "both" else [getattr(ax, axis + "axis")]):
        if a.get_scale() == "log":
            a.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1,) if decades_only else (1, 2, 5)))
            a.set_major_formatter(fmt)
            a.set_minor_formatter(matplotlib.ticker.NullFormatter())


# ------------------------------------------------------------------ raw data
@lru_cache(maxsize=None)
def load_series(city):
    rel = CFG.get("data", {}).get(KEY[city], {}).get("dataset_path", f"data/{KEY[city]}/dataset.npy")
    p = ROOT / rel
    if not p.exists():
        MISSING.add(str(rel))
        return None
    a = np.load(p, mmap_mode="r")
    t_ax = [i for i, s in enumerate(a.shape) if s == N_STEPS[city]]
    a = np.moveaxis(np.asarray(a, dtype=np.float32), t_ax[0] if t_ax else 0, 0)
    if a.ndim == 3:
        n_ax = [i for i in (1, 2) if a.shape[i] == N_NODES[city]]
        a = np.moveaxis(a, n_ax[0] if n_ax else 1, 1)[:, :, 0]
    return a


def split_idx(n):
    return int(round(SPLIT[0] * n, 6)), int(round((SPLIT[0] + SPLIT[1]) * n, 6))


# ============================================================ protocol (C3)
def fig_protocol():
    fig = plt.figure(figsize=(TEXT_W, 1.75), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.0, 1.6])

    ax = fig.add_subplot(gs[0])
    c_tr, c_va, c_te = "#BBD7EE", "#F6D7A7", "#E8B4B8"
    for i, c in enumerate(CITIES):
        days = N_STEPS[c] * STEP_MIN[c] / 1440
        tr, va = SPLIT[0] * days, (SPLIT[0] + SPLIT[1]) * days
        y = len(CITIES) - 1 - i
        ax.barh(y, tr, color=c_tr, height=0.6, lw=0)
        ax.barh(y, va - tr, left=tr, color=c_va, height=0.6, lw=0)
        ax.barh(y, days - va, left=va, color=c_te, height=0.6, lw=0)
        fs_left = tr - TARGET_DAYS if FEWSHOT_AT_END_OF_TRAIN else 0
        ax.barh(y, TARGET_DAYS, left=fs_left, height=0.6, fill=False, hatch="//////",
                edgecolor="#D55E00", lw=0.6)
        ax.text(days + 1.5, y, f"{days:.0f} d, {STEP_MIN[c]} min", va="center", fontsize=6.5)
    ax.set_yticks(range(len(CITIES)), CITIES[::-1])
    ax.set_xlabel("Days (chronological)")
    ax.set_xlim(0, 225)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=c_tr, label="Train 70%"), Patch(color=c_va, label="Val 10%"),
                       Patch(color=c_te, label="Test 20%"),
                       Patch(fill=False, hatch="//////", edgecolor="#D55E00",
                             label=f"Target support ({TARGET_DAYS} d)")],
              loc="upper left", ncol=4, fontsize=6, bbox_to_anchor=(0, 1.0), handlelength=1.2,
              columnspacing=0.7)
    ax.set_ylim(-0.5, len(CITIES) + 0.15)
    panel(ax, 0, "Chronological splits")

    ax = fig.add_subplot(gs[1])
    M = np.ones((4, 4))
    np.fill_diagonal(M, 2)
    ax.imshow(M, cmap=ListedColormap(["#CFE3F3", "#F4B183"]), vmin=1, vmax=2)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, "T" if i == j else "S", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(4), [SHORT[c] for c in CITIES])
    ax.set_yticks(range(4), [f"Run {k + 1}" for k in range(4)])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    panel(ax, 1, "Leave-one-city-out")

    ax = fig.add_subplot(gs[2])
    for y, step in [(1, 5), (0, 10)]:
        g = np.arange(step, 61, step)
        ax.plot(g, np.full_like(g, y), "|", color="0.55", ms=6)
    ax.plot([15, 30, 60], [1, 1, 1], "o", color="#0072B2", ms=4, label="Evaluated")
    ax.plot([5], [1], "o", color="#0072B2", ms=4)
    ax.plot([10, 30, 60], [0, 0, 0], "o", color="#0072B2", ms=4)
    ax.plot([15], [0], "x", color="#D55E00", ms=5, mew=1.2, label="Not on grid")
    ax.set_ylim(-0.6, 1.9)
    ax.set_xlim(0, 63)
    ax.set_yticks([0, 1], ["10-min data", "5-min data"])
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([5, 10, 15, 30, 60])
    ax.set_xlabel("Horizon (min)")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", ncol=2, fontsize=6, bbox_to_anchor=(0.0, 1.02))
    panel(ax, 2, "Horizon grids")
    fig.align_xlabels(fig.axes)
    save(fig, "data_protocol")


def dataset_table():
    rows = []
    for c in CITIES:
        n = N_STEPS[c]
        tr, va = split_idx(n)
        fs = TARGET_DAYS * 1440 // STEP_MIN[c]
        X = load_series(c)
        mean = std = edges = "??"
        if X is not None:
            v = X[:tr][X[:tr] > 0]
            mean, std = f"{v.mean():.2f}", f"{v.std():.2f}"
        adj_rel = CFG.get("data", {}).get(KEY[c], {}).get("adjacency_matrix_path", f"data/{KEY[c]}/matrix.npy")
        if (ROOT / adj_rel).exists():
            A = np.load(ROOT / adj_rel)
            edges = f"{int((A > 0).sum()):,}"
        rows.append(f"{c} & {N_NODES[c]} & {edges} & {STEP_MIN[c]} & {n:,} & {n * STEP_MIN[c] / 1440:.0f} "
                    f"& [0, {tr:,}) & [{tr:,}, {va:,}) & [{va:,}, {n:,}) & {fs} & {mean} & {std} \\\\")
    RESULTS.mkdir(parents=True, exist_ok=True)
    head = [r"\begin{tabular}{lrrrrrcccrrr}", r"\toprule",
            r"Dataset & Nodes & Edges & Interval (min) & Steps & Days & Train & Validation & Test & "
            r"Support steps & Mean & Std \\", r"\midrule"]
    (RESULTS / "dataset_table.tex").write_text("\n".join(head + rows + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    print("[ok]   dataset_table.tex")


# ============================================================ statistics
def dm_test(d, lag):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    z = d.mean() / math.sqrt(max(lrv, 1e-300) / n)
    return z, (stats.norm.logsf(abs(z)) + math.log(2)) / math.log(10)


def holm_log10(lp):
    lp = np.asarray(lp, float)
    m, out, run = lp.size, np.empty(lp.size), -np.inf
    for i, j in enumerate(np.argsort(lp)):
        run = max(run, lp[j] + math.log10(m - i))
        out[j] = min(0.0, run)
    return out


@lru_cache(maxsize=None)
def paired_stats(h_idx=2):
    rows = []
    for c in CITIES:
        f = art(f"errors_{KEY[c]}.npz")
        if f is None:
            continue
        z = np.load(f)
        if OURS not in z.files:
            continue
        ours = z[OURS][:, h_idx]
        steps = int(round(HORIZONS[c][h_idx] / STEP_MIN[c]))
        for b in BASELINES:
            if b not in z.files:
                continue
            e = z[b][:, h_idx]
            ok = np.isfinite(e) & np.isfinite(ours)
            d = e[ok] - ours[ok]
            zs, lp = dm_test(d, T_IN + steps)
            rows.append(dict(city=c, baseline=b, z=zs, log10p=lp, dz=d.mean() / d.std(ddof=1),
                             diff=d, frac=float((d > 0).mean())))
    df = pd.DataFrame(rows)
    if len(df):
        df["log10p_holm"] = holm_log10(df.log10p.values)
        setm("MinNegLogP", float(-df.log10p_holm.max()), "{:.1f}")
        setm("DzMin", df.dz.min())
        setm("DzMax", df.dz.max())
        setm("NTests", str(len(df)))
        for _, r in df[df.baseline == "TSKNET"].iterrows():
            setm(f"DzTsk{TAG[r.city]}", r.dz)
            setm(f"FracPosTsk{TAG[r.city]}", 100 * r.frac, "{:.1f}")
    return df


def p_diff_hist(city="METR-LA", base="TSKNET"):
    df = paired_stats()
    if df.empty or not ((df.city == city) & (df.baseline == base)).any():
        return None
    r = df[(df.city == city) & (df.baseline == base)].iloc[0]

    def fn(ax):
        d = r["diff"]
        lo, hi = np.percentile(d, [0.5, 99.5])
        ax.hist(np.clip(d, lo, hi), bins=50, color=MSTYLE[base][0], alpha=0.75, lw=0)
        ax.axvline(0, color="k", lw=0.7)
        ax.axvline(d.mean(), color=MSTYLE[OURS][0], ls="--", lw=1)
        ax.set_xlabel(f"{base} − {OURS} error")
        ax.set_ylabel("Test windows")
        ax.text(0.02, 0.97, f"mean {d.mean():.3f}\n{100 * r.frac:.1f}% > 0", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.5, bbox=dict(fc="white", ec="none", pad=0.5))
    return (f"Paired differences, {SHORT[city]}", fn)


def p_stat_heat(kind):
    df = paired_stats()
    if df.empty:
        return None
    cities = [c for c in CITIES if c in set(df.city)]
    M = np.full((len(BASELINES), len(cities)), np.nan)
    for _, r in df.iterrows():
        M[BASELINES.index(r.baseline), cities.index(r.city)] = -r.log10p_holm if kind == "p" else r.dz
    cols = [SHORT[c] for c in cities]

    def fn(ax):
        if kind == "p":
            heat(ax, M, BASELINES, cols, lambda v: ">99" if v > 99 else f"{v:.0f}", "Blues", 0, 20,
                 r"$-\log_{10}p$ (Holm)")
        else:
            lim = max(1.0, float(np.nanmax(np.abs(M))))
            heat(ax, M, BASELINES, cols, lambda v: f"{v:.2f}", "RdBu", -lim, lim, r"Cohen's $d_z$")
            ax.set_yticklabels([])
    return ("DM test, 30 min" if kind == "p" else "Effect size, 30 min", fn)


def fig_statistics():
    compose("statistical_significance_testing",
            [p_diff_hist(), p_stat_heat("p"), p_stat_heat("d")],
            ncols=3, width=TEXT_W, row_h=1.9)


# ============================================================ chaos
def load_npz(name):
    f = art(name)
    if f is None:
        return None
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


def lyap_key(d):
    return next((k for k in d if k.startswith("ind_") and "lyap" in k.lower()), None)


def chaos_data():
    out = {}
    for c in CITIES:
        d = load_npz(f"chaos_{KEY[c]}.npz")
        if d is not None and lyap_key(d):
            out[c] = d
    return out


PRETTY = {"lyapunov": "LLE", "sample_entropy": "SampEn", "correlation_dim": "CorrDim",
          "hurst": "Hurst", "fractal_dim": "FracDim", "multifractal": "MF width",
          "permutation_entropy": "PermEn"}


def p_chaos_hour(data):
    if not data:
        return None

    def fn(ax):
        for a, b in RUSH_H:
            ax.axvspan(a, b, color="0.9", lw=0, zorder=0)
        ratios = []
        for c, d in data.items():
            h, v = d["hour"].astype(int), d[lyap_key(d)]
            q = np.array([np.nanpercentile(v[h == k], [25, 50, 75]) if (h == k).any() else [np.nan] * 3
                          for k in range(24)])
            col, mk = CSTYLE[c]
            ax.plot(range(24), q[:, 1], color=col, marker=mk, ms=2.2, markevery=3, label=c)
            ax.fill_between(range(24), q[:, 0], q[:, 2], color=col, alpha=0.07, lw=0)
            rush = np.isin(h, [k for a, b in RUSH_H for k in range(a, b)])
            ratios.append(np.nanmedian(v[rush]) / np.nanmedian(v[~rush]))
        setm("RushLyapRatio", float(np.mean(ratios)))
        ax.set_xlim(0, 23)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Largest Lyapunov exp.")
        ax.legend(ncol=2, loc="upper left")
    return ("Daily chaos profile", fn)


def p_chaos_gain(data):
    eo, eb = f"err_{OURS}", "err_TSKNET"
    data = {c: d for c, d in data.items() if eo in d and eb in d}
    if not data:
        return None

    def fn(ax):
        lows, highs, rhos = [], [], []
        for c, d in data.items():
            v = d[lyap_key(d)]
            q = pd.qcut(v, 5, labels=False, duplicates="drop")
            g = [100 * (1 - d[eo][q == k].mean() / d[eb][q == k].mean()) for k in range(int(q.max()) + 1)]
            col, mk = CSTYLE[c]
            ax.plot(range(1, len(g) + 1), g, color=col, marker=mk, label=c)
            lows.append(g[0])
            highs.append(g[-1])
            rhos.append(stats.spearmanr(v, d[eo], nan_policy="omit").statistic)
        setm("ChaosGainLow", float(np.mean(lows)), "{:.1f}")
        setm("ChaosGainHigh", float(np.mean(highs)), "{:.1f}")
        setm("SpearmanLyapErr", float(np.mean(rhos)))
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Lyapunov quintile (1 = lowest)")
        ax.set_ylabel("MAE reduction vs. TSKNET (%)")
    return ("Gain vs. chaos level", fn)


def p_chaos_corr(data):
    if not data:
        return None
    pref = ["ind_lyapunov", "ind_correlation_dim", "ind_sample_entropy", "ind_permutation_entropy",
            "ind_fractal_dim", "ind_hurst"]
    common = set.intersection(*[{k for k in d if k.startswith("ind_")} for d in data.values()])
    keys = [k for k in pref if k in common] + sorted(common - set(pref))
    if len(keys) < 2:
        return None
    df = pd.concat([pd.DataFrame({k: d[k] for k in keys}) for d in data.values()])
    C = df.corr(method="spearman").values
    lab = [PRETTY.get(k[4:].lower(), k[4:12].title()) for k in keys]

    def fn(ax):
        heat(ax, C, lab, lab, lambda v: f"{v + 0.0:.2f}".replace("-0.00", "0.00"), "RdBu_r", -1, 1,
             r"Spearman $\rho$")
        ax.tick_params(axis="x", rotation=30)
    return ("Indicator correlation", fn)


def fig_chaos():
    data = chaos_data()
    panels = [p for p in (p_chaos_hour(data), p_chaos_gain(data), p_chaos_corr(data)) if p]
    compose("chaos_feature_analysis", panels, ncols=3, width=TEXT_W, row_h=2.0,
            width_ratios=[1.25, 1.1, 1.0] if len(panels) == 3 else None)


# ============================================================ transfer
def p_transfer_matrix():
    f = art("transfer_pairs.csv")
    if f is None:
        return None
    df = pd.read_csv(f)
    if "horizon" in df and (df.horizon == 30).any():
        df = df[df.horizon == 30]
    df["gain"] = 100 * (1 - df.mae_transfer / df.mae_scratch)
    g = df.groupby(["source", "target"]).gain.mean()
    M = np.full((4, 4), np.nan)
    within, cross = [], []
    for (s, t), v in g.items():
        if s in CITIES and t in CITIES and s != t:
            M[CITIES.index(s), CITIES.index(t)] = v
            (within if ROAD[s] == ROAD[t] else cross).append(v)
    if within:
        setm("TransferWithin", float(np.mean(within)), "{:.1f}")
    if cross:
        setm("TransferCross", float(np.mean(cross)), "{:.1f}")
    lim = max(5.0, float(np.nanmax(np.abs(M))))

    def fn(ax):
        heat(ax, M, [SHORT[c] for c in CITIES], [SHORT[c] for c in CITIES], lambda v: f"{v:.1f}",
             "RdBu", -lim, lim, "MAE gain vs. scratch (%)")
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
    return ("Single-source transfer", fn)


def p_embedding():
    d = load_npz("embeddings.npz")
    if d is None:
        return None
    emb, lab = d["emb"], d["city"].astype(str)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(emb), min(3000, len(emb)), replace=False)
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(emb[idx])
    try:
        Y = TSNE(2, perplexity=30, random_state=0, init="pca", max_iter=1000).fit_transform(X)
    except TypeError:
        Y = TSNE(2, perplexity=30, random_state=0, init="pca", n_iter=1000).fit_transform(X)

    def fn(ax):
        for c in CITIES:
            m = lab[idx] == c
            if m.any():
                ax.scatter(Y[m, 0], Y[m, 1], s=2, alpha=0.5, color=CSTYLE[c][0], label=c, lw=0,
                           rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color("0.75")
            sp.set_linewidth(0.5)
        ax.legend(markerscale=3, ncol=2, loc="best", fontsize=6)
    return ("Learned chaos embeddings", fn)


def p_adaptation():
    f = art("adaptation.csv")
    if f is None:
        return None
    df = pd.read_csv(f)
    target = CFG.get("training", {}).get("test_dataset")
    names = {KEY[c]: c for c in CITIES}
    target = names.get(target, target)
    if target not in set(df.target):
        target = df.target.mode().iloc[0]
    df = df[df.target == target]
    for m, col in [(OURS, "AdaptSamplesOurs"), ("TSKNET", "AdaptSamplesTsk")]:
        s = df[df.model == m].groupby("n_samples").mae.mean()
        if len(s):
            setm(col, f"{int(s.index[(s <= 1.05 * s.min()).values][0]):,}")

    def fn(ax):
        for m, g in df.groupby("model"):
            s = g.groupby("n_samples").mae.agg(["mean", "std", "count"])
            c, mk, ls = MSTYLE.get(m, ("#444444", "o", "-"))
            ci = stats.t.ppf(0.975, np.maximum(s["count"] - 1, 1)) * s["std"].fillna(0) / np.sqrt(s["count"])
            ax.plot(s.index, s["mean"], color=c, marker=mk, ls=ls, lw=1.5 if m == OURS else 0.9,
                    label=m)
            ax.fill_between(s.index, s["mean"] - ci, s["mean"] + ci, color=c, alpha=0.15, lw=0)
        ax.set_xscale("log")
        clean_log(ax, "x")
        ax.set_xlabel("Target adaptation samples")
        ax.set_ylabel("MAE")
        ax.legend(fontsize=6)
    return (f"Adaptation on {target}", fn)


def fig_transfer():
    compose("cross_city_transfer_analysis", [p_transfer_matrix(), p_embedding(), p_adaptation()],
            ncols=3, width=TEXT_W, row_h=1.9)


# ============================================================ wave (C7)
def p_wave_recon():
    d = None
    for c in CITIES:
        d = load_npz(f"wave_example_{KEY[c]}.npz")
        if d is not None:
            city = c
            break
    if d is None:
        return None, None
    setm("WaveCorr", float(np.corrcoef(d["y"], d["recon"])[0, 1]), "{:.3f}")

    def recon(ax):
        for a, b in RUSH_H:
            ax.axvspan(a, b, color="0.92", lw=0, zorder=0)
        ax.plot(d["t_h"], d["y"], color="k", lw=0.9, label="Measured")
        ax.plot(d["t_h"], d["recon"], color=MSTYLE[OURS][0], ls="--", lw=1.1, label="Wave superposition")
        ax.set_xlim(d["t_h"].min(), d["t_h"].max())
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Speed")
        ax.legend(loc="lower left", ncol=2)

    def comps(ax):
        C = np.atleast_2d(d["components"])
        amp = np.abs(C).max(1)
        cols = ["#0072B2", "#009E73", "#CC79A7"]
        per = d.get("period_h")
        for k, i in enumerate(np.argsort(amp)[::-1][:3]):
            lab = f"Wave {i + 1}" + (f" ({float(per[i]):.1f} h)" if per is not None else "")
            ax.plot(d["t_h"], C[i], color=cols[k], lw=0.9, label=lab)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xlim(d["t_h"].min(), d["t_h"].max())
        ax.set_xlabel("Hour of day")
        ax.set_ylabel(r"$W^{(n)}$ (dominant 3)")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.35 * (hi - lo))
        ax.legend(loc="upper center", ncol=3, fontsize=6)

    p1 = (f"Measured vs. superposition, {SHORT[city]}", recon)
    p2 = ("Dominant learned waves", comps) if "components" in d else None
    return p1, p2


def psd_city(city, max_nodes=64):
    X = load_series(city)
    if X is None:
        return None
    tr, _ = split_idx(len(X))
    spd = 1440 // STEP_MIN[city]
    rng = np.random.default_rng(0)
    P, f = [], None
    for n in rng.choice(X.shape[1], min(max_nodes, X.shape[1]), replace=False):
        x = X[:tr, n].astype(float)
        bad = x <= 0
        if bad.mean() > 0.2:
            continue
        if bad.any():
            x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
        f, p = signal.welch(signal.detrend(x), fs=spd, nperseg=min(len(x), 7 * spd))
        P.append(p)
    return (f, np.mean(P, 0)) if P else None


def p_wave_psd():
    spectra = {c: psd_city(c) for c in CITIES}
    spectra = {c: s for c, s in spectra.items() if s is not None}
    if not spectra:
        return None
    freqs = None
    f = art("wave_freqs.npz")
    if f is not None:
        z = np.load(f)
        freqs = {k: z[k] for k in z.files}

    def fn(ax):
        fmax = 12.5
        for c, (fr, p) in spectra.items():
            m = (fr > 0) & (fr <= fmax)
            ax.semilogy(fr[m], p[m] / p[m].max(), color=CSTYLE[c][0], lw=0.9, label=c)
        for fx, lab in [(1, "24 h"), (2, "12 h"), (3, "8 h"), (4, "6 h"), (12, "2 h")]:
            ax.axvline(fx, color="0.6", lw=0.5, ls=":")
            ax.text(fx, 1.5, lab, ha="center", va="bottom", fontsize=6)
        lo = min((p[(fr > 0) & (fr <= fmax)] / p[(fr > 0) & (fr <= fmax)].max()).min()
                 for fr, p in spectra.values())
        if freqs:
            for c, fv in freqs.items():
                fv = fv[(fv > 0) & (fv <= fmax)]
                ax.plot(fv, np.full_like(fv, lo * 0.4), "|", ms=6, mew=1.0,
                        color=CSTYLE.get({KEY[k]: k for k in CITIES}.get(c, c), ("k", ""))[0])
            ax.text(fmax, lo * 0.55, "learned wave freq.", ha="right", va="bottom", fontsize=6)
        ax.set_xlim(0, fmax)
        ax.set_ylim(lo * 0.25, 4)
        ax.set_xticks([0, 1, 2, 3, 4, 6, 8, 10, 12])
        ax.set_xlabel("Frequency (cycles/day)")
        ax.set_ylabel("Normalised PSD")
        ax.legend(loc="upper right", ncol=2, fontsize=6, bbox_to_anchor=(1, 0.9))
    return ("Spectrum of measured speed (train)", fn)


def event_aligned(g, v, step_min, win_min=60, q=0.95):
    g, v = np.atleast_2d(np.asarray(g, float).T).T, np.atleast_2d(np.asarray(v, float).T).T
    if g.shape[0] < g.shape[1]:
        g, v = g.T, v.T
    W = win_min // step_min
    sep = max(1, 60 // step_min)
    segs = {"con": [], "des": [], "rand": []}
    rng = np.random.default_rng(0)
    for j in range(g.shape[1]):
        gj, vj = g[:, j], v[:, j]
        for key, sig in (("con", gj), ("des", -gj)):
            idx, _ = signal.find_peaks(sig, height=np.nanquantile(sig, q), distance=sep)
            idx = idx[(idx >= W) & (idx < len(gj) - W)]
            segs[key] += [vj[i - W:i + W + 1] - vj[i] for i in idx]
        for i in rng.integers(W, len(gj) - W, size=max(20, len(segs["con"]) // max(1, j + 1))):
            segs["rand"].append(vj[i - W:i + W + 1] - vj[i])
    tau = np.arange(-W, W + 1) * step_min
    res = {}
    for k, S in segs.items():
        if S:
            S = np.asarray(S)
            n = np.sum(np.isfinite(S), 0)
            res[k] = (np.nanmean(S, 0), 1.96 * np.nanstd(S, 0, ddof=1) / np.sqrt(np.maximum(n, 1)), len(S))
    return tau, res


def p_wave_events():
    for c in CITIES:
        d = load_npz(f"interference_{KEY[c]}.npz")
        if d is not None:
            break
    else:
        return None
    tau, res = event_aligned(d["gain"], d["speed"], STEP_MIN[c])
    i30 = int(np.argmin(np.abs(tau - 30)))
    for k, name in [("con", "PeakDropCon"), ("des", "PeakDropDes"), ("rand", "PeakDropRand")]:
        if k in res:
            setm(name, float(res[k][0][i30]))

    def fn(ax):
        sty = {"con": ("#D55E00", "Constructive peaks"), "des": ("#0072B2", "Destructive troughs"),
               "rand": ("0.5", "Random times")}
        for k, (mu, ci, n) in res.items():
            col, lab = sty[k]
            ax.plot(tau, mu, color=col, lw=1.1 if k != "rand" else 0.8,
                    ls="-" if k != "rand" else "--", label=f"{lab} (n={n:,})")
            ax.fill_between(tau, mu - ci, mu + ci, color=col, alpha=0.18, lw=0)
        ax.axvline(0, color="k", lw=0.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Time from event (min)")
        ax.set_ylabel(r"$\Delta$ speed from event time")
        ax.legend(loc="lower left", fontsize=6)
    return (f"Interference events vs. speed, {SHORT[c]}", fn)


def fig_wave():
    p1, p2 = p_wave_recon()
    compose("wave_interference_visualization", [p1, p2, p_wave_psd(), p_wave_events()],
            ncols=2, width=TEXT_W, row_h=1.8)


# ============================================================ theory
def phi_groups(X, sig_min=0.05, mu_min=0.25, L=6):
    n, T = X.shape
    t = np.arange(T, dtype=float)
    mu = X.mean(1)
    A = X - mu[:, None]
    m2 = (A ** 2).mean(1)
    sd = np.sqrt(np.maximum(m2, sig_min ** 2))
    acf = [(A[:, l:] * A[:, :-l]).sum(1) / (T * sd ** 2) for l in range(1, L + 1)]
    tc = t - t.mean()
    mu_s = np.where(mu >= 0, 1, -1) * np.maximum(np.abs(mu), mu_min)
    mom = np.column_stack([mu, m2, (A ** 3).mean(1) / sd ** 3, (A ** 4).mean(1) / sd ** 4,
                           *acf, A @ tc / (tc @ tc), sd / mu_s])
    dX = np.diff(X, axis=1)
    q25, q50, q75 = np.percentile(X, [25, 50, 75], axis=1)
    i, j = np.triu_indices(T, 1)
    ts = np.median((X[:, j] - X[:, i]) / (j - i), axis=1)
    ordg = np.column_stack([dX.max(1), dX.min(1), X.max(1) - X.min(1), q50,
                            np.median(np.abs(X - q50[:, None]), axis=1), q75 - q25, ts,
                            np.linalg.norm(X, axis=1)])
    return mom, ordg


@lru_cache(maxsize=None)
def lemma1_curves(city, Ts=(6, 12, 24, 48, 96), n_win=300, n_dir=16, eps=1e-3):
    X = load_series(city)
    if X is None:
        return None
    tr, _ = split_idx(len(X))
    Xtr = X[:tr]
    v = Xtr[Xtr > 0]
    mu, sd = float(v.mean()), float(v.std())
    rng = np.random.default_rng(0)
    res = {"T": [], "mom": [], "ord": []}
    for T in Ts:
        wins, tries = [], 0
        while len(wins) < n_win and tries < 100 * n_win:
            tries += 1
            t0, nd = rng.integers(0, tr - T), rng.integers(0, X.shape[1])
            w = Xtr[t0:t0 + T, nd]
            if np.all(w > 0):
                z = (w - mu) / sd
                if z.std() >= 0.05 and abs(z.mean()) >= 0.25:
                    wins.append(z)
        if len(wins) < 20:
            continue
        W = np.asarray(wins, float)
        U = rng.standard_normal((len(W), n_dir, T))
        U /= np.linalg.norm(U, axis=2, keepdims=True)
        X0 = np.repeat(W, n_dir, axis=0)
        X1 = (W[:, None, :] + eps * U).reshape(-1, T)
        m0, o0 = phi_groups(X0)
        m1, o1 = phi_groups(X1)
        rm = (np.linalg.norm(m1 - m0, axis=1) / eps).reshape(len(W), n_dir).max(1)
        ro = (np.linalg.norm(o1 - o0, axis=1) / eps).reshape(len(W), n_dir).max(1)
        res["T"].append(T)
        res["mom"].append(np.percentile(rm, 99))
        res["ord"].append(np.percentile(ro, 99))
    return {k: np.asarray(v) for k, v in res.items()} if res["T"] else None


def p_lemma1():
    curves = {c: lemma1_curves(c) for c in CITIES}
    curves = {c: r for c, r in curves.items() if r is not None}
    if not curves:
        return None
    setm("LphiOrdMax", float(max(r["ord"].max() for r in curves.values())))
    sl = [np.polyfit(np.log(r["T"]), np.log(r["mom"]), 1)[0] for r in curves.values()]
    setm("LphiMomSlope", f"{min(sl):.2f} to {max(sl):.2f}")

    def fn(ax):
        for c, r in curves.items():
            col = CSTYLE[c][0]
            ax.loglog(r["T"], r["mom"], color=col, marker="o", ms=2.5, lw=0.9)
            ax.loglog(r["T"], r["ord"], color=col, marker="s", ms=2.5, lw=0.9, ls="--")
        r0 = next(iter(curves.values()))
        ax.loglog(r0["T"], r0["mom"][0] * (r0["T"] / r0["T"][0]) ** -0.5, color="k", lw=0.6, ls=":")
        ax.axhline(math.sqrt(26), color="k", lw=0.8)
        ax.text(r0["T"][0], math.sqrt(26) * 0.85, r"$\sqrt{26}$ ($\Phi_{\rm ord}$ bound)", ha="left",
                va="top", fontsize=6)
        hs = [Line2D([], [], color="k", marker="o", ms=2.5, label=r"$\Phi_{\rm mom}$"),
              Line2D([], [], color="k", marker="s", ms=2.5, ls="--", label=r"$\Phi_{\rm ord}$"),
              Line2D([], [], color="k", ls=":", label=r"$T^{-1/2}$")]
        hs += [Line2D([], [], color=CSTYLE[c][0], label=SHORT[c]) for c in curves]
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 8)
        ax.legend(handles=hs, ncol=4, fontsize=5.6, loc="upper center", handlelength=1.3,
                  columnspacing=0.6)
        ax.set_xticks(r0["T"], [str(int(t)) for t in r0["T"]])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        clean_log(ax, "y")
        ax.set_xlabel("Window length $T$")
        ax.set_ylabel("Empirical Lipschitz ratio (p99)")
    return ("Lemma 1 on raw data", fn)


def p_gamma():
    f = art("gamma.csv")
    if f is None:
        return None
    df = pd.read_csv(f)

    def fn(ax):
        for idx, col in [("raw", "#D55E00"), ("normalised", "#0072B2")]:
            g = df[df["index"] == idx].groupby("T").agg(fd=("fd", "max"), bound=("bound", "max")).reset_index()
            if g.empty:
                continue
            ax.loglog(g["T"], g["bound"], color=col, lw=1.0, label=f"bound, {idx}")
            ax.loglog(g["T"], g["fd"], color=col, marker="o", ms=2.5, ls="--", lw=0.8,
                      label=f"finite diff., {idx}")
            if len(g) > 1:
                setm("GammaSlopeRaw" if idx == "raw" else "GammaSlopeNorm",
                     float(np.polyfit(np.log(g["T"]), np.log(g["fd"]), 1)[0]))
        setm("GammaTightness", float(np.median(df.fd / df.bound)), "{:.3f}")
        Ts = sorted(df["T"].unique())
        ax.set_xticks(Ts, [str(int(t)) for t in Ts])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        clean_log(ax, "y")
        ax.set_xlabel("Window length $T$")
        ax.set_ylabel(r"$\Gamma$")
        ax.legend(fontsize=5.8, loc="upper left")
    return (r"Chaos-path constant $\Gamma$", fn)


def p_superposition():
    f = art("superposition.csv")
    if f is None:
        return None
    df = pd.read_csv(f).groupby("N_w").mean(numeric_only=True).reset_index()
    setm("SuperTight", float((df.var_emp / df.bound).max()), "{:.3f}")

    def fn(ax):
        ax.semilogy(df.N_w, df.bound, color="k", lw=1.0, label="Lemma bound")
        ax.semilogy(df.N_w, df.var_emp, color=MSTYLE[OURS][0], marker="o", ms=3, ls="--",
                    label="Empirical")
        ax.set_xticks(df.N_w.astype(int))
        ax.set_xlabel("$N_w$")
        ax.set_ylabel(r"Var$(\mathbf{W}_{\rm sum})$")
        clean_log(ax, "y")
        ax.legend(fontsize=6)
    return ("Wave superposition", fn)


def p_prototypes():
    f = art("prototypes.csv")
    if f is None:
        return None
    df = pd.read_csv(f)
    setm("DdivMean", float(df.d_div.mean()))
    if "mean_norm_rel" in df:
        setm("MeanProtoNorm", float(df.mean_norm_rel.max()), "{:.3f}")
    e = art("episodes.csv")
    if e is not None:
        ep = pd.read_csv(e)
        setm("NeffMin", float(ep.n_eff.min()), "{:.1f}")
        setm("NeffMax", float(ep.n_eff.max()), "{:.1f}")
        setm("RhoEpisodes", float(ep.rho_plus.mean()), "{:.3f}")

    def fn(ax):
        x = np.arange(len(df))
        names = [SHORT.get({KEY[k]: k for k in CITIES}.get(t, t), str(t)) for t in df.target]
        ax.bar(x, df.d_div, color="#0072B2", width=0.6)
        for i, r in df.iterrows():
            lb = -1 / (r.n_sources - 1)
            ax.hlines(lb, i - 0.35, i + 0.35, color="#D55E00", lw=1.2)
        ax.axhline(0, color="k", lw=0.5)
        if "mean_norm_rel" in df:
            names = [f"{n}\n{v:.2f}" for n, v in zip(names, df.mean_norm_rel)]
        ax.set_xticks(x, names)
        ax.set_xlabel(r"Target city / $\|\bar{\mathbf{p}}\|/\overline{\|\mathbf{p}\|}$")
        ax.set_ylabel(r"$\mathcal{D}_{\rm div}$")
        ax.legend(handles=[Patch(color="#0072B2", label="measured"),
                           Line2D([], [], color="#D55E00", lw=1.2, label=r"min $-1/(|\mathcal{C}|-1)$")],
                  fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 0.1))
    return ("Prototype diversity", fn)


def fig_theory():
    compose("theoretical_validation", [p_gamma(), p_superposition(), p_prototypes()],
            ncols=2, width=COL_W, row_h=1.7)


# ============================================================ data characteristics (raw data only)
WEEKEND_IDX = {"METR-LA": (2, 3), "PEMS-BAY": (0, 6), "Chengdu": (5, 6), "Shenzhen": (5, 6)}


@lru_cache(maxsize=None)
def daily_cube(city):
    X = load_series(city)
    if X is None:
        return None
    spd = 1440 // STEP_MIN[city]
    nd = X.shape[0] // spd
    return X[:nd * spd].reshape(nd, spd, X.shape[1]).astype(float)


@lru_cache(maxsize=None)
def residual_train(city):
    """Training-split speed minus each sensor's weekday/weekend mean daily profile."""
    C = daily_cube(city)
    if C is None:
        return None
    nd_tr = int(round(SPLIT[0] * C.shape[0]))
    C = C[:nd_tr].copy()
    C[C <= 0] = np.nan
    we = np.isin(np.arange(nd_tr) % 7, WEEKEND_IDX[city])
    R = np.empty_like(C)
    for m in (we, ~we):
        R[m] = C[m] - np.nanmean(C[m], 0, keepdims=True)
    return (R.reshape(-1, C.shape[2]).astype(np.float32), C.reshape(-1, C.shape[2]).astype(np.float32))


def p_profiles():
    cubes = {c: daily_cube(c) for c in CITIES}
    cubes = {c: v for c, v in cubes.items() if v is not None}
    if not cubes:
        return None

    def fn(ax):
        for c, C in cubes.items():
            nd_tr = int(round(SPLIT[0] * C.shape[0]))
            D = np.where(C[:nd_tr] > 0, C[:nd_tr], np.nan)
            we = np.isin(np.arange(nd_tr) % 7, WEEKEND_IDX[c])
            mu = np.nanmean(D)
            h = np.arange(C.shape[1]) * STEP_MIN[c] / 60
            col = CSTYLE[c][0]
            ax.plot(h, np.nanmean(D[~we], (0, 2)) / mu, color=col, lw=1.0, label=c)
            ax.plot(h, np.nanmean(D[we], (0, 2)) / mu, color=col, lw=0.7, ls=":")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Speed / city mean")
        hs, ls = ax.get_legend_handles_labels()
        hs += [Line2D([], [], color="k", lw=1.0), Line2D([], [], color="k", lw=0.7, ls=":")]
        ls += ["weekday", "weekend"]
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.42 * (hi - lo))
        ax.legend(hs, ls, ncol=3, fontsize=5.6, loc="upper right", handlelength=1.3, columnspacing=0.6)
    return ("Daily profile (train)", fn)


def p_acf(max_min=180):
    res = {c: residual_train(c) for c in CITIES}
    res = {c: r for c, r in res.items() if r is not None}
    if not res:
        return None
    out = {}
    for c, (R, _) in res.items():
        L = max_min // STEP_MIN[c]
        cols = np.random.default_rng(0).choice(R.shape[1], min(100, R.shape[1]), replace=False)
        acf = []
        for j in cols:
            x = R[:, j]
            x = x[np.isfinite(x)]
            if x.size < 1000:
                continue
            x = x - x.mean()
            v = x @ x
            acf.append([1.0] + [(x[l:] @ x[:-l]) / v for l in range(1, L + 1)])
        out[c] = (np.arange(L + 1) * STEP_MIN[c], np.mean(acf, 0))
    for c, (lag, a) in out.items():
        i = np.searchsorted(lag, 60)
        setm(f"AcfSixty{TAG[c]}", float(a[i]))

    def fn(ax):
        for c, (lag, a) in out.items():
            ax.plot(lag, a, color=CSTYLE[c][0], marker=CSTYLE[c][1], ms=2, markevery=max(1, 30 // STEP_MIN[c]),
                    label=c)
        for h in (15, 30, 60):
            ax.axvline(h, color="0.6", lw=0.5, ls=":")
        ax.set_xlim(0, max_min)
        ax.set_xticks([0, 15, 30, 60, 120, 180])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Lag (min)")
        ax.set_ylabel("ACF of de-seasonalised speed")
        ax.legend(fontsize=6, loc="upper right")
    return ("Temporal memory", fn)


def p_spatial(max_hop=8, n_pairs=6000):
    from scipy.sparse.csgraph import shortest_path
    out = {}
    for c in CITIES:
        res = residual_train(c)
        rel = CFG.get("data", {}).get(KEY[c], {}).get("adjacency_matrix_path", f"data/{KEY[c]}/matrix.npy")
        if res is None or not (ROOT / rel).exists():
            continue
        A = np.load(ROOT / rel)
        Hd = shortest_path((A > 0) | (A.T > 0), unweighted=True, directed=False)
        R = res[0]
        rng = np.random.default_rng(0)
        Z = (R - np.nanmean(R, 0)) / np.nanstd(R, 0)
        Z = np.nan_to_num(Z)
        i, j = rng.integers(0, A.shape[0], (2, n_pairs))
        keep = i != j
        i, j = i[keep], j[keep]
        near = np.argwhere((Hd >= 1) & (Hd <= 3))
        near = near[rng.choice(len(near), min(n_pairs, len(near)), replace=False)]
        i, j = np.concatenate([i, near[:, 0]]), np.concatenate([j, near[:, 1]])
        Z = Z.astype(np.float32)
        r = np.concatenate([np.einsum("ti,ti->i", Z[:, i[a:a + 400]], Z[:, j[a:a + 400]]) / Z.shape[0]
                            for a in range(0, len(i), 400)])
        h = Hd[i, j]
        hops = np.arange(1, max_hop + 1)
        med = [np.median(r[h == k]) if (h == k).sum() >= 20 else np.nan for k in hops]
        far = np.median(r[~np.isfinite(h) | (h > max_hop)]) if (~np.isfinite(h) | (h > max_hop)).sum() >= 20 else np.nan
        out[c] = (hops, np.asarray(med), far)
    if not out:
        return None

    def fn(ax):
        for c, (hops, med, far) in out.items():
            ax.plot(hops, med, color=CSTYLE[c][0], marker=CSTYLE[c][1], ms=2.5, label=c)
            if np.isfinite(far):
                ax.plot([max_hop + 1.2], [far], color=CSTYLE[c][0], marker=CSTYLE[c][1], ms=3, ls="none")
        ax.set_xticks(list(range(1, max_hop + 1)) + [max_hop + 1.2],
                      [str(k) for k in range(1, max_hop + 1)] + [f">{max_hop}"])
        ax.set_xlabel("Graph distance (hops)")
        ax.set_ylabel("Median residual correlation")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        ax.legend(fontsize=6, loc="upper right", ncol=2)
    return ("Spatial dependence", fn)


def _change_hist(city, minutes=30, bins=np.linspace(-6, 6, 121)):
    res = residual_train(city)
    if res is None:
        return None
    raw = res[1]
    k = minutes // STEP_MIN[city]
    d = raw[k:] - raw[:-k]
    d = d[np.isfinite(d)]
    d = d / np.std(d)
    h, _ = np.histogram(np.clip(d, bins[0], bins[-1]), bins=bins)
    return (h + 1e-12) / h.sum(), d


def p_shift():
    hists = {c: _change_hist(c) for c in CITIES}
    hists = {c: v for c, v in hists.items() if v is not None}
    if len(hists) < 2:
        return None
    from scipy.spatial.distance import jensenshannon
    cs = list(hists)
    M = np.array([[jensenshannon(hists[a][0], hists[b][0], base=2) if a != b else np.nan for b in cs] for a in cs])
    within = [M[i, j] for i, a in enumerate(cs) for j, b in enumerate(cs) if i < j and ROAD[a] == ROAD[b]]
    cross = [M[i, j] for i, a in enumerate(cs) for j, b in enumerate(cs) if i < j and ROAD[a] != ROAD[b]]
    if within:
        setm("ShiftWithin", float(np.mean(within)), "{:.3f}")
    if cross:
        setm("ShiftCross", float(np.mean(cross)), "{:.3f}")

    def fn(ax):
        heat(ax, M, [SHORT[c] for c in cs], [SHORT[c] for c in cs], lambda v: f"{v:.3f}", "Purples",
             0, float(np.nanmax(M)) * 1.05, "JS distance")
    return ("Cross-city shift, 30-min change", fn)


def p_tails():
    hists = {c: _change_hist(c) for c in CITIES}
    hists = {c: v for c, v in hists.items() if v is not None}
    if not hists:
        return None
    xs = np.linspace(0, 8, 81)

    def fn(ax):
        for c, (_, d) in hists.items():
            a = np.sort(np.abs(d))
            ccdf = 1 - np.searchsorted(a, xs) / a.size
            ax.semilogy(xs, np.where(ccdf > 0, ccdf, np.nan), color=CSTYLE[c][0], label=c)
            setm(f"TailFour{TAG[c]}", float((np.abs(d) > 4).mean() / (2 * stats.norm.sf(4))), "{:.0f}")
        ax.semilogy(xs, 2 * stats.norm.sf(xs), color="k", ls="--", lw=0.8, label="Gaussian")
        ax.set_ylim(1e-6, 1.5)
        ax.set_xlim(0, 8)
        ax.set_xlabel(r"$|\Delta v_{30}|/\sigma$")
        ax.set_ylabel("Exceedance probability")
        clean_log(ax, "y", decades_only=True)
        ax.legend(fontsize=6, loc="lower left")
    return ("Heavy-tailed changes", fn)


def p_fewshot(max_days=21):
    out = {}
    for c in CITIES:
        C = daily_cube(c)
        if C is None:
            continue
        nd = C.shape[0]
        tr_end = int(round(SPLIT[0] * nd))
        te0 = int(round((SPLIT[0] + SPLIT[1]) * nd))
        rng = np.random.default_rng(0)
        cols = rng.choice(C.shape[2], min(60, C.shape[2]), replace=False)

        def prof(days):
            D = np.where(C[days][:, :, cols] > 0, C[days][:, :, cols], np.nan)
            we = np.isin(days % 7, WEEKEND_IDX[c])
            return np.nanmean(D[~we], 0) if (~we).any() else np.nanmean(D, 0)

        test = prof(np.arange(te0, nd))
        ks = np.arange(1, max_days + 1)
        err = []
        for k in ks:
            p = prof(np.arange(tr_end - k, tr_end))
            err.append(np.nanmean(np.abs(p - test)) / np.nanmean(test))
        out[c] = (ks, 100 * np.asarray(err))
    if not out:
        return None
    for c, (ks, e) in out.items():
        setm(f"FewShotGap{TAG[c]}", float(e[TARGET_DAYS - 1]), "{:.1f}")
        setm(f"FewShotGapFull{TAG[c]}", float(e[-1]), "{:.1f}")

    def fn(ax):
        for c, (ks, e) in out.items():
            ax.plot(ks, e, color=CSTYLE[c][0], marker=CSTYLE[c][1], ms=2, markevery=3, label=c)
        ax.axvline(TARGET_DAYS, color="#D55E00", lw=0.9, ls="--")
        ax.text(TARGET_DAYS + 0.3, ax.get_ylim()[1] * 0.95, f"{TARGET_DAYS}-day budget", fontsize=6,
                color="#D55E00", va="top")
        ax.set_xlabel("Target days available (end of train split)")
        ax.set_ylabel("Weekday-profile gap to test (%)")
        ax.set_xlim(0.5, max_days + 0.5)
        ax.legend(fontsize=6, loc="upper right")
    return ("How representative is the support?", fn)


def fig_data_characteristics():
    compose("data_characteristics",
            [p_profiles(), p_acf(), p_spatial(), p_shift(), p_tails(), p_fewshot()],
            ncols=3, width=TEXT_W, row_h=1.85)


# ============================================================ input stability (Prop. 1)
def p_robust_curve():
    f = art("robustness.csv")
    if f is None:
        return None
    df = pd.read_csv(f)
    base = df[df.sigma == 0].groupby(["model", "dataset", "seed"]).mae.first().rename("mae0")
    df = df.join(base, on=["model", "dataset", "seed"])
    df["rel"] = 100 * (df.mae / df.mae0 - 1)
    g = df.groupby(["model", "sigma"]).rel.agg(["mean", "std", "count"]).reset_index()
    smax = g.sigma.max()
    at = g[g.sigma == smax].set_index("model")["mean"]
    if OURS in at:
        setm("RobustOurs", float(at[OURS]), "{:.1f}")
        others = at.drop(OURS)
        if len(others):
            setm("RobustBestBase", float(others.min()), "{:.1f}")
            setm("RobustBestBaseName", str(others.idxmin()))
    setm("RobustSigmaMax", float(smax), "{:g}")

    def fn(ax):
        order = [OURS] + [m for m in g.model.unique() if m != OURS]
        for m in order:
            d = g[g.model == m].sort_values("sigma")
            if d.empty:
                continue
            c, mk, ls = MSTYLE.get(m, ("#444444", "o", "-"))
            ci = stats.t.ppf(0.975, np.maximum(d["count"] - 1, 1)) * d["std"].fillna(0) / np.sqrt(d["count"])
            ax.plot(d.sigma, d["mean"], color=c, marker=mk, ls=ls, lw=1.5 if m == OURS else 0.9,
                    ms=3.2 if m == OURS else 2.6, label=m, zorder=4 if m == OURS else 2)
            ax.fill_between(d.sigma, d["mean"] - ci, d["mean"] + ci, color=c, alpha=0.15, lw=0)
        ax.set_xlabel(r"Input noise $\sigma$ (normalised units)")
        ax.set_ylabel("MAE increase vs. clean (%)")
        ax.legend(fontsize=6, loc="upper left")
    return ("Robustness to input noise", fn)


def p_lipschitz():
    f = art("lipschitz.csv")
    if f is None:
        return None
    df = pd.read_csv(f)
    models = [OURS] + [m for m in df.model.unique() if m != OURS]
    models = [m for m in models if m in set(df.model)]
    med = df.groupby("model").ratio.median()
    p99 = df.groupby("model").ratio.quantile(0.99)
    if OURS in med:
        setm("LfMedOurs", float(med[OURS]))
        setm("LfPninetynineOurs", float(p99[OURS]))
        b = df[df.model == OURS].bound.dropna()
        if len(b):
            setm("LfBound", float(b.iloc[0]), "{:.3g}")
            setm("LfTightness", float(p99[OURS] / b.iloc[0]), "{:.3g}")

    def fn(ax):
        data = [df[df.model == m].ratio.values for m in models]
        bp = ax.boxplot(data, vert=True, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="k", lw=0.8), whiskerprops=dict(lw=0.6),
                        capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(MSTYLE.get(m, ("#999999",))[0])
            patch.set_alpha(0.8)
        b = df[df.model == OURS].bound.dropna()
        if len(b):
            ax.axhline(b.iloc[0], color=MSTYLE[OURS][0], ls="--", lw=0.9)
            ax.text(len(models) + 0.45, b.iloc[0], r"$L_F$ bound", ha="right", va="bottom", fontsize=6,
                    color=MSTYLE[OURS][0])
            ax.set_yscale("log")
            clean_log(ax, "y")
        ax.set_xticks(range(1, len(models) + 1), models, rotation=30, ha="right")
        ax.set_ylabel(r"$\|\Delta F\| / \|\Delta x\|$ (finite diff.)")
        ax.grid(axis="x", visible=False)
    return ("Empirical Lipschitz ratio", fn)


def fig_stability():
    compose("input_stability", [p_robust_curve(), p_lipschitz()], ncols=2, width=COL_W, row_h=1.8)


# ============================================================ calibration (C9)
def calib_city(c):
    d = load_npz(f"predictability_{KEY[c]}.npz")
    if d is None:
        return None
    kappa = float(d.get("kappa", 1.0))
    s, e = d["s_test"].astype(float), d["e_test"].astype(float)
    tgt = np.exp(-kappa * e)
    bins = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(s, bins) - 1, 0, 9)
    conf, acc, cnt = [], [], []
    for k in range(10):
        m = b == k
        if m.sum() >= 10:
            conf.append(s[m].mean())
            acc.append(tgt[m].mean())
            cnt.append(m.sum())
    conf, acc, cnt = map(np.asarray, (conf, acc, cnt))
    ece = float(np.sum(cnt / cnt.sum() * np.abs(acc - conf)))

    sc, ec = d["s_cal"].astype(float), d["e_cal"].astype(float)
    sig = lambda x: -np.log(np.clip(x, 1e-12, 1)) / kappa
    floor = 0.05 * np.median(sig(sc)) + 1e-8
    scale_c, scale_t = sig(sc) + floor, sig(s) + floor
    r = ec / scale_c
    n = r.size
    levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    cov, wn, wu = [], [], []
    for a in levels:
        qlev = min(1.0, math.ceil((n + 1) * a) / n)
        q = np.quantile(r, qlev, method="higher")
        q0 = np.quantile(ec, qlev, method="higher")
        cov.append(np.mean(e <= q * scale_t))
        wn.append(np.mean(q * scale_t))
        wu.append(q0)
    i90 = int(np.where(levels == 0.9)[0][0])
    rho = stats.spearmanr(s, e).statistic
    setm(f"ECE{TAG[c]}", ece, "{:.3f}")
    setm(f"CovNinety{TAG[c]}", 100 * cov[i90], "{:.1f}")
    setm(f"WidthRatio{TAG[c]}", wn[i90] / wu[i90])
    setm(f"SpearmanScore{TAG[c]}", rho)
    dec = pd.qcut(s, 10, labels=False, duplicates="drop")
    return dict(conf=conf, acc=acc, cnt=cnt, ece=ece, levels=levels, cov=np.asarray(cov), rho=rho,
                ece_val=ece, wr=wn[i90] / wu[i90], cov90=cov[i90],
                dec_s=np.array([s[dec == k].mean() for k in np.unique(dec)]),
                dec_e=np.array([e[dec == k].mean() for k in np.unique(dec)]),
                dec_lb=np.array([sig(np.mean(np.exp(-kappa * e[dec == k]))) for k in np.unique(dec)]))


def fig_calibration():
    res = {c: calib_city(c) for c in CITIES}
    res = {c: r for c, r in res.items() if r is not None}
    if not res:
        print("[skip] predictability_calibration: no inputs")
        return
    setm("ECEMean", float(np.mean([r["ece"] for r in res.values()])), "{:.3f}")
    setm("ECEMax", float(np.max([r["ece"] for r in res.values()])), "{:.3f}")
    setm("CovNinetyMean", float(100 * np.mean([r["cov90"] for r in res.values()])), "{:.1f}")
    setm("WidthRatioMean", float(np.mean([r["wr"] for r in res.values()])))
    setm("SpearmanMean", float(np.mean([r["rho"] for r in res.values()])))

    def rel(ax):
        ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls=":")
        for c, r in res.items():
            ax.plot(r["conf"], r["acc"], color=CSTYLE[c][0], marker=CSTYLE[c][1], ms=2.5,
                    label=f"{SHORT[c]} ECE={r['ece']:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(r"Predicted score $s_{\rm pred}$")
        ax.set_ylabel(r"Mean $s_{\rm target}$ in bin")
        ax.legend(fontsize=6, loc="upper left")

    def cov(ax):
        ax.plot([0.5, 0.95], [0.5, 0.95], color="k", lw=0.6, ls=":")
        for c, r in res.items():
            ax.plot(r["levels"], r["cov"], color=CSTYLE[c][0], marker=CSTYLE[c][1], ms=2.5,
                    label=f"{SHORT[c]} width×{r['wr']:.2f}")
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage (test)")
        ax.legend(fontsize=6, loc="upper left")

    def err(ax):
        for c, r in res.items():
            col = CSTYLE[c][0]
            ax.plot(r["dec_s"], r["dec_e"], color=col, marker=CSTYLE[c][1], ms=2.5,
                    label=rf"{SHORT[c]} $\rho_s$={r['rho']:.2f}")
            ax.plot(r["dec_s"], r["dec_lb"], color=col, ls=":", lw=0.7)
        ax.set_xlabel(r"$s_{\rm pred}$ (decile mean)")
        ax.set_ylabel(r"Mean error $\|\hat{\mathbf{Y}}-\mathbf{Y}\|$")
        ax.legend(fontsize=6, loc="upper right")
    compose("predictability_calibration",
            [("Reliability diagram", rel), ("Interval coverage", cov), ("Score ranks errors", err)],
            ncols=3, width=TEXT_W, row_h=2.0)


# ============================================================ alternatives (C7)
def alternatives_table():
    """Comment 7: replaces the interference operator; written as LaTeX rows (no figure)."""
    f = art("alternatives.csv")
    if f is None:
        hz = {c: ([15, 30, 60] if STEP_MIN[c] == 5 else [10, 30, 60]) for c in CITIES}
        ncol = sum(len(v) for v in hz.values())
        lines = [r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule",
                 "Variant & " + " & ".join(rf"\multicolumn{{{len(hz[c])}}}{{c}}{{{c}}}" for c in CITIES) + r" \\",
                 " ".join(rf"\cmidrule(lr){{{2 + sum(len(hz[d]) for d in CITIES[:i])}-"
                          rf"{1 + sum(len(hz[d]) for d in CITIES[:i + 1])}}}" for i in range(len(CITIES))),
                 " & " + " & ".join(f"{h} min" for c in CITIES for h in hz[c]) + r" \\", r"\midrule"]
        lines += [f"{ALT_NAME[v]} & " + " & ".join(["--"] * ncol) + r" \\" for v in ALT_ORDER]
        lines += [r"\bottomrule", r"\end{tabular}"]
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "alternatives_table.tex").write_text(
            "% PLACEHOLDER: no alternatives.csv yet; run the variants and export_artifacts.log_alternative.\n"
            + "\n".join(lines) + "\n")
        print("[ok]   alternatives_table.tex (placeholder: -- until alternatives.csv exists)")
        return
    df = pd.read_csv(f)
    df["variant"] = df.variant.str.lower()
    variants = [v for v in ALT_ORDER if v in set(df.variant)]
    full = df[df.variant == "full"].set_index(["dataset", "horizon", "seed"]).mae
    df = df.join(full.rename("mae_full"), on=["dataset", "horizon", "seed"])
    df["rel"] = 100 * (df.mae / df.mae_full - 1)
    for v, name in [("additive", "AltAdditive"), ("concat", "AltConcat"), ("fourier", "AltFourier"),
                    ("wavelet", "AltWavelet"), ("seasonal", "AltSeasonal")]:
        if v in variants:
            setm(name, float(df[(df.variant == v) & (df.horizon == 30)].rel.mean()), "{:.1f}")
    datasets = [c for c in CITIES if c in set(df.dataset)] or sorted(df.dataset.unique())
    hz = {c: sorted(int(h) for h in df[df.dataset == c].horizon.unique()) for c in datasets}
    best = df.groupby(["dataset", "horizon", "variant"]).mae.mean()
    ncol = sum(len(v) for v in hz.values())
    lines = [r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule",
             "Variant & " + " & ".join(rf"\multicolumn{{{len(hz[c])}}}{{c}}{{{c}}}" for c in datasets) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + sum(len(hz[d]) for d in datasets[:i])}-"
                      rf"{1 + sum(len(hz[d]) for d in datasets[:i + 1])}}}" for i, c in enumerate(datasets)),
             " & " + " & ".join(f"{h} min" for c in datasets for h in hz[c]) + r" \\", r"\midrule"]
    for v in variants:
        cells = []
        for c in datasets:
            for h in hz[c]:
                g = df[(df.variant == v) & (df.dataset == c) & (df.horizon == h)].mae
                if not len(g):
                    cells.append("--")
                    continue
                cell = f"{g.mean():.4f}$_{{\\pm {g.std(ddof=1):.2f}}}$"
                if np.isclose(g.mean(), best.loc[(c, h)].min()):
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
        lines.append(f"{ALT_NAME[v]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RESULTS / "alternatives_table.tex").write_text("\n".join(lines) + "\n")
    print("[ok]   alternatives_table.tex")


def efficiency_table():
    """Table VIII merged with Table IX (parameters, FLOPs): complete tabular, best (lowest) per column in bold."""
    cx = COMPLEXITY.set_index("model")
    extra = COMPLEXITY[~COMPLEXITY.model.isin(EFF.model)][["model", "family"]]
    df = pd.concat([EFF, extra], ignore_index=True)
    order = {"Classical": 0, "Reptile": 1, "Transfer": 2, "Prompt": 3, "Ours": 4}
    df = df.assign(_o=df.family.map(order)).sort_values("_o", kind="stable").drop(columns="_o")
    df["params"] = [cx.params_m.get(m, np.nan) for m in df.model]
    df["gflops"] = [cx.gflops.get(m, np.nan) for m in df.model]
    cols = [("t_ep", "{:.1f}"), ("mem_tr", "{:.1f}"), ("epochs", "{:.0f}"), ("inf_ms", "{:.1f}"),
            ("mem_inf", "{:.1f}"), ("ad_min", "{:.1f}"), ("ad_samples", "{:,.0f}"), ("ad_mem", "{:.1f}"),
            ("params", "{:.1f}"), ("gflops", "{:.1f}")]
    best = {c: df[c].min() for c, _ in cols}
    fam = {"Classical": "Classical", "Reptile": "Reptile", "Transfer": "Transfer", "Prompt": "Prompt", "Ours": "Ours"}
    lines = [r"\begin{tabular}{llrrrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{Training} & \multicolumn{2}{c}{Inference} & "
             r"\multicolumn{3}{c}{Adaptation} & \multicolumn{2}{c}{Size} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}\cmidrule(lr){11-12}",
             r"Model & Type & s/epoch & Mem (GB) & Epochs & ms/sample & Mem (GB) & Time (min) & Samples & "
             r"Mem (GB) & Params (M) & GFLOPs \\", r"\midrule"]
    prev = None
    for _, r in df.iterrows():
        if prev is not None and r.family != prev:
            lines.append(r"\midrule")
        prev = r.family
        cells = []
        for c, f in cols:
            v = r[c]
            if pd.isna(v):
                cells.append("--")
                continue
            t = f.format(v)
            cells.append(rf"\textbf{{{t}}}" if np.isclose(v, best[c]) else t)
        name = rf"\textbf{{{r.model}}}" if r.model == OURS else r.model
        lines.append(f"{name} & {fam[r.family]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RESULTS / "efficiency_table.tex").write_text("\n".join(lines) + "\n")
    print("[ok]   efficiency_table.tex")


# ============================================================ numbers
def report_table_numbers():
    print("\nTable-derived numbers (use these in the text):")
    for c in CITIES:
        o = np.asarray(MAE[c][OURS][0])
        for ref in ("TSKNET", "STGP"):
            b = np.asarray(MAE[c][ref][0])
            print(f"  {c:9s} gain vs {ref:7s}: " + ", ".join(f"{h}m {100 * (1 - x / y):.1f}%"
                                                          for h, x, y in zip(HORIZONS[c], o, b)))
        r = o / np.asarray(MAE[c]["TSKNET"][0])
        print(f"  {c:9s} CIWI/TSKNET MAE ratio: " + ", ".join(f"{x:.4f}" for x in r))
        rel = np.asarray(MAE[c][OURS][1]) / o * 100
        print(f"  {c:9s} CIWI relative std (%): " + ", ".join(f"{x:.1f}" for x in rel))


def write_macros():
    RESULTS.mkdir(parents=True, exist_ok=True)
    lines = ["% Auto-generated by make_figures.py; ?? marks values whose artifact is missing."]
    for k in sorted(set(EXPECTED) | set(MACROS)):
        lines.append("\\newcommand{\\%s}{%s}" % (k, MACROS.get(k, r"\textbf{??}")))
    (RESULTS / "figure_numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"[ok]   figure_numbers.tex ({len(MACROS)}/{len(set(EXPECTED) | set(MACROS))} filled)")


def main():
    set_style()
    print(f"ROOT={ROOT}\nRESULTS={RESULTS}\nARTIFACTS={ART}\n")
    for build in (fig_protocol, dataset_table, fig_data_characteristics, fig_statistics, fig_chaos,
                  fig_transfer, fig_wave, fig_theory, fig_stability, fig_calibration,
                  alternatives_table, efficiency_table, report_table_numbers):
        try:
            build()
        except Exception:
            print(f"[fail] {build.__name__}")
            traceback.print_exc()
    write_macros()
    if MISSING:
        print("\nMissing inputs (their panels were skipped, not simulated):")
        for m in sorted(MISSING):
            print("   ", m)


if __name__ == "__main__":
    main()
