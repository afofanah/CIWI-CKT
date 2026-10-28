#!/usr/bin/env python3
"""Representation-level ablation of the chaos proxy (Table VII groups), on real windows.

Groups of the 20-feature proxy Phi (Table VII):
  A: mean, variance, skewness, kurtosis
  B: autocorrelation lags 1-6, trend slope, L2 energy
  C: max/min consecutive difference, range, median, MAD, IQR, CV, Theil-Sen slope

Panels
  (a)-(b) t-SNE of raw windows and of the full proxy, coloured by traffic regime
          (last speed / sensor free-flow speed: >=0.85 free flow, 0.6-0.85 transition, <0.6 congested)
  (c)     full proxy coloured by city            (are cities mixed, i.e. is the representation shared?)
  (d)     full proxy coloured by 30-min difficulty (is the representation ordered by forecast difficulty?)
  (e)     ablation: R^2 for predicting 30-min difficulty within a city vs in an unseen city
          (leave-one-city-out), mean over the four target cities, bars = min-max over cities
  (f)     ablation: unseen-city R^2 per feature set at 10, 30 and 60 min (horizons on both grids)
Difficulty = log(1 + |z(t+h) - z(t)|), z = training-split z-score.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402
import theory_checks as tc  # noqa: E402

N_PER_CITY, T, SEED = 1000, 12, 0
HORIZONS_MIN = (10, 30, 60)
MOM = ["mean", "var", "skew", "kurt", *[f"acf{l}" for l in range(1, 7)], "trend", "cv"]
ORD = ["dmax", "dmin", "range", "median", "mad", "iqr", "theil_sen", "energy"]
GROUPS = {"A": ["mean", "var", "skew", "kurt"],
          "B": [*[f"acf{l}" for l in range(1, 7)], "trend", "energy"],
          "C": ["dmax", "dmin", "range", "median", "mad", "iqr", "cv", "theil_sen"]}
SETS = {"Raw window": None, "Full proxy": "ABC", "w/o A": "BC", "w/o B": "AC", "w/o C": "AB",
        "A only": "A", "B only": "B", "C only": "C"}
BAR_SETS = ["Raw window", "Full proxy", "A only", "B only", "C only"]
REGIMES = ["Free flow", "Transition", "Congested"]
REG_C = ["#56B4E9", "#E69F00", "#D55E00"]


def sample(city, rng):
    X = mf.load_series(city)
    tr, _ = mf.split_idx(len(X))
    Xtr = X[:tr]
    v = Xtr[Xtr > 0]
    mu, sd = v.mean(), v.std()
    ff = np.nanpercentile(np.where(Xtr > 0, Xtr, np.nan), 85, axis=0)
    ks = {h: h // mf.STEP_MIN[city] for h in HORIZONS_MIN}
    kmax = max(ks.values())
    W, D, reg = [], {h: [] for h in HORIZONS_MIN}, []
    while len(W) < N_PER_CITY:
        t0, nd = rng.integers(0, tr - T - kmax), rng.integers(0, X.shape[1])
        w = Xtr[t0:t0 + T + kmax, nd]
        if not np.all(w > 0) or not np.isfinite(ff[nd]):
            continue
        z = (w - mu) / sd
        W.append(z[:T])
        for h, k in ks.items():
            D[h].append(np.log1p(abs(z[T - 1 + k] - z[T - 1])))
        r = w[T - 1] / ff[nd]
        reg.append(0 if r >= 0.85 else (1 if r >= 0.6 else 2))
    return np.asarray(W), {h: np.asarray(d) for h, d in D.items()}, np.asarray(reg)


def build():
    rng = np.random.default_rng(SEED)
    Ws, Ds, regs, cities = [], {h: [] for h in HORIZONS_MIN}, [], []
    for c in mf.CITIES:
        W, D, r = sample(c, rng)
        Ws.append(W)
        regs.append(r)
        cities += [c] * len(W)
        for h in HORIZONS_MIN:
            Ds[h].append(D[h])
    W = np.concatenate(Ws)
    m, o = tc.phi(W)
    F = pd.DataFrame(np.c_[m, o], columns=MOM + ORD)
    return W, F, {h: np.concatenate(v) for h, v in Ds.items()}, np.concatenate(regs), np.asarray(cities)


def feats(W, F, key):
    if key is None:
        return W
    return F[[c for g in key for c in GROUPS[g]]].values


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def knn_fit_predict(Xtr, ytr, Xte, k=25):
    sc = StandardScaler().fit(Xtr)
    return KNeighborsRegressor(k).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def scores(X, y, city):
    """Per target city: within-city 5-fold R^2 and leave-one-city-out R^2."""
    X = np.nan_to_num(X)
    within, unseen = [], []
    for c in mf.CITIES:
        s = city == c
        pred = np.empty(s.sum())
        for tr_i, te_i in KFold(5, shuffle=True, random_state=0).split(X[s]):
            pred[te_i] = knn_fit_predict(X[s][tr_i], y[s][tr_i], X[s][te_i])
        within.append(r2(y[s], pred))
        unseen.append(r2(y[s], knn_fit_predict(X[~s], y[~s], X[s])))
    return np.asarray(within), np.asarray(unseen)


def tsne(X):
    Xs = StandardScaler().fit_transform(np.nan_to_num(X))
    try:
        return TSNE(2, perplexity=40, random_state=0, init="pca", max_iter=1000).fit_transform(Xs)
    except TypeError:
        return TSNE(2, perplexity=40, random_state=0, init="pca", n_iter=1000).fit_transform(Xs)


def frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("0.75")
        sp.set_linewidth(0.5)


def p_regime(Y, reg, legend):
    def fn(ax):
        for i, (lab, col) in enumerate(zip(REGIMES, REG_C)):
            s = reg == i
            ax.scatter(Y[s, 0], Y[s, 1], s=1.4, color=col, alpha=0.6, lw=0, rasterized=True, label=lab)
        frame(ax)
        if legend:
            ax.legend(markerscale=5, fontsize=mf.fs(6), loc="lower right", handletextpad=0.2, frameon=True,
                      facecolor="white", edgecolor="0.8", framealpha=0.9)
    return fn


def p_city(Y, city):
    def fn(ax):
        for c in mf.CITIES:
            s = city == c
            ax.scatter(Y[s, 0], Y[s, 1], s=1.4, color=mf.CSTYLE[c][0], alpha=0.55, lw=0, rasterized=True, label=c)
        frame(ax)
        ax.legend(markerscale=5, fontsize=mf.fs(6), loc="lower right", handletextpad=0.2, frameon=True,
                  facecolor="white", edgecolor="0.8", framealpha=0.9)
    return fn


def p_difficulty(Y, d):
    def fn(ax):
        q = np.argsort(np.argsort(d)) / len(d)
        sc = ax.scatter(Y[:, 0], Y[:, 1], s=1.4, c=q, cmap="viridis", alpha=0.7, lw=0, rasterized=True)
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label("Difficulty quantile")
        cb.ax.tick_params(labelsize=mf.fs(6), length=2)
        cb.outline.set_linewidth(0.4)
        frame(ax)
    return fn


def p_bars(res30):
    def fn(ax):
        x = np.arange(len(BAR_SETS))
        for i, (kind, col, lab) in enumerate((("within", "#999999", "Within city"),
                                              ("unseen", "#0072B2", "Unseen city"))):
            v = np.array([res30[s][kind] for s in BAR_SETS])
            ax.bar(x + (i - 0.5) * 0.38, v.mean(1), 0.38, color=col, label=lab,
                   yerr=[v.mean(1) - v.min(1), v.max(1) - v.mean(1)], error_kw=dict(lw=0.6, capsize=1.5))
        ax.set_xticks(x, ["Raw", "Full", "A", "B", "C"])
        ax.set_ylabel(r"$R^2$, 30-min difficulty")
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(axis="x", visible=False)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 0), hi * 1.45)
        ax.legend(fontsize=mf.fs(6), loc="upper left", ncol=2, columnspacing=0.6)
    return fn


def p_matrix(tab):
    def fn(ax):
        M = tab.values
        im = ax.imshow(M, cmap="Blues", vmin=min(0, M.min()), vmax=M.max() * 1.1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=mf.fs(6),
                        color="white" if M[i, j] > 0.7 * M.max() else "black")
        ax.set_xticks(range(M.shape[1]), [f"{h} min" for h in tab.columns])
        ax.set_yticks(range(M.shape[0]), tab.index)
        ax.tick_params(length=0)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.set_label(r"Unseen-city $R^2$")
        cb.ax.tick_params(labelsize=mf.fs(6), length=2)
        cb.outline.set_linewidth(0.4)
    return fn


def main():
    mf.set_style(1.3)
    W, F, D, reg, city = build()
    rows, res30 = [], {}
    for lab, key in SETS.items():
        X = feats(W, F, key)
        for h in HORIZONS_MIN:
            wi, un = scores(X, D[h], city)
            rows.append(dict(set=lab, horizon=h, within=wi.mean(), unseen=un.mean(), unseen_min=un.min(),
                             unseen_max=un.max(), **{f"unseen_{mf.SHORT[c]}": v for c, v in zip(mf.CITIES, un)}))
            if h == 30:
                res30[lab] = dict(within=wi, unseen=un)
        print(f"[ok]   {lab}")
    df = pd.DataFrame(rows)
    mf.RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(mf.RESULTS / "ablation_embedding_scores.csv", index=False)
    tab = df.pivot(index="set", columns="horizon", values="unseen").reindex(list(SETS))
    reg_share = pd.Series(reg).value_counts(normalize=True).sort_index()
    print("regime shares:", dict(zip(REGIMES, reg_share.round(3))))
    print(df[df.horizon == 30][["set", "within", "unseen", "unseen_min", "unseen_max"]].round(3).to_string())

    Y_raw, Y_full = tsne(W), tsne(feats(W, F, "ABC"))
    panels = [("Embedding by traffic regime\n(raw window)", p_regime(Y_raw, reg, True)),
              ("Embedding by traffic regime\n(full proxy)", p_regime(Y_full, reg, False)),
              ("Embedding by city\n(full proxy)", p_city(Y_full, city)),
              ("Embedding by 30-min difficulty\n(full proxy)", p_difficulty(Y_full, D[30])),
              ("Ablation: within vs. unseen city\n(30 min)", p_bars(res30)),
              ("Ablation: unseen-city $R^2$\n(by horizon)", p_matrix(tab))]
    layout_rows("ablation_embedding", panels, row_h=2.35)


def layout_rows(name, panels, row_h):
    """Two rows laid out independently, so colourbars in row 2 do not shrink row 1."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(mf.TEXT_W, 2 * row_h), layout="constrained")
    subs = fig.subfigures(2, 1, hspace=0.02)
    for r, sub in enumerate(subs):
        axes = sub.subplots(1, 3)
        for c, ax in enumerate(axes):
            i = 3 * r + c
            title, fn = panels[i]
            fn(ax)
            mf.panel(ax, i, title)
        sub.align_xlabels(axes)
    mf.save(fig, name)


if __name__ == "__main__":
    main()
