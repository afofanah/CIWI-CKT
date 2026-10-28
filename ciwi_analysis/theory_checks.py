#!/usr/bin/env python3
"""Numerical validation of Lemmas 1-5, Theorems 1-3 and Propositions 1-2 (Theoretical Analysis).

Every module is implemented exactly as defined in the theory section, with random
weights that satisfy the assumptions: spectral norm 1 (A5), LayerNorm gain 1 and
eps>0 (A4), symmetric-normalised adjacency (A6). Inputs are real windows and road
graphs from the four datasets. Each bound is compared with finite-difference
ratios (lower estimates of the true Lipschitz constant) or Monte-Carlo estimates.
A check passes when the measurement never exceeds its bound.

Two versions of each architectural bound are reported:
  worst case : LayerNorm constant 1/sqrt(eps), attention ball radius R_I (as stated)
  local      : LayerNorm constant 1/min(row std) and attention radius max ||I~||
               observed on the data (valid in a neighbourhood of the data)

Outputs in RESULTS/theory_checks/: CSV files, lemma_validation.pdf,
theorem_validation.pdf and theory_numbers.tex (LaTeX macros).
Run: python theory_checks.py   (needs data/<key>/dataset.npy and matrix.npy)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.csgraph import breadth_first_order

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402

OUT = mf.RESULTS / "theory_checks"
LG = 1.13            # GELU Lipschitz constant (max slope 1.1289)
EPS_LN = 1e-5        # LayerNorm epsilon (PyTorch default)
SIG_MIN, MU_MIN = 0.1, 0.1   # stabilisers of Phi (A3, stabilised form)
D, DC, NW, DM, NH, NP = 16, 20, 12, 64, 8, 3
DK = D // NH
GAMMA = 0.5
FD_EPS, N_DIR = 1e-5, 12
RNG = np.random.default_rng(2026)
MACROS: dict[str, str] = {}


def setm(k, v, fmt="{:.3g}"):
    txt = v if isinstance(v, str) else fmt.format(v)
    if not isinstance(v, str) and "e" in txt:
        a, b = txt.split("e")
        txt = f"${a}\\times 10^{{{int(b)}}}$"
    MACROS[k] = txt


# ------------------------------------------------------------------ primitives
def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def sig(x):
    return 1 / (1 + np.exp(-x))


def layernorm(x, eps=EPS_LN):
    s = np.sqrt(x.var(-1, keepdims=True) + eps)
    return (x - x.mean(-1, keepdims=True)) / s, s.min()


def sn(shape, rng=RNG):
    W = rng.standard_normal(shape)
    if W.ndim == 2:
        return W / np.linalg.norm(W, 2)
    return np.stack([sn(w.shape, rng) for w in W.reshape(-1, *shape[-2:])]).reshape(shape)


def opn(W):
    return float(np.linalg.norm(W, 2))


def softmax(S):
    S = S - S.max(-1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(-1, keepdims=True)


def fd_ratio(f, x, in_axis=0, out_axis=0, keep_axis=None, n_dir=N_DIR, eps=FD_EPS, rng=RNG):
    """Max over random unit directions of ||f(x+eps u)-f(x)||/eps per sample (and per keep_axis).
    Finite differences give a lower estimate of the local Lipschitz constant."""
    fx = f(x)
    best = None
    for _ in range(n_dir):
        u = np.moveaxis(rng.standard_normal(x.shape), in_axis, 0)
        u /= np.linalg.norm(u.reshape(u.shape[0], -1), axis=1).reshape(-1, *[1] * (u.ndim - 1))
        d = f(x + eps * np.moveaxis(u, 0, in_axis)) - fx
        if keep_axis is None:
            d = np.moveaxis(d, out_axis, 0)
            r = np.linalg.norm(d.reshape(d.shape[0], -1), axis=1) / eps
        else:
            d = np.moveaxis(d, (keep_axis, out_axis), (0, 1))
            r = np.linalg.norm(d.reshape(d.shape[0], d.shape[1], -1), axis=2) / eps
        best = r if best is None else np.maximum(best, r)
    return best


# ------------------------------------------------------------------ data
def windows(city, n, T, rng=RNG, split="train"):
    X = mf.load_series(city)
    raw = np.load(mf.ROOT / "data" / mf.KEY[city] / "dataset.npy", mmap_mode="r")
    tr, va = mf.split_idx(len(X))
    v = X[:tr][X[:tr] > 0]
    mu, sd = float(v.mean()), float(v.std())
    lo, hi = (0, tr) if split == "train" else (va, len(X))
    out, tod = [], []
    while len(out) < n:
        t0, nd = rng.integers(lo, hi - T), rng.integers(0, X.shape[1])
        w = X[t0:t0 + T, nd]
        if np.all(w > 0):
            out.append((w - mu) / sd)
            tod.append(np.asarray(raw[t0:t0 + T, nd, 1]))
    return np.asarray(out, float), np.asarray(tod, float)


def inputs(city, n, T):
    z, tod = windows(city, n, T)
    return np.stack([z, tod], -1)                      # (n, T, 2)


# ============================================================ Lemma 1: chaos proxy
def phi(X, s=SIG_MIN, mu=MU_MIN, L=6):
    n, T = X.shape
    t = np.arange(1, T + 1, dtype=float)
    m = X.mean(1)
    A = X - m[:, None]
    m2 = (A ** 2).mean(1)
    sd = np.sqrt(m2 + s * s)
    mom = [m, m2, (A ** 3).mean(1) / sd ** 3, (A ** 4).mean(1) / sd ** 4]
    mom += [(A[:, l:] * A[:, :-l]).sum(1) / T / sd ** 2 for l in range(1, L + 1)]
    tc = t - t.mean()
    mom += [A @ tc / (tc @ tc), sd / np.sqrt(m * m + mu * mu)]
    dX = np.diff(X, axis=1)
    q25, q50, q75 = np.percentile(X, [25, 50, 75], axis=1)
    i, j = np.triu_indices(T, 1)
    ordg = [dX.max(1), dX.min(1), X.max(1) - X.min(1), q50, np.median(np.abs(X - q50[:, None]), 1),
            q75 - q25, np.median((X[:, j] - X[:, i]) / (j - i), 1), np.linalg.norm(X, axis=1)]
    return np.column_stack(mom), np.column_stack(ordg)


def phi_bounds(R, T, s=SIG_MIN, mu=MU_MIN, L=6):
    """Constants from the proof of Lemma 1 for the stabilised proxy (||x||_inf <= R)."""
    F2 = 4 * R * R + s * s                               # bound on sd^2
    F = math.sqrt(F2)
    c = [1.0, 4 * R,
         12 * R ** 2 / s ** 3 + 8 * R ** 3 * 3 * F2 * (2 * R / s) / s ** 6,
         32 * R ** 3 / s ** 4 + 16 * R ** 4 * 2 * F2 * 4 * R / s ** 8]
    c += [4 * R / s ** 2 + 4 * R * R * 4 * R / s ** 4] * L
    c += [(2 * R / s) / mu + F / mu ** 2]
    tc = np.arange(1, T + 1) - (T + 1) / 2
    l_mom = math.sqrt(sum(x * x for x in c) / T + 1 / float(tc @ tc))
    return l_mom, math.sqrt(26)


def phi_avg(X):
    """Phi applied to each channel of X (B,T,D) and averaged over channels -> (B,20)."""
    B, T, Dd = X.shape
    m, o = phi(np.moveaxis(X, 2, 1).reshape(-1, T))
    return np.concatenate([m, o], 1).reshape(B, Dd, -1).mean(1)


def check_lemma1():
    rows = []
    for city in mf.CITIES:
        for T in (6, 12, 24, 48):
            z, _ = windows(city, 400, T)
            R = 1.01 * np.abs(z).max()
            Lm, Lo = phi_bounds(R, T)
            rm = fd_ratio(lambda x: phi(x)[0], z)
            ro = fd_ratio(lambda x: phi(x)[1], z)
            rows.append(dict(city=city, T=T, R=R, mom_fd=rm.max(), mom_bound=Lm, ord_fd=ro.max(), ord_bound=Lo))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lemma1.csv", index=False)
    setm("LoneOrdTight", (df.ord_fd / df.ord_bound).max())
    setm("LoneMomTight", (df.mom_fd / df.mom_bound).max())
    setm("LonePass", "all" if ((df.mom_fd <= df.mom_bound) & (df.ord_fd <= df.ord_bound)).all() else "NOT all")
    return df


# ============================================================ Lemma 2: wave generator
class Model:
    def __init__(self, Nw=NW, rng=RNG, lam_k=1.0, T=12):
        self.Nw = Nw
        self.Wp = sn((2, D), rng)
        self.WE1, self.WE2 = sn((20, DC), rng), sn((DC, DC), rng)
        self.WA1 = sn((Nw, D + DC, D), rng)
        self.WA2 = sn((Nw, D, D), rng)
        self.WK = sn((Nw, D + DC, D), rng)
        self.bA1, self.bA2 = 0.1 * rng.standard_normal((2, Nw, D))
        self.k0 = rng.uniform(0.5, 1.5, (Nw, D)) * 2 * np.pi / 12
        self.ph = rng.uniform(0, 2 * np.pi, (Nw, D))
        self.lam = lam_k
        self.Wctx = sn((D, DM), rng)
        self.Wb1, self.Wb2 = sn((Nw, DM, D), rng), sn((Nw, D, D), rng)
        self.WQ, self.WKa = sn((NP, NH, D, DK), rng), sn((NP, NH, D, DK), rng)
        self.WV = sn((NP, NH, D, DK), rng)
        self.WO = sn((NP, D, D), rng)
        self.W1, self.W2 = sn((NP * D, D), rng), sn((D, D), rng)
        self.wX, self.wH, self.wM = sn((D, 1), rng), sn((D, 1), rng), sn((DM, 1), rng)
        self.ln_min = {}

    # chaos encoder
    def encode(self, Xt):
        h, s = layernorm(phi_avg(Xt) @ self.WE1)
        self.ln_min["E"] = min(self.ln_min.get("E", np.inf), s)
        return gelu(h) @ self.WE2

    def waves_from(self, Xt, C, t):
        B, T, _ = Xt.shape
        Xe = np.concatenate([Xt, np.broadcast_to(C[:, None, :], (B, T, DC))], -1)
        h, s = layernorm(np.einsum("btf,nfd->nbtd", Xe, self.WA1) + self.bA1[:, None, None])
        self.ln_min["A"] = min(self.ln_min.get("A", np.inf), s)
        A = sig(np.einsum("nbtd,nde->nbte", gelu(h), self.WA2) + self.bA2[:, None, None])
        k = self.lam * self.k0[:, None, :] * np.tanh(np.einsum("bf,nfd->nbd", Xe.mean(1), self.WK))
        arg = t[None, None, :, None] * k[:, :, None, :] + self.ph[:, None, None, :]
        psi = np.where((np.arange(self.Nw) % 2 == 0)[:, None, None, None], np.sin(arg), np.cos(arg))
        return A * psi, A

    def waves(self, Xt, t):
        return self.waves_from(Xt, self.encode(Xt), t)[0]

    # interference
    def context(self, Ws, Wq):
        u = (Ws.mean(1)[:, None, :] + Wq.mean(2)).sum(0) / (2 * self.Nw)
        return gelu(u @ self.Wctx)

    def interfere(self, Ws, Wq, gamma=GAMMA):
        M = self.context(Ws, Wq)
        beta = sig(np.einsum("nbd,nde->nbe", gelu(np.einsum("bm,nmd->nbd", M, self.Wb1)), self.Wb2))
        G = Ws[:, None] + Wq + gamma * Ws[:, None] * Wq
        return (beta[:, :, None, :] * G).sum(0), M

    # attention
    def head_out(self, X):
        Q = np.einsum("btd,phdk->phbtk", X, self.WQ)
        K = np.einsum("btd,phdk->phbtk", X, self.WKa)
        V = np.einsum("btd,phdk->phbtk", X, self.WV)
        P = softmax(np.einsum("phbtk,phbsk->phbts", Q, K) / math.sqrt(DK))
        return np.einsum("phbts,phbsk->phbtk", P, V)                      # (NP,NH,B,T,DK)

    def attention(self, X):
        Hh = self.head_out(X)
        cat = np.concatenate(list(np.moveaxis(Hh, 1, 0)), -1)              # (NP,B,T,D)
        Z = np.concatenate(list(np.einsum("pbtd,pde->pbte", cat, self.WO)), -1)
        h, s = layernorm(gelu(Z @ self.W1))
        self.ln_min["F"] = min(self.ln_min.get("F", np.inf), s)
        return X + gelu(h) @ self.W2

    # full forward map (per node, sensors folded into batch)
    def head(self, Xt, H, M):
        return Xt @ self.wX + H @ self.wH + (M @ self.wM)[:, None, :]

    # ----- bounds
    def nu(self, key, local):
        return 1 / (0.9 * self.ln_min[key]) if local else 1 / math.sqrt(EPS_LN)

    def L_A(self, n, local):
        return 0.25 * LG * self.nu("A", local) * opn(self.WA2[n]) * opn(self.WA1[n])

    def L_k(self, n):
        return self.lam * np.abs(self.k0[n]).max() * opn(self.WK[n])

    def L_E(self, local):
        return LG * self.nu("E", local) * opn(self.WE2) * opn(self.WE1)

    def L_M(self):
        return LG * opn(self.Wctx) / (2 * math.sqrt(self.Nw * T_GLOBAL[0]))

    def L_beta(self, n):
        return 0.25 * LG * opn(self.Wb2[n]) * opn(self.Wb1[n])

    def L_I(self, gamma=GAMMA):
        T = T_GLOBAL[0]
        return (1 + gamma) * math.sqrt(self.Nw) + (2 + gamma) * math.sqrt(T) * self.L_M() * \
            sum(self.L_beta(n) for n in range(self.Nw))

    def L_head(self, p, h, R):
        T = T_GLOBAL[0]
        q = opn(self.WQ[p, h]) * opn(self.WKa[p, h]) / math.sqrt(DK)
        return opn(self.WV[p, h]) * (math.sqrt(T) + 2 * R * R * q)

    def L_att(self, R, local):
        Lh = max(self.L_head(p, h, R) for p in range(NP) for h in range(NH))
        wO = max(opn(self.WO[p]) for p in range(NP))
        return 1 + LG ** 2 * self.nu("F", local) * opn(self.W2) * opn(self.W1) * math.sqrt(NP * NH) * wO * Lh


T_GLOBAL = [12]


def tau(t):
    return float(np.linalg.norm(t) / math.sqrt(len(t)))


def proj(Xin, m):
    return Xin @ m.Wp


def check_lemma2():
    rows, scal = [], []
    for city in mf.CITIES:
        T = 12
        T_GLOBAL[0] = T
        t = np.arange(1, T + 1, dtype=float)
        m = Model(T=T)
        Xt = proj(inputs(city, 48, T), m)
        C = m.encode(Xt)
        W, _ = m.waves_from(Xt, C, t)
        Rch = 1.01 * np.abs(Xt).max()
        LPhi = sum(phi_bounds(Rch, T)) / math.sqrt(D)
        g_fd = fd_ratio(lambda c: m.waves_from(Xt, c, t)[0], C, out_axis=1, keep_axis=0).T
        x_fd = fd_ratio(lambda x: m.waves_from(x, C, t)[0], Xt, out_axis=1, keep_axis=0).T
        f_fd = fd_ratio(lambda x: m.waves(x, t), Xt, out_axis=1, keep_axis=0).T
        for local in (False, True):
            Gam = np.array([math.sqrt(T) * (m.L_A(n, local) + tau(t) * m.L_k(n)) for n in range(NW)])
            LGX = np.array([m.L_A(n, local) + tau(t) * m.L_k(n) for n in range(NW)])
            LGf = LGX + Gam * m.L_E(local) * LPhi
            rows.append(dict(city=city, version="local" if local else "worst case", max_abs_W=np.abs(W).max(),
                             gamma_ratio=(g_fd / Gam).max(), gx_ratio=(x_fd / LGX).max(),
                             g_ratio=(f_fd / LGf).max()))
    # T-scaling of Gamma (Remark), raw vs normalised indices, METR-LA, local constants
    for T in (6, 12, 24, 48):
        T_GLOBAL[0] = T
        m = Model(T=T)
        Xt = proj(inputs("METR-LA", 32, T), m)
        C = m.encode(Xt)
        for idx in ("raw", "normalised"):
            t = np.arange(1, T + 1, dtype=float) / (T if idx == "normalised" else 1)
            m.waves_from(Xt, C, t)
            fd = fd_ratio(lambda c: m.waves_from(Xt, c, t)[0], C, out_axis=1, keep_axis=0).max()
            bound = max(math.sqrt(T) * (m.L_A(n, True) + tau(t) * m.L_k(n)) for n in range(NW))
            scal.append(dict(T=T, index=idx, fd=fd, bound=bound))
    T_GLOBAL[0] = 12
    df, sc = pd.DataFrame(rows), pd.DataFrame(scal)
    df.to_csv(OUT / "lemma2.csv", index=False)
    sc.to_csv(OUT / "lemma2_scaling.csv", index=False)
    setm("LtwoMaxW", df.max_abs_W.max(), "{:.3f}")
    for idx, k in (("raw", "LtwoSlopeRaw"), ("normalised", "LtwoSlopeNorm")):
        g = sc[sc["index"] == idx]
        setm(k, np.polyfit(np.log(g["T"]), np.log(g.bound), 1)[0], "{:.2f}")
        setm(k + "Fd", np.polyfit(np.log(g["T"]), np.log(g.fd), 1)[0], "{:.2f}")
    return df, sc


# ============================================================ Lemmas 3-4: interference and attention
def check_lemma34():
    r3, r4 = [], []
    T = T_GLOBAL[0] = 12
    t = np.arange(1, T + 1, dtype=float)
    for city in mf.CITIES:
        m = Model(T=T)
        Ws = m.waves(proj(inputs(city, 64, T), m), t).mean(1)          # support waves, batch-averaged
        Wq = m.waves(proj(inputs(city, 48, T), m), t)                  # (Nw,B,T,D)
        for g in (0.0, 0.5, 1.0):
            fdI = fd_ratio(lambda w: m.interfere(Ws, w, g)[0], Wq, in_axis=1)
            fdM = fd_ratio(lambda w: m.context(Ws, w), Wq, in_axis=1)
            r3.append(dict(city=city, gamma=g, I_ratio=(fdI / m.L_I(g)).max(), M_ratio=(fdM / m.L_M()).max()))
        I, _ = m.interfere(Ws, Wq)
        RI = NW * (2 + GAMMA) * math.sqrt(T * D)
        Rdata = 1.05 * np.linalg.norm(I.reshape(len(I), -1), axis=1).max()
        hd = fd_ratio(lambda x: m.head_out(x)[0, 0], I)
        md = fd_ratio(m.attention, I)
        m.attention(I)
        r4.append(dict(city=city, ball_ratio=Rdata / 1.05 / RI,
                       head_ratio_ball=(hd / m.L_head(0, 0, RI)).max(),
                       head_ratio_data=(hd / m.L_head(0, 0, Rdata)).max(),
                       module_ratio_ball=(md / m.L_att(RI, False)).max(),
                       module_ratio_data=(md / m.L_att(Rdata, True)).max()))
    d3, d4 = pd.DataFrame(r3), pd.DataFrame(r4)
    d3.to_csv(OUT / "lemma3.csv", index=False)
    d4.to_csv(OUT / "lemma4.csv", index=False)
    setm("LthreeTight", d3.I_ratio.max())
    setm("LfourBall", d4.ball_ratio.max(), "{:.3f}")
    setm("LfourHeadData", d4.head_ratio_data.max())
    return d3, d4


# ============================================================ Lemma 5: superposition
def check_lemma5():
    rows = []
    T = T_GLOBAL[0] = 12
    t = np.arange(1, T + 1, dtype=float)
    for city in mf.CITIES:
        Xin = inputs(city, 400, T)
        for nw in (2, 4, 8, 12, 16, 20):
            m = Model(Nw=nw, T=T)
            W, A = m.waves_from(proj(Xin, m), m.encode(proj(Xin, m)), t)
            Wf = W.reshape(nw, W.shape[1], -1)
            Wc = Wf - Wf.mean(1, keepdims=True)
            cov = np.einsum("ibk,jbk->ij", Wc, Wc) / Wf.shape[1]
            v = np.diag(cov)
            rho = cov / np.sqrt(np.outer(v, v))
            rp = np.clip(rho[~np.eye(nw, dtype=bool)], 0, None).mean() if nw > 1 else 0.0
            var_emp = float((Wc.mean(0) ** 2).sum(1).mean())
            Af = A.reshape(nw, A.shape[1], -1)
            a2var = float(((Af - Af.mean(1, keepdims=True)) ** 2).sum(2).mean(1).max())
            rows.append(dict(city=city, N_w=nw, var_emp=var_emp, rho_plus=rp,
                             bound_vmax=v.max() / nw * (1 + (nw - 1) * rp),
                             bound_avar=a2var / nw * (1 + (nw - 1) * rp)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lemma5.csv", index=False)
    setm("LfiveTight", (df.var_emp / df.bound_vmax).max(), "{:.3f}")
    setm("LfiveOrigViol", (df.var_emp / df.bound_avar).max(), "{:.1f}")
    setm("LfiveOrigViolFrac", 100 * (df.var_emp > df.bound_avar).mean(), "{:.0f}")
    return df


# ============================================================ Theorem 1: input stability
def subgraph(city, n=24):
    A = np.load(mf.ROOT / "data" / mf.KEY[city] / "matrix.npy")
    A = ((A > 0) | (A.T > 0)).astype(float)
    np.fill_diagonal(A, 0)
    start = int(np.argmax(A.sum(1)))
    order = breadth_first_order(A, start, directed=False, return_predecessors=False)[:n]
    S = A[np.ix_(order, order)] + np.eye(len(order))
    d = S.sum(1)
    return S / np.sqrt(np.outer(d, d)), order


def full_forward(m, An, theta, Ws, X, t):
    """X: (B, Nn, T, 2) -> Y: (B, Nn, T, 1)."""
    Xd = sum(th * np.einsum("ij,bjtc->bitc", np.linalg.matrix_power(An, k), X) for k, th in enumerate(theta))
    B, Nn = X.shape[:2]
    Xt = (Xd @ m.Wp).reshape(B * Nn, len(t), D)
    Wq = m.waves(Xt, t)
    I, M = m.interfere(Ws, Wq)
    H = m.attention(I)
    return m.head(Xt, H, M).reshape(B, Nn, len(t), 1)


def check_prop1():
    rows, stages = [], []
    T = T_GLOBAL[0] = 12
    t = np.arange(1, T + 1, dtype=float)
    theta = np.array([0.5, 0.3, 0.2])
    for city in mf.CITIES:
        m = Model(T=T)
        An, order = subgraph(city)
        X = mf.load_series(city)
        raw = np.load(mf.ROOT / "data" / mf.KEY[city] / "dataset.npy", mmap_mode="r")
        tr, _ = mf.split_idx(len(X))
        v = X[:tr][X[:tr] > 0]
        mu, sd = v.mean(), v.std()
        t0s = RNG.integers(0, tr - T, 6)
        Xb = np.stack([np.stack([(X[a:a + T, order] - mu) / sd, raw[a:a + T, order, 1]], -1).transpose(1, 0, 2)
                       for a in t0s])
        Ws = m.waves(proj(inputs(city, 64, T), m), t).mean(1)
        f = lambda x: full_forward(m, An, theta, Ws, x, t)  # noqa: E731
        fd = fd_ratio(f, Xb, n_dir=6)
        Xd = sum(th * np.einsum("ij,bjtc->bitc", np.linalg.matrix_power(An, k), Xb) for k, th in enumerate(theta))
        Xt = (Xd @ m.Wp).reshape(-1, T, D)
        Wq = m.waves(Xt, t)
        I, _ = m.interfere(Ws, Wq)
        m.attention(I)
        RI = NW * (2 + GAMMA) * math.sqrt(T * D)
        Rdata = 1.05 * np.linalg.norm(I.reshape(len(I), -1), axis=1).max()
        Rch = 1.05 * np.abs(Xt).max()
        LPhi = sum(phi_bounds(Rch, T)) / math.sqrt(D)
        L_pre = opn(m.Wp) * np.abs(theta).sum()
        for local in (False, True):
            LG_ = math.sqrt(NW) * max(m.L_A(n, local) + tau(t) * m.L_k(n) +
                                      math.sqrt(T) * (m.L_A(n, local) + tau(t) * m.L_k(n)) * m.L_E(local) * LPhi
                                      for n in range(NW))
            LF = L_pre * (opn(m.wX) + (opn(m.wH) * m.L_att(Rdata if local else RI, local) * m.L_I()
                                       + math.sqrt(T) * opn(m.wM) * m.L_M()) * LG_)
            rows.append(dict(city=city, version="local" if local else "worst case", fd_max=fd.max(),
                             fd_median=np.median(fd), L_F=LF, ratio=fd.max() / LF))
        Ct = m.encode(Xt)
        LGX = max(m.L_A(n, True) + tau(t) * m.L_k(n) for n in range(NW))
        Gam = max(math.sqrt(T) * (m.L_A(n, True) + tau(t) * m.L_k(n)) for n in range(NW))
        stages.append(dict(city=city, stage="pre", fd=fd_ratio(
            lambda x: sum(th * np.einsum("ij,bjtc->bitc", np.linalg.matrix_power(An, k), x)
                          for k, th in enumerate(theta)) @ m.Wp, Xb, n_dir=4).max(), bound=L_pre))
        stages.append(dict(city=city, stage="chaos path", fd=fd_ratio(m.encode, Xt, n_dir=4).max(),
                           bound=m.L_E(True) * LPhi))
        stages.append(dict(city=city, stage="waves | C", fd=fd_ratio(
            lambda x: m.waves_from(x, Ct, t)[0], Xt, out_axis=1, n_dir=4).max(), bound=math.sqrt(NW) * LGX))
        stages.append(dict(city=city, stage="C -> waves", fd=fd_ratio(
            lambda c: m.waves_from(Xt, c, t)[0], Ct, out_axis=1, n_dir=4).max(), bound=math.sqrt(NW) * Gam))
        stages.append(dict(city=city, stage="interference", fd=fd_ratio(
            lambda w: m.interfere(Ws, w)[0], Wq, in_axis=1, n_dir=4).max(), bound=m.L_I()))
        stages.append(dict(city=city, stage="attention", fd=fd_ratio(m.attention, I, n_dir=4).max(),
                           bound=m.L_att(Rdata, True)))
        stages.append(dict(city=city, stage="full map", fd=fd.max(), bound=rows[-1]["L_F"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "prop1.csv", index=False)
    st = pd.DataFrame(stages)
    st.to_csv(OUT / "prop1_stages.csv", index=False)
    setm("PoneChaosLoose", float((st[st.stage == "chaos path"].bound / st[st.stage == "chaos path"].fd).min()), "{:.1e}")
    setm("PoneOtherLooseMax", float((st[~st.stage.isin(["chaos path", "full map", "waves | C", "C -> waves"])].bound /
                                     st[~st.stage.isin(["chaos path", "full map", "waves | C", "C -> waves"])].fd).max()), "{:.0f}")
    setm("PoneFd", df.fd_max.max(), "{:.2f}")
    setm("PoneTightWorst", df[df.version == "worst case"].ratio.max())
    setm("PoneTightLocal", df[df.version == "local"].ratio.max())
    setm("PonePass", "all" if (df.ratio <= 1).all() else "NOT all")
    return df


# ============================================================ Theorem 2: calibration
def check_prop2():
    """(i)-(ii) semi-synthetic with known s*; (iii) and stop-gradient (A7) on real AR(12) errors."""
    kappa = 1.0
    rows, a7, p3 = [], [], []
    for city in mf.CITIES:
        z, _ = windows(city, 60000, 13)
        x, y = z[:, :12], z[:, 12]
        vol = np.c_[np.ones(len(x)), x[:, -6:].std(1), np.abs(np.diff(x[:, -3:], axis=1)).mean(1), x[:, -1]]
        Z = np.c_[np.ones(len(x)), (vol[:, 1:] - vol[:, 1:].mean(0)) / vol[:, 1:].std(0)]
        # (i)-(ii): e | z ~ |N(0, sigma(z)^2)|, sigma driven by real volatility features
        sigz = 0.15 + 1.5 * vol[:, 1] + 0.5 * vol[:, 2]
        e = np.abs(RNG.standard_normal(len(x)) * sigz)
        st = np.exp(-kappa * e)
        sstar = 2 * np.exp(kappa ** 2 * sigz ** 2 / 2) * stats.norm.cdf(-kappa * sigz)
        es2 = 2 * np.exp(2 * kappa ** 2 * sigz ** 2) * stats.norm.cdf(-2 * kappa * sigz)
        varz = es2 - sstar ** 2
        w = np.zeros(Z.shape[1])
        for it in range(401):
            h = sig(Z @ w)
            if it in (0, 5, 10, 20, 40, 80, 160, 400):
                R_h = np.mean((h - st) ** 2)
                decomp = np.mean((h - sstar) ** 2) + np.mean(varz)
                excess = np.mean((h - sstar) ** 2)
                bins = np.clip((h * 20).astype(int), 0, 19)
                ce = sum((bins == b).mean() * abs(st[bins == b].mean() - h[bins == b].mean())
                         for b in range(20) if (bins == b).sum() > 50)
                rows.append(dict(city=city, iter=it, risk=R_h, decomposition=decomp, ce=ce,
                                 bound=math.sqrt(excess)))
            g = 2 * (h - st) * h * (1 - h)
            w -= 2.0 * Z.T @ g / len(Z)
        # (iii) and A7 on real errors of a jointly trained linear AR(12) predictor
        n_tr = len(x) // 2
        for detach in (True, False):
            for alpha in (0.0, 0.5, 1.0, 2.0, 4.0):
                rng = np.random.default_rng(1)
                wp = np.linalg.lstsq(x[:n_tr], y[:n_tr], rcond=None)[0] + 0.01 * rng.standard_normal(12)
                wh = np.zeros(Z.shape[1])
                for _ in range(300):
                    yh = x[:n_tr] @ wp
                    r = yh - y[:n_tr]
                    s_t = np.exp(-kappa * np.abs(r))
                    h = sig(Z[:n_tr] @ wh)
                    g_pred = 2 * r[:, None] * x[:n_tr]
                    if not detach:
                        g_pred = g_pred + alpha * 2 * (h - s_t)[:, None] * (kappa * s_t * np.sign(r))[:, None] * x[:n_tr]
                    wp -= 0.02 * g_pred.mean(0)
                    wh -= 1.0 * (alpha * 2 * (h - s_t) * h * (1 - h)) @ Z[:n_tr] / n_tr
                r_te = x[n_tr:] @ wp - y[n_tr:]
                a7.append(dict(city=city, detach=detach, alpha=alpha, mae=np.abs(r_te).mean()))
                if detach and alpha == 1.0:
                    e_te, h_te = np.abs(r_te), sig(Z[n_tr:] @ wh)
                    st_te = np.exp(-kappa * e_te)
                    bins = np.clip((h_te * 10).astype(int), 0, 9)
                    for b in range(10):
                        sel = bins == b
                        if sel.sum() < 200:
                            continue
                        s_b = st_te[sel].mean()
                        for epsv in (0.1, 0.25, 0.5):
                            lb = (s_b - math.exp(-kappa * epsv)) / (1 - math.exp(-kappa * epsv))
                            p3.append(dict(city=city, bin=b, eps=epsv, prob=(e_te[sel] <= epsv).mean(), lb=lb,
                                           mean_err=e_te[sel].mean(), lb_err=-math.log(s_b) / kappa))
    d2, da7, d3 = pd.DataFrame(rows), pd.DataFrame(a7), pd.DataFrame(p3)
    d2.to_csv(OUT / "prop2_ce.csv", index=False)
    da7.to_csv(OUT / "prop2_stopgrad.csv", index=False)
    d3.to_csv(OUT / "prop2_iii.csv", index=False)
    setm("PtwoDecompErr", (np.abs(d2.risk - d2.decomposition) / d2.risk).max() * 100, "{:.2f}")
    setm("PtwoCEPass", "all" if (d2.ce <= d2.bound + 1e-3).all() else "NOT all")
    setm("PtwoiiiPass", f"{100 * ((d3.prob >= d3.lb - 1e-9) & (d3.mean_err >= d3.lb_err - 1e-9)).mean():.0f}")
    base = da7[da7.alpha == 0].groupby("city").mae.mean()
    worst = da7[(~da7.detach) & (da7.alpha == 4.0)].groupby("city").mae.mean()
    det = da7[(da7.detach) & (da7.alpha == 4.0)].groupby("city").mae.mean()
    setm("PtwoNoDetach", 100 * float((worst / base - 1).mean()), "{:.1f}")
    setm("PtwoDetach", 100 * float((det / base - 1).mean()), "{:.2f}")
    return d2, da7, d3


# ============================================================ Theorem 3: correlated episodes
def check_prop3(n_tau=20, reps=600, b=3.0, horizon_min=30, n_q=12):
    rows, cov = [], []
    for city in mf.CITIES:
        X = mf.load_series(city)
        _, va = mf.split_idx(len(X))
        tr, _ = mf.split_idx(len(X))
        v = X[:tr][X[:tr] > 0]
        Z = (X[va:] - v.mean()) / v.std()
        k = horizon_min // mf.STEP_MIN[city]
        spd = 1440 // mf.STEP_MIN[city]
        nd = (len(Z) - k) // spd

        def episode(day, rng):
            t0 = day * spd + rng.integers(0, spd - k, n_q)
            nds = rng.integers(0, Z.shape[1], n_q)
            return min(b, float(np.abs(Z[t0 + k, nds] - Z[t0, nds]).mean()))

        rng = np.random.default_rng(3)
        R_true = np.mean([episode(rng.integers(0, nd), rng) for _ in range(20000)])
        for n_days in (n_tau, 4, 1):
            L = np.empty((reps, n_tau))
            for r in range(reps):
                days = rng.integers(0, nd, n_days)
                L[r] = [episode(days[i % n_days], rng) for i in range(n_tau)]
            C = np.corrcoef(L.T)
            rp = float(np.clip(C[~np.eye(n_tau, dtype=bool)], 0, None).mean())
            n_eff = n_tau / (1 + (n_tau - 1) * rp)
            vmax = float(L.var(0).max())
            Rh = L.mean(1)
            rows.append(dict(city=city, days=n_days, rho_plus=rp, n_eff=n_eff, var_emp=Rh.var(),
                             bound=vmax / n_eff, naive=vmax / n_tau))
            for d in (0.05, 0.1, 0.2, 0.3, 0.5):
                cov.append(dict(city=city, days=n_days, delta=d,
                                cov_neff=np.mean(np.abs(Rh - R_true) <= math.sqrt(vmax / (n_eff * d))),
                                cov_naive=np.mean(np.abs(Rh - R_true) <= math.sqrt(vmax / (n_tau * d)))))
    d, c = pd.DataFrame(rows), pd.DataFrame(cov)
    d.to_csv(OUT / "prop3.csv", index=False)
    c.to_csv(OUT / "prop3_coverage.csv", index=False)
    c["ok"] = c.cov_neff >= 1 - c.delta - 0.02
    setm("PthreeVarPass", "all" if (d.var_emp <= d.bound * 1.05).all() else "NOT all")
    setm("PthreeCovMin", float((c.cov_neff - (1 - c.delta)).min()), "{:.3f}")
    one = c[c.days == 1]
    setm("PthreeNaiveCov", 100 * float(one[one.delta == 0.05].cov_naive.mean()), "{:.0f}")
    setm("PthreeNeffCov", 100 * float(one[one.delta == 0.05].cov_neff.mean()), "{:.0f}")
    setm("PthreeRhoOne", float(d[d.days == 1].rho_plus.max()), "{:.2f}")
    setm("PthreeNaiveViol", float((d.var_emp / d.naive).max()), "{:.1f}")
    setm("PthreeNeffTight", float((d.var_emp / d.bound).max()), "{:.2f}")
    return d, c


# ============================================================ Proposition 1: prototype diversity
def check_prop4(Dm=DM, iters=300, lr=0.05):
    rows = []
    for C in (2, 3, 4, 8, 16):
        Q = np.linalg.qr(RNG.standard_normal((Dm, C)))[0].T                 # orthonormal start
        for init, U in (("orthogonal", Q.copy()), ("random", RNG.standard_normal((C, Dm)))):
            U /= np.linalg.norm(U, axis=1, keepdims=True)
            for it in range(iters + 1):
                G = U @ U.T
                ddiv = (G.sum() - np.trace(G)) / (C * (C - 1))
                if it % 10 == 0:
                    rows.append(dict(C=C, init=init, iter=it, d_div=ddiv, mean_norm=np.linalg.norm(U.mean(0)),
                                     bound=-1 / (C - 1)))
                grad = 2 * (U.sum(0, keepdims=True) - U) / (C * (C - 1))
                U = U - lr * C * grad
                U /= np.linalg.norm(U, axis=1, keepdims=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "prop4.csv", index=False)
    fin = df[df.iter == iters]
    setm("PfourGap", float((fin.d_div - fin.bound).abs().max()), "{:.1e}")
    setm("PfourMeanNorm", float(fin.mean_norm.max()), "{:.1e}")
    return df


# ============================================================ Proposition 2: size and cost
KS = (3, 5, 7, 9)


def count_params(Dd, Nw=NW, Np=NP, Dm=DM, dc=DC, Din=2, Kg=2, C=3, c0=4):
    ln = lambda n: 2 * n  # noqa: E731
    lin = lambda i, o: i * o + o  # noqa: E731
    p = lin(Din, Dd) + (Kg + 1)
    p += lin(dc, dc) + ln(dc) + lin(dc, dc)                                  # chaos encoder 2 d_c^2
    p += Nw * (lin(Dd + dc, Dd) + ln(Dd) + lin(Dd, Dd) + (Dd + dc) * Dd + 2 * Dd)  # waves 3D^2+2 d_c D
    p += Nw * (lin(Dm, Dd) + lin(Dd, Dd))                                   # interference D^2 + D_m D
    p += Np * 4 * (Dd * Dd + Dd)                                            # attention blocks
    p += lin(Np * Dd, Dd) + ln(Dd) + lin(Dd, Dd)                            # attention fusion (N_p+1) D^2
    p += sum(k * Dd * Dd + Dd for k in KS)                                  # convolutions
    p += lin(len(KS) * Dd, Dd) + lin(Dd, Dd)                                # scale fusion (N_f+1) D^2
    p += c0 * (Dd * Dd + Dd) + lin(Dd, Dm) + lin(Dm, Dd)                    # predictor + context
    p += C * Dm + 3 * Dm * Dm                                               # prototype bank
    return p


def c_ciwi(Nw=NW, Np=NP, c0=4):
    return 4 * Nw + 5 * Np + sum(KS) + len(KS) + 2 + c0


def check_prop5(L=4, dm=64):
    rows = []
    for Dd in (8, 16, 32, 64, 128, 256, 512, 1024):
        lead = c_ciwi() * Dd * Dd + NW * (2 * DC + DM) * Dd
        rows.append(dict(D=Dd, counted=count_params(Dd), leading=lead, transformer=12 * L * dm * dm + 13 * L * dm))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "prop5.csv", index=False)
    dstar = dm * math.sqrt(12 * L / c_ciwi())
    grid = np.arange(4, 400)
    counted_cross = grid[np.argmax([count_params(int(g)) > 12 * L * dm * dm + 13 * L * dm for g in grid])]
    setm("PfiveC", str(c_ciwi()))
    setm("PfiveDstar", dstar, "{:.1f}")
    setm("PfiveDcross", str(int(counted_cross)))
    setm("PfiveParamsSixteen", f"{count_params(16):,}")
    setm("PfiveParamsOneTwoEight", f"{count_params(128):,}")
    setm("PfiveRatioLarge", float(df.counted.iloc[-1] / df.leading.iloc[-1]), "{:.3f}")
    return df, dstar


# ============================================================ figures
def fig_lemmas(l1, l2, l2s, l3, l4, l5):
    cs = mf.CSTYLE

    def a(ax):
        for c in mf.CITIES:
            g = l1[l1.city == c]
            ax.semilogy(g["T"], g.mom_fd / g.mom_bound, color=cs[c][0], marker="o", ms=2.5, label=c)
            ax.semilogy(g["T"], g.ord_fd / g.ord_bound, color=cs[c][0], marker="s", ms=2.5, ls="--")
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xticks([6, 12, 24, 48])
        ax.set_xlabel("Window length $T$")
        ax.set_ylabel("Measured / bound")
        hs = [mf.Line2D([], [], color="k", marker="o", ms=2.5, label=r"$\Phi_{\rm mom}$"),
              mf.Line2D([], [], color="k", marker="s", ms=2.5, ls="--", label=r"$\Phi_{\rm ord}$")]
        hs += [mf.Line2D([], [], color=cs[c][0], label=mf.SHORT[c]) for c in mf.CITIES]
        ax.legend(handles=hs, fontsize=mf.fs(5.8), loc="center right", ncol=2, columnspacing=0.6)
        mf.clean_log(ax, "y", decades_only=True)

    def b(ax):
        for idx, col in (("raw", "#D55E00"), ("normalised", "#0072B2")):
            g = l2s[l2s["index"] == idx]
            ax.loglog(g["T"], g.bound, color=col, label=f"bound, {idx} $t$")
            ax.loglog(g["T"], g.fd, color=col, ls="--", marker="o", ms=2.5, label=f"measured, {idx} $t$")
        ax.set_xticks([6, 12, 24, 48], ["6", "12", "24", "48"])
        ax.xaxis.set_minor_formatter(mf.matplotlib.ticker.NullFormatter())
        mf.clean_log(ax, "y", decades_only=True)
        ax.set_xlabel("Window length $T$")
        ax.set_ylabel(r"$\Gamma$")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 60)
        ax.legend(fontsize=mf.fs(5.6), loc="upper left", ncol=2, columnspacing=0.6, handlelength=1.6)

    def bars(ax, df, cols, labels, colors, ylabel):
        x = np.arange(len(mf.CITIES))
        w = 0.8 / len(cols)
        for i, (col, lab, cc) in enumerate(zip(cols, labels, colors)):
            vals = [df[df.city == c][col].max() for c in mf.CITIES]
            ax.bar(x + (i - (len(cols) - 1) / 2) * w, vals, w, color=cc, label=lab)
        ax.axhline(1, color="k", lw=0.8)
        ax.set_yscale("log")
        mf.clean_log(ax, "y", decades_only=True)
        ax.set_xticks(x, [mf.SHORT[c] for c in mf.CITIES])
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)

    def c(ax):
        loc = l2[l2.version == "local"]
        bars(ax, loc, ["gamma_ratio", "gx_ratio", "g_ratio"], [r"$\Gamma$ (ii)", r"$L_{G,X}$ (iii)", r"$L_G$ (iv)"],
             ["#0072B2", "#56B4E9", "#009E73"], "Measured / bound (local)")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 1e7)
        ax.legend(fontsize=mf.fs(6), loc="upper right", ncol=3, columnspacing=0.6, handlelength=1)

    def d(ax):
        for cc in mf.CITIES:
            g = l3[l3.city == cc]
            ax.plot(g.gamma, g.I_ratio, color=cs[cc][0], marker=cs[cc][1], ms=2.5, label=cc)
            ax.plot(g.gamma, g.M_ratio, color=cs[cc][0], marker=cs[cc][1], ms=2.5, ls="--")
        ax.axhline(1, color="k", lw=0.8)
        ax.set_yscale("log")
        mf.clean_log(ax, "y", decades_only=True)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel(r"Interaction weight $\gamma$")
        ax.set_ylabel("Measured / bound")
        hs = [mf.Line2D([], [], color="k", label=r"$\tilde{\mathbf{I}}$: $L_{\mathcal{I}}$"),
              mf.Line2D([], [], color="k", ls="--", label=r"$\mathbf{M}_{\rm ctx}$: $L_M$")]
        ax.legend(handles=hs, fontsize=mf.fs(6), loc="center right")

    def e(ax):
        bars(ax, l4, ["ball_ratio", "head_ratio_ball", "head_ratio_data", "module_ratio_data"],
             [r"$\|\tilde{\mathbf{I}}\|/R_I$", r"head, $R_I$", "head, data radius", "module, local"],
             ["#999999", "#E69F00", "#D55E00", "#CC79A7"], "Ratio")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 3e4)
        ax.legend(fontsize=mf.fs(5.6), loc="upper right", ncol=2, columnspacing=0.6, handlelength=1)

    def f(ax):
        for cc in mf.CITIES:
            g = l5[l5.city == cc]
            ax.plot(g.N_w, g.var_emp / g.bound_vmax, color=cs[cc][0], marker=cs[cc][1], ms=2.5, label=cc)
            ax.plot(g.N_w, g.var_emp / g.bound_avar, color=cs[cc][0], marker=cs[cc][1], ms=2.5, ls=":")
        ax.axhline(1, color="k", lw=0.8)
        ax.set_yscale("log")
        mf.clean_log(ax, "y")
        ax.set_xticks([2, 4, 8, 12, 16, 20])
        ax.set_xlabel("$N_w$")
        ax.set_ylabel("Measured / bound")
        hs = [mf.Line2D([], [], color="k", label=r"bound with $v_{\max}$"),
              mf.Line2D([], [], color="k", ls=":", label=r"bound with Var$(A)$")]
        ax.set_ylim(0.15, 300)
        ax.legend(handles=hs, fontsize=mf.fs(6), loc="upper right", ncol=1)

    mf.compose("lemma_validation",
               [("Lemma 1: chaos proxy", a), (r"Lemma 2: $\Gamma$ vs. $T$", b), ("Lemma 2: wave generator", c),
                ("Lemma 3: interference", d), ("Lemma 4: attention", e), ("Lemma 5: superposition", f)],
               ncols=3, width=mf.TEXT_W, row_h=1.9)


def fig_props(p1, p2, p2a7, p3c, p4, p5, dstar):
    cs = mf.CSTYLE

    def a(ax):
        st = pd.read_csv(OUT / "prop1_stages.csv")
        order = ["pre", "chaos path", "waves | C", "C -> waves", "interference", "attention", "full map"]
        g = st.groupby("stage")[["fd", "bound"]].max().reindex(order)
        x = np.arange(len(order))
        ax.bar(x - 0.2, g.bound, 0.4, color="#E69F00", label="bound (local)")
        ax.bar(x + 0.2, g.fd, 0.4, color="#D55E00", label="measured")
        ax.set_yscale("log")
        mf.clean_log(ax, "y", decades_only=True)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 1e4)
        ax.set_xticks(x, ["pre", "chaos", "W|C", "C→W", "int.", "attn", "full"], rotation=35, ha="right",
                      rotation_mode="anchor")
        ax.set_ylabel("Lipschitz constant")
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=mf.fs(6), loc="upper left", ncol=2)

    def b(ax):
        for c in mf.CITIES:
            g = p2[p2.city == c]
            ax.plot(g["iter"] + 1, g.ce, color=cs[c][0], marker=cs[c][1], ms=2.5, label=c)
            ax.plot(g["iter"] + 1, g.bound, color=cs[c][0], ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        mf.clean_log(ax)
        ax.set_xticks([1, 10, 100], ["1", "10", "100"])
        ax.xaxis.set_minor_formatter(mf.matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Head training iteration + 1")
        ax.set_ylabel("Calibration error")
        hs = [mf.Line2D([], [], color="k", marker="o", ms=2.5, label="CE (measured)"),
              mf.Line2D([], [], color="k", ls=":", label=r"$\sqrt{\mathcal{R}(h)-\mathcal{R}(s^*)}$")]
        ax.legend(handles=hs, fontsize=mf.fs(6), loc="upper right")

    def c(ax):
        base = p2a7[p2a7.alpha == 0].groupby("city").mae.mean()
        for cc in mf.CITIES:
            for det, ls in ((True, "-"), (False, "--")):
                g = p2a7[(p2a7.city == cc) & (p2a7.detach == det)]
                ax.plot(g.alpha, 100 * (g.mae / base[cc] - 1), color=cs[cc][0], ls=ls, marker=cs[cc][1], ms=2.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(r"Weight of $\mathcal{L}_{\rm pred\_reg}$")
        ax.set_ylabel("Test MAE change (%)")
        hs = [mf.Line2D([], [], color="k", label="stop-gradient (A7)"),
              mf.Line2D([], [], color="k", ls="--", label="no stop-gradient")]
        hs += [mf.Line2D([], [], color=cs[c][0], marker=cs[c][1], ms=2.5, label=c) for c in mf.CITIES]
        ax.legend(handles=hs, fontsize=mf.fs(5.8), loc="upper left", ncol=1)

    def d(ax):
        p3 = pd.read_csv(OUT / "prop3.csv")
        settings = [20, 4, 1]
        x = np.arange(len(settings))
        for k, c in enumerate(mf.CITIES):
            g = p3[p3.city == c].set_index("days").reindex(settings)
            off = (k - 1.5) * 0.08
            ax.plot(x + off, g.var_emp / g.bound, color=cs[c][0], marker=cs[c][1], ms=3, ls="-")
            ax.plot(x + off, g.var_emp / g.naive, color=cs[c][0], marker=cs[c][1], ms=3, ls="--", mfc="none")
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xticks(x, ["20 days", "4 days", "1 day"])
        ax.set_xlabel(r"Distinct days per draw of $n_\tau$=20 episodes")
        ax.set_ylabel(r"Var$(\hat R)$ / bound")
        hs = [mf.Line2D([], [], color="k", marker="o", ms=3, label=r"$v/n_{\rm eff}$ (Thm. 3)"),
              mf.Line2D([], [], color="k", marker="o", ms=3, ls="--", mfc="none", label=r"naive $v/n_\tau$")]
        ax.legend(handles=hs, fontsize=mf.fs(6), loc="upper left")

    def e(ax):
        cols = {2: "#0072B2", 3: "#E69F00", 4: "#009E73", 8: "#CC79A7", 16: "#D55E00"}
        for C, col in cols.items():
            g = p4[(p4.C == C) & (p4.init == "orthogonal")]
            ax.plot(g["iter"], g.d_div, color=col, label=f"$|\\mathcal{{C}}|$={C}")
            ax.axhline(-1 / (C - 1), color=col, lw=0.6, ls=":")
        ax.set_xlabel("Gradient steps (from orthogonal)")
        ax.set_ylabel(r"$\mathcal{D}_{\rm div}$")
        ax.legend(fontsize=mf.fs(5.6), loc="lower right", ncol=2, columnspacing=0.6)

    def f(ax):
        ax.loglog(p5.D, p5.counted, color=mf.MSTYLE[mf.OURS][0], marker="o", ms=2.5, label="counted")
        ax.loglog(p5.D, p5.leading, color="k", ls="--", label="leading order")
        ax.loglog(p5.D, p5.transformer, color="#0072B2", ls="-.", label="Transformer")
        ax.axvline(dstar, color="0.5", lw=0.6, ls=":")
        ax.text(dstar * 1.08, p5.counted.min() * 1.5, f"$D^*$={dstar:.0f}", fontsize=mf.fs(6))
        ax.set_xticks([8, 16, 64, 256, 1024], ["8", "16", "64", "256", "1024"])
        ax.xaxis.set_minor_formatter(mf.matplotlib.ticker.NullFormatter())
        mf.clean_log(ax, "y", decades_only=True)
        ax.set_xlabel("Width $D$")
        ax.set_ylabel("Parameters")
        ax.legend(fontsize=mf.fs(5.6), loc="upper left")

    mf.compose("theorem_validation",
               [("Theorem 1: input stability", a), ("Theorem 2: calibration (i)-(ii)", b),
                ("Theorem 2: stop-gradient (A7)", c), ("Theorem 3: correlated episodes", d),
                ("Proposition 1: prototype diversity", e), ("Proposition 2: size and cost", f)],
               ncols=3, width=mf.TEXT_W, row_h=1.9)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mf.set_style(1.3)
    mf.RESULTS = OUT
    steps = [("Lemma 1", check_lemma1), ("Lemma 2", check_lemma2), ("Lemmas 3-4", check_lemma34),
             ("Lemma 5", check_lemma5), ("Theorem 1", check_prop1), ("Theorem 2", check_prop2),
             ("Theorem 3", check_prop3), ("Proposition 1", check_prop4), ("Proposition 2", check_prop5)]
    if "--plot-only" in sys.argv:
        rd = lambda n: pd.read_csv(OUT / f"{n}.csv")  # noqa: E731
        fig_lemmas(rd("lemma1"), rd("lemma2"), rd("lemma2_scaling"), rd("lemma3"), rd("lemma4"), rd("lemma5"))
        fig_props(rd("prop1"), rd("prop2_ce"), rd("prop2_stopgrad"), rd("prop3_coverage"), rd("prop4"),
                  rd("prop5"), DM * math.sqrt(12 * 4 / c_ciwi()))
        return
    res = {}
    for name, fn in steps:
        res[name] = fn()
        print(f"[ok]   {name}")
    l2, l2s = res["Lemma 2"]
    l3, l4 = res["Lemmas 3-4"]
    p2, p2a7, _ = res["Theorem 2"]
    _, p3c = res["Theorem 3"]
    p5, dstar = res["Proposition 2"]
    fig_lemmas(res["Lemma 1"], l2, l2s, l3, l4, res["Lemma 5"])
    fig_props(res["Theorem 1"], p2, p2a7, p3c, res["Proposition 1"], p5, dstar)
    (OUT / "theory_numbers.tex").write_text(
        "% Auto-generated by theory_checks.py\n" +
        "\n".join("\\newcommand{\\%s}{%s}" % (k, v) for k, v in sorted(MACROS.items())) + "\n")
    for k, v in sorted(MACROS.items()):
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
