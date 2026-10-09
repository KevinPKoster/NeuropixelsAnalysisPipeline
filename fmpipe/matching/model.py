"""
Cross-day matching, step 3: probabilities that two clusters on different days are one neuron.

Idea (no labelled cross-day ground truth exists, so everything is anchored in data we can trust):

* NEGATIVE reference: pairs of DIFFERENT units from the SAME recording that sit close to each
  other on the probe (they are certainly different neurons, and they are as similar as the
  hardest cross-day impostors are).
* For each pair of days, a two-component mixture is fit on the cross-day candidate pairs:
  "same neuron" (density learnt from the data by EM, starting from a seed of obvious
  matches) versus "different neurons" (density fixed from the negative reference).
  Features are treated as conditionally independent (naive Bayes) inside each component.
* One neuron can have only one partner: posteriors are scaled so that no unit's posteriors sum
  above 1, and the final matching is a one-to-one assignment.
* Calibration is CHECKED, not assumed: `decoy_fdr` repeats the matching with the other day's
  units artificially displaced by 160-320 um (pairs that cannot be the same neuron) and counts
  how many decoy 'matches' pass each confidence level.

Features: waveform footprint cosine, amplitude ratio, centroid offsets (after drift), ACG
shape, ISI shape, firing-rate ratio. No behaviour of any kind.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

from .features import (FEATURES, ROW_UM, DayProfiles, pair_table)

N_BINS = 160


# -------------------------------------------------------------- feature space
def to_space(df: pd.DataFrame) -> np.ndarray:
    """Transform raw features to roughly bell-shaped, unbounded coordinates (columns = FEATURES)."""
    z = lambda r: np.arctanh(np.clip(r, -0.999, 0.999))
    cols = {
        "wf_cos": z(df.wf_cos.values),
        "amp_abs_logratio": np.log(df.amp_abs_logratio.values + 0.03),
        "dy_abs": np.log(df.dy_abs.values + 2.0),
        "dx_abs": np.log(df.dx_abs.values + 2.0),
        "acg_mid_r": z(df.acg_mid_r.values),
        "isi_r": z(df.isi_r.values),
        "rate_abs_logratio": np.log(df.rate_abs_logratio.values + 0.03),
    }
    return np.column_stack([cols[f] for f in FEATURES])


ALL_COLS = tuple(range(len(FEATURES)))
# Features that do not depend on behavioural state: waveform footprint, amplitude, position.
# (ACG / ISI / firing rate do change with state, e.g. a different task condition on a given day.)
STATE_FREE_COLS = tuple(FEATURES.index(f) for f in ("wf_cos", "amp_abs_logratio", "dy_abs", "dx_abs"))


@dataclass
class Density:
    """Per-feature 1-D densities on a fixed grid (log scale), naive-Bayes combined."""
    lo: np.ndarray
    hi: np.ndarray
    logp: np.ndarray                     # (n_features_used, N_BINS)
    cols: tuple = ALL_COLS               # which of FEATURES this density covers

    def loglik(self, X):
        out = np.zeros(len(X))
        for k in range(X.shape[1]):
            t = (X[:, k] - self.lo[k]) / (self.hi[k] - self.lo[k]) * (N_BINS - 1)
            out += np.interp(t, np.arange(N_BINS), self.logp[k])
        return out


def fit_density(X, w=None, lo=None, hi=None, smooth_bins=4.0, floor=2e-3, cols=ALL_COLS) -> Density:
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    if lo is None:
        lo = np.percentile(X, 0.2, axis=0) - 0.5
        hi = np.percentile(X, 99.8, axis=0) + 0.5
    logp = np.zeros((X.shape[1], N_BINS))
    for k in range(X.shape[1]):
        h, _ = np.histogram(X[:, k], bins=N_BINS, range=(lo[k], hi[k]), weights=w)
        h = gaussian_filter1d(h.astype(float), smooth_bins, mode="nearest")
        p = h / max(h.sum(), 1e-12)
        p = np.maximum(p, floor / N_BINS)               # never exactly zero
        logp[k] = np.log(p / p.sum())
    return Density(lo, hi, logp, tuple(cols))


def common_range(*Xs):
    X = np.vstack(Xs)
    return np.percentile(X, 0.2, axis=0) - 0.5, np.percentile(X, 99.8, axis=0) + 0.5


# ------------------------------------------------------------------------ EM
def scale_one_to_one(p, ii, jj, n_a, n_b):
    """A unit has at most one partner: scale posteriors so that row and column sums do not exceed 1."""
    p = p.copy()
    for _ in range(3):
        rs = np.bincount(ii, weights=p, minlength=n_a)
        p = p / np.maximum(1.0, rs[ii])
        cs = np.bincount(jj, weights=p, minlength=n_b)
        p = p / np.maximum(1.0, cs[jj])
    return p


def seed_posteriors(df):
    """Obvious matches to start EM: mutual-best waveform cosine, high cosine, small offset."""
    best_i = df.groupby("i").wf_cos.idxmax()
    best_j = df.groupby("j").wf_cos.idxmax()
    mutual = df.index.isin(best_i.values) & df.index.isin(best_j.values)
    seed = mutual & (df.wf_cos.values > 0.8) & (df.dy_abs.values < 40)
    return np.where(seed, 0.9, 0.01)


def em_fit(df, f0: Density, lo, hi, n_a, n_b, n_iter=25):
    X = to_space(df)[:, list(f0.cols)]
    ii, jj = df.i.values, df.j.values
    p = seed_posteriors(df)
    for _ in range(n_iter):
        pi = float(np.clip(p.mean(), 1e-4, 0.5))
        f1 = fit_density(X, w=p, lo=lo, hi=hi, cols=f0.cols)
        llr = f1.loglik(X) - f0.loglik(X)
        post = 1.0 / (1.0 + np.exp(-(llr + np.log(pi / (1 - pi)))))
        p = scale_one_to_one(post, ii, jj, n_a, n_b)
    return f1, pi, p, llr


def posterior(df, f0, f1, pi, n_a, n_b):
    X = to_space(df)[:, list(f1.cols)]
    llr = f1.loglik(X) - f0.loglik(X)
    post = 1.0 / (1.0 + np.exp(-(llr + np.log(pi / (1 - pi)))))
    return scale_one_to_one(post, df.i.values, df.j.values, n_a, n_b)


def assign(df, p, min_p=0.05):
    """One-to-one assignment maximising the summed posterior; returns the assigned rows' index."""
    if len(df) == 0:
        return np.array([], int)
    ui, ri = np.unique(df.i.values, return_inverse=True)
    uj, cj = np.unique(df.j.values, return_inverse=True)
    M = np.zeros((len(ui), len(uj)))
    np.maximum.at(M, (ri, cj), p)
    r, c = linear_sum_assignment(-M)
    chosen = set(zip(ui[r], uj[c]))
    sel = [k for k, (i, j) in enumerate(zip(df.i.values, df.j.values)) if (i, j) in chosen and p[k] >= min_p]
    return np.array(sel, int)


# --------------------------------------------------------------------- drift
def estimate_drift(A: DayProfiles, B: DayProfiles, scan_um=(-80, -60, -40, -20, 0, 20, 40, 60, 80)):
    """
    Rigid depth shift of day B relative to day A (um): the shift at which the most waveform-mutual-best
    pairs agree, then refined to sub-row accuracy from the median centroid offset of those pairs.
    """
    best = (-1, 0.0, None)
    scores = {}
    for d in scan_um:
        df = pair_table(A, B, float(d))
        if len(df) == 0:
            continue
        bi = df.groupby("i").wf_cos.idxmax(); bj = df.groupby("j").wf_cos.idxmax()
        # a good drift value makes obvious matches ALSO agree in position (small centroid offset)
        mutual = (df.index.isin(bi.values) & df.index.isin(bj.values) & (df.wf_cos.values > 0.9)
                  & (df.dy_abs.values < 25))
        score = float(mutual.sum()) + 0.01 * float(df.wf_cos[mutual].sum())
        scores[d] = int(mutual.sum())
        if score > best[0]:
            best = (score, float(d), df[mutual])
    d0, matched = best[1], best[2]
    refine = float(np.median(matched.dy)) if matched is not None and len(matched) else 0.0
    return d0 + refine, scores


# ------------------------------------------------------------------- day pair
@dataclass
class DayPairResult:
    a: str
    b: str
    drift_um: float
    drift_scan: dict
    pi: float
    pairs: pd.DataFrame                  # candidate pairs with features, posterior 'p', 'assigned'
    n_a: int
    n_b: int
    f1: Density = None
    f0: Density = None
    decoys: list = field(default_factory=list)       # list of (shift_um, assigned posteriors)


def match_day_pair(A: DayProfiles, B: DayProfiles, f0: Density, lo, hi, decoy_shifts=(-320, -240, -160, 160, 240, 320)):
    drift, scan = estimate_drift(A, B)
    df = pair_table(A, B, drift)
    f1, pi, p, _ = em_fit(df, f0, lo, hi, A.n, B.n)
    df["p"] = p
    df["assigned"] = False
    df.loc[assign(df, p), "assigned"] = True
    res = DayPairResult(A.name, B.name, drift, scan, pi, df, A.n, B.n, f1, f0)
    for s in decoy_shifts:
        dd = pair_table(A, B, drift + s)
        if len(dd) == 0:
            continue
        pd_ = posterior(dd, f0, f1, pi, A.n, B.n)
        sel = assign(dd, pd_)
        res.decoys.append((s, pd_[sel]))
    return res


def feature_evidence(res: DayPairResult, rows: pd.DataFrame) -> np.ndarray:
    """Per-feature log-likelihood ratio (natural log; > 0 favours 'same neuron'); columns follow res.f1.cols."""
    X = to_space(rows)[:, list(res.f1.cols)]
    out = np.zeros_like(X)
    for k in range(X.shape[1]):
        for dens, sign in ((res.f1, 1.0), (res.f0, -1.0)):
            t = (X[:, k] - dens.lo[k]) / (dens.hi[k] - dens.lo[k]) * (N_BINS - 1)
            out[:, k] += sign * np.interp(t, np.arange(N_BINS), dens.logp[k])
    return out


def decoy_fdr(res: DayPairResult, levels=(0.6, 0.7, 0.8, 0.9, 0.95)):
    """Per confidence level: matches found, mean decoy 'matches', and the implied false-discovery rate."""
    out = []
    tp = res.pairs[res.pairs.assigned]
    for L in levels:
        n_t = int((tp.p >= L).sum())
        n_d = float(np.mean([(pd_ >= L).sum() for _, pd_ in res.decoys])) if res.decoys else np.nan
        out.append(dict(level=L, n_matches=n_t, mean_decoy_matches=n_d,
                        est_fdr=(n_d / n_t if n_t else np.nan), model_fdr=float(1 - tp.p[tp.p >= L].mean()) if n_t else np.nan))
    return pd.DataFrame(out)


def reference_densities(days: list[DayProfiles], cols=ALL_COLS):
    """Negative reference (different units, same day, neighbours) pooled over all days + same-unit reference."""
    from .features import same_unit_reference
    neg = pd.concat([pair_table(D, D, 0.0, same_day=True) for D in days], ignore_index=True)
    pos = pd.concat([same_unit_reference(D) for D in days], ignore_index=True)
    Xn, Xp = to_space(neg)[:, list(cols)], to_space(pos)[:, list(cols)]
    lo, hi = common_range(Xn, Xp)
    return fit_density(Xn, lo=lo, hi=hi, cols=cols), fit_density(Xp, lo=lo, hi=hi, cols=cols), lo, hi, neg, pos
