"""
DOWNSTREAM tests built on the cross-day matching. The matching itself never saw any of these data (reach responses are the
independent variable); here they are used to ask two things.

1. validity_table / summarise_validity
   Do units that were matched (posterior >= level) have MORE SIMILAR reach responses than the clearly-rejected candidates
   for the same unit? For every matched pair (i on day A, j on day B) the correlation of their z-scored PETHs (response window) is compared
   with the correlation of i with the other candidates j' near it that the model rejected (posterior < 0.05), and of j with the other
   candidates i' (paired design: the unit's own response is held fixed). Run at every confidence level, never used to choose one.

2. training_table / summarise_training
   Do individual tracked neurons change their reach response over the days (Sept 15 -> 18)? Per neuron and day the response is the
   net z (mean z over the response window) at a COMMON number of events (see peth.run_matched_peth), with bootstrap draws. Per neuron:
   slope across days with a bootstrap CI. Per group: mean slope against a day-label permutation null, and the excess of the observed
   spread of slopes over what sampling noise alone produces (a measure of genuine cell-to-cell differences in change).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .crossday import tracked_trajectories
from .peth import THALAMUS, load_matched_peth, load_reach_peth

CTRL_P = 0.05            # candidates with posterior below this count as clearly rejected


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def load_full(cfgs: dict, subset: str = "all") -> dict:
    out = {}
    for d, cfg in cfgs.items():
        r = load_reach_peth(cfg, subset)
        c, rw = r["centers"], tuple(r["response_win"])
        m = (c >= rw[0]) & (c < rw[1])
        out[d] = dict(row={int(u): k for k, u in enumerate(r["units"])}, z=r["z"][:, m], mod=r["modulated"], pk=r["peak_z"])
    return out


# ------------------------------------------------------------------------------ 1. validity
def validity_table(match: dict, full: dict, levels=(0.6, 0.7, 0.8, 0.9, 0.95)) -> pd.DataFrame:
    P = match["profiles"]
    rows = []
    for (a, b), res in match["results"].items():
        df = res.pairs
        ua, ub = P[a].unit_ids, P[b].unit_ids
        fa, fb = full[a], full[b]
        rej = df[df.p < CTRL_P]
        rej_i = rej.groupby("i").j.apply(list).to_dict()
        rej_j = rej.groupby("j").i.apply(list).to_dict()
        for L in levels:
            for i, j, p in df[df.assigned & (df.p >= L)][["i", "j", "p"]].itertuples(index=False):
                A, B = int(ua[i]), int(ub[j])
                if A not in fa["row"] or B not in fb["row"]:
                    continue
                ra, rb = fa["row"][A], fb["row"][B]
                r_m = _corr(fa["z"][ra], fb["z"][rb])
                ctrl = []          # (r, both_modulated, same_sign)
                for jj in rej_i.get(i, []):
                    if jj != j and int(ub[jj]) in fb["row"]:
                        rr = fb["row"][int(ub[jj])]
                        ctrl.append((_corr(fa["z"][ra], fb["z"][rr]), bool(fa["mod"][ra] and fb["mod"][rr]), np.sign(fa["pk"][ra]) == np.sign(fb["pk"][rr])))
                for ii in rej_j.get(j, []):
                    if ii != i and int(ua[ii]) in fa["row"]:
                        rr = fa["row"][int(ua[ii])]
                        ctrl.append((_corr(fa["z"][rr], fb["z"][rb]), bool(fa["mod"][rr] and fb["mod"][rb]), np.sign(fa["pk"][rr]) == np.sign(fb["pk"][rb])))
                cr = np.array([c[0] for c in ctrl], float)
                both = bool(fa["mod"][ra] and fb["mod"][rb])
                rows.append(dict(pair=f"{a} vs {b}", level=L, unit_a=A, unit_b=B, p=float(p), r_matched=r_m,
                                 r_control=np.nanmean(cr) if len(cr) else np.nan, n_controls=len(cr),
                                 delta=(r_m - np.nanmean(cr)) if len(cr) else np.nan,
                                 mod_a=bool(fa["mod"][ra]), mod_b=bool(fb["mod"][rb]), informative=bool(fa["mod"][ra] or fb["mod"][rb]), both_modulated=both,
                                 same_sign_matched=bool(np.sign(fa["pk"][ra]) == np.sign(fb["pk"][rb])) if both else np.nan,
                                 ctrl_both_n=int(sum(c[1] for c in ctrl)), ctrl_both_same=int(sum(c[1] and c[2] for c in ctrl))))
    return pd.DataFrame(rows)


def summarise_validity(vt: pd.DataFrame, n_boot=2000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for L, g in vt.groupby("level"):
        for name, sub in (("all matched pairs", g), ("pairs with >= 1 modulated unit", g[g.informative])):
            d = sub.delta.dropna().values
            if len(d) < 5:
                continue
            boots = [np.median(rng.choice(d, len(d))) for _ in range(n_boot)]
            w = stats.wilcoxon(d, alternative="two-sided") if np.any(d != 0) else None
            both = sub[sub.both_modulated]
            n_both = len(both); same = int(both.same_sign_matched.sum()) if n_both else 0
            cn, cs = int(sub.ctrl_both_n.sum()), int(sub.ctrl_both_same.sum())
            out.append(dict(level=L, subset=name, n_pairs=len(d), median_r_matched=sub.r_matched.median(), median_r_control=sub.r_control.median(),
                            median_delta=float(np.median(d)), ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)),
                            wilcoxon_p=float(w.pvalue) if w is not None else np.nan,
                            frac_delta_positive=float((d > 0).mean()),
                            n_both_modulated=n_both, same_direction_matched=(same / n_both if n_both else np.nan),
                            n_ctrl_both_modulated=cn, same_direction_control=(cs / cn if cn else np.nan)))
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------ 2. training
def _slope(x, Y):
    xc = x - x.mean()
    return ((Y - Y.mean(axis=-1, keepdims=True)) * xc).sum(axis=-1) / (xc ** 2).sum()


def training_table(match: dict, cfgs: dict, chains: pd.DataFrame, n_events: int = 23, subset: str = "all", min_days: int = 3,
                   full: dict | None = None, seed: int = 3):
    """Per tracked neuron: net response per day (matched N, bootstrap) and its slope across days with a CI."""
    days = match["days"]
    full = full or load_full(cfgs, subset)
    matched = {d: load_matched_peth(cfg, n_events, subset) for d, cfg in cfgs.items()}
    row = {d: {int(u): k for k, u in enumerate(matched[d]["units"])} for d in days}
    tr = tracked_trajectories(cfgs, match["profiles"], match["results"], days, chains, subset=subset, min_days=min_days)
    rng = np.random.default_rng(seed)
    neurons, perday = [], []
    for ch, g in tr.groupby("chain"):
        g = g.sort_values("day"); ds = [d for d in g.day if int(g[g.day == d].unit.iloc[0]) in row[d]]
        if len(ds) < min_days:
            continue
        x = np.array([days.index(d) for d in ds], float)
        Y = np.column_stack([matched[d]["net"][:, row[d][int(g[g.day == d].unit.iloc[0])]] for d in ds])      # (n_boot, n_days)
        sl = _slope(x, Y)
        mean_y = Y.mean(0)
        obs = float(_slope(x, mean_y[None, :])[0])
        ever_mod = any(bool(full[d]["mod"][full[d]["row"][int(g[g.day == d].unit.iloc[0])]]) for d in ds if int(g[g.day == d].unit.iloc[0]) in full[d]["row"])
        lo, hi = np.percentile(sl, [2.5, 97.5])
        neurons.append(dict(chain=int(ch), region=g.region.iloc[0], n_days=len(ds), days=",".join(ds), slope=obs, slope_lo=float(lo), slope_hi=float(hi),
                            slope_boot_sd=float(sl.std()), ever_modulated=ever_mod,
                            change="increase" if lo > 0 else ("decrease" if hi < 0 else "stable"),
                            **{f"net_{d}": (float(mean_y[ds.index(d)]) if d in ds else np.nan) for d in days},
                            **{f"net_sd_{d}": (float(Y[:, ds.index(d)].std()) if d in ds else np.nan) for d in days}))
    return pd.DataFrame(neurons)


def summarise_training(nt: pd.DataFrame, days: list, n_perm=5000, seed=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = {"CFA": ("CFA",), "thalamus (>=3000)": THALAMUS, "VAL (>=3500)": ("VAL",), "WM": ("WM",)}
    out = []
    for name, regs in groups.items():
        for sub_name, sub in (("all tracked", nt[nt.region.isin(regs)]), ("modulated on >= 1 day", nt[nt.region.isin(regs) & nt.ever_modulated])):
            if len(sub) < 4:
                continue
            m = sub.slope.mean()
            # permutation null: shuffle the day labels of each neuron's per-day values
            vals = [np.array([r[f"net_{d}"] for d in days if not np.isnan(r[f"net_{d}"])]) for _, r in sub.iterrows()]
            xs = [np.array([days.index(d) for d in days if not np.isnan(r[f"net_{d}"])], float) for _, r in sub.iterrows()]
            perm = np.empty(n_perm)
            for k in range(n_perm):
                perm[k] = np.mean([_slope(x, v[rng.permutation(len(v))][None, :])[0] for x, v in zip(xs, vals)])
            p_mean = float((np.abs(perm) >= abs(m)).mean())
            noise_var = float((sub.slope_boot_sd ** 2).mean()); obs_var = float(sub.slope.var(ddof=1))
            out.append(dict(group=name, subset=sub_name, n=len(sub), mean_slope_per_day=m, mean_slope_perm_p=p_mean,
                            frac_increase=float((sub.change == "increase").mean()), frac_decrease=float((sub.change == "decrease").mean()),
                            frac_stable=float((sub.change == "stable").mean()), slope_sd_observed=float(np.sqrt(obs_var)),
                            slope_sd_from_sampling_noise=float(np.sqrt(noise_var)), excess_slope_sd=float(np.sqrt(max(obs_var - noise_var, 0.0)))))
    return pd.DataFrame(out)
