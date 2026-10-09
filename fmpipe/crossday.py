"""
DOWNSTREAM of the matching: how the tracked neurons respond around reaches, day by day.

The matching (fmpipe.matching) never sees these data. This module takes the neurons that the
matching tracked (at a confidence level fixed beforehand) and just displays their reach responses.
Everything plotted is a measured value: position = the unit's depth and x (day-1 frame, shifted by the
estimated drift), colour = signed peak z, size = |peak z|; no smoothing or clipping of values (colour limits are
shown and only affect the colour scale).

Population summaries use ONLY measured values: a day on which a neuron was not tracked (or has no PETH) simply
does not contribute to that day's average; a tracked neuron that is not significantly modulated contributes its
measured (small) peak z, not a substituted 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .matching.plots import _dv_center
from .peth import REGION_BAND_COLORS, THEMES, _theme_cmap

# Population groups. "thalamus" = everything >= 3000 um (THAL + VAL); VAL (>= 3500 um) is its further subdivision.
GROUPS = {"CFA": ("CFA",), "WM": ("WM",), "Thalamus (>=3000)": ("THAL", "VAL"), "VAL (>=3500)": ("VAL",)}
GROUP_COLORS = {"CFA": "#3b82f6", "WM": "#c4c9d1", "Thalamus (>=3000)": "#14b8a6", "VAL (>=3500)": "#22c55e"}
GROUP_STYLE = {"CFA": "-", "WM": ":", "Thalamus (>=3000)": "-", "VAL (>=3500)": "--"}


def tracked_trajectories(cfgs: dict, profiles: dict, results: dict, days: list, chains, subset="all", min_days=3):
    """
    Long table, one row per (tracked neuron, day it was found on): measured peak z, whether it was significant,
    latency, depth, and the neuron's region (from its mean drift-corrected depth over the days it was tracked).
    """
    resp = {d: pd.read_csv(c.analysis_dir / "peth" / f"reach_peth_{subset}_units.csv").set_index("id") for d, c in cfgs.items()}
    first = days[0]
    drift = {d: (0.0 if d == first else results[(first, d)].drift_um) for d in days}
    bounds = cfgs[first].region_bounds_dv_um
    rows = []
    for _, ch in chains[chains.n_days >= min_days].iterrows():
        recs = []
        for d in days:
            if d not in ch.members:
                continue
            i = ch.members[d]; P = profiles[d]; uid = int(P.unit_ids[i])
            has = uid in resp[d].index
            recs.append(dict(chain=int(ch.chain), day=d, unit=uid, dv_um=float(_dv_center(P)[i] + drift[d]),
                             x_um=float(P.cent_x.mean(1)[i]),
                             peak_z=float(resp[d].loc[uid, "peak_z"]) if has else np.nan,
                             modulated=bool(resp[d].loc[uid, "modulated"]) if has else False,
                             latency_s=float(resp[d].loc[uid, "peak_latency_s"]) if has else np.nan))
        mean_dv = np.mean([r["dv_um"] for r in recs])
        region = next((n for n, (b0, b1) in bounds.items() if b0 <= mean_dv < b1), "other")
        for r in recs:
            r["region"] = region
        rows += recs
    return pd.DataFrame(rows)


def summarise_by_region(tr: pd.DataFrame, days: list):
    """Per region and day: n neurons, mean +- SEM of signed peak z and of |peak z| (measured values only)."""
    out = []
    for reg, members in GROUPS.items():
        for d in days:
            v = tr[tr.region.isin(members) & (tr.day == d)].peak_z.dropna().values
            out.append(dict(region=reg, day=d, n=len(v),
                            mean_z=v.mean() if len(v) else np.nan,
                            sem_z=v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                            mean_abs_z=np.abs(v).mean() if len(v) else np.nan,
                            sem_abs_z=np.abs(v).std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan))
    return pd.DataFrame(out)


def plot_tracked_probe(cfgs: dict, profiles: dict, results: dict, days: list, chains, subset="all", theme="dark",
                       vlim=10.0, min_days=3, title="", save_path=None):
    """Tracked neurons along the probe on every day (lines join the same neuron) + population trajectories by region."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    th = THEMES[theme]; cmap = _theme_cmap(th); norm = TwoSlopeNorm(0, -vlim, vlim)
    cfg0 = cfgs[days[0]]
    tr = tracked_trajectories(cfgs, profiles, results, days, chains, subset, min_days)
    summ = summarise_by_region(tr, days)
    n_chains = tr.chain.nunique()
    rcol = dict(GROUP_COLORS)
    if theme == "light":
        rcol["WM"] = "#6b7280"
    line_col = {"CFA": rcol["CFA"], "WM": rcol["WM"], "THAL": rcol["Thalamus (>=3000)"], "VAL": rcol["Thalamus (>=3000)"]}

    plt.rcParams.update({"text.color": th["fg"], "axes.labelcolor": th["fg"], "xtick.color": th["dim"],
                         "ytick.color": th["dim"], "axes.edgecolor": th["dim"]})
    fig = plt.figure(figsize=(18, 11.5), facecolor=th["bg"])
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.15], wspace=0.14, left=0.06, right=0.985, top=0.87, bottom=0.14)
    ax = fig.add_subplot(gs[0])
    gs2 = gs[1].subgridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.30)
    ax2, ax3 = fig.add_subplot(gs2[0]), fig.add_subplot(gs2[1], sharex=None)
    for a in (ax, ax2, ax3):
        a.set_facecolor(th["bg"])
        for s in a.spines.values():
            s.set_alpha(0.4)
    for name, (b0, b1) in cfg0.region_bounds_dv_um.items():
        ax.axhspan(b0, b1, color=REGION_BAND_COLORS.get(name, "#888"), alpha=th["band_alpha"], lw=0, zorder=0)

    first = days[0]
    drift = {d: (0.0 if d == first else results[(first, d)].drift_um) for d in days}
    xpos = lambda x_um, k: k + (x_um - 35.0) / 150.0
    for k, d in enumerate(days):                     # every good unit of every day, faint
        P = profiles[d]
        ax.scatter(xpos(P.cent_x.mean(1), k), _dv_center(P) + drift[d], s=5, color=th["nonmod"], alpha=0.25, lw=0, zorder=1)
    for _, g in tr.groupby("chain"):
        g = g.sort_values("day")
        kx = [days.index(d) for d in g.day]
        xs = [xpos(x, k) for x, k in zip(g.x_um, kx)]
        ax.plot(xs, g.dv_um, color=th["fg"], alpha=0.22, lw=0.8, zorder=2)
        for x, (_, r) in zip(xs, g.iterrows()):
            if r.modulated:
                ax.scatter([x], [r.dv_um], s=26 + 15 * min(abs(r.peak_z), vlim), c=[r.peak_z], cmap=cmap, norm=norm, alpha=0.95, lw=0, zorder=4)
            else:
                ax.scatter([x], [r.dv_um], s=20, color=th["nonmod"], alpha=0.8, lw=0, zorder=3)
    allv = np.concatenate([_dv_center(profiles[d]) + drift[d] for d in days])
    ax.set_ylim(allv.max() + 60, allv.min() - 60); ytop = allv.min() - 60
    ax.set_xticks(range(len(days))); ax.set_xticklabels(days, fontsize=12); ax.set_xlim(-0.5, len(days) - 0.5)
    for name, (b0, b1) in cfg0.region_bounds_dv_um.items():
        ax.text(-0.44, max(b0, ytop) + 18, name, color=REGION_BAND_COLORS.get(name, th["fg"]), fontsize=11, fontweight="bold", va="top", zorder=6)
    ax.set_ylabel(f"DV depth below surface (um), {first} frame (drift-corrected)")
    ax.set_title(f"{n_chains} neurons tracked on >= {min_days} days  (lines join the same neuron)", fontsize=12, pad=8)

    # population trajectories by region: measured values only, days a neuron was not tracked are absent
    dd = np.arange(len(days))
    for a, key, skey, ylab, ttl in ((ax2, "mean_z", "sem_z", "peak z of reach response (measured)",
                                     "signed response: mean +- SEM by region"),
                                    (ax3, "mean_abs_z", "sem_abs_z", "|peak z| (response strength)",
                                     "strength regardless of direction: mean +- SEM")):
        if key == "mean_z":
            for _, g in tr.groupby("chain"):
                g = g.sort_values("day")
                a.plot([days.index(d) for d in g.day], g.peak_z, color=line_col.get(g.region.iloc[0], th["dim"]),
                       alpha=0.13, lw=0.8)
        for reg in GROUPS:
            s = summ[summ.region == reg].set_index("day").loc[days]
            if s.n.sum() == 0:
                continue
            m, e = s[key].values.astype(float), s[skey].values.astype(float)
            a.fill_between(dd, m - np.nan_to_num(e), m + np.nan_to_num(e), color=rcol[reg], alpha=0.12 if reg.startswith("VAL") else 0.18, lw=0)
            a.plot(dd, m, GROUP_STYLE[reg], color=rcol[reg], lw=2.6 if not reg.startswith("VAL") else 2.0, marker="o", ms=7 if not reg.startswith("VAL") else 5,
                   zorder=5, label=f"{reg} (n per day: {', '.join(str(int(n)) for n in s.n)})")
        if key == "mean_z":
            a.axhline(0, color=th["dim"], lw=0.8)
        a.set_xticks(dd); a.set_xticklabels(days); a.set_ylabel(ylab, fontsize=10)
        a.set_title(ttl, fontsize=12, pad=8)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.set_ylim(np.nanpercentile(tr.peak_z, 1) - 0.5, np.nanpercentile(tr.peak_z, 99) + 0.5)

    cax = fig.add_axes([0.07, 0.055, 0.25, 0.016])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label(f"peak z vs shuffled (colour limits +-{vlim:g}; values are not altered)", color=th["fg"], fontsize=9)
    cb.ax.tick_params(colors=th["dim"], labelsize=8); cb.outline.set_edgecolor(th["dim"])
    fig.text(0.06, 0.955, title or "Mouse 4103  |  tracked neurons: reach responses along the probe", fontsize=18,
             fontweight="bold", color=th["fg"], va="center")
    fig.text(0.06, 0.915, "Matching used waveforms, position, ACG/ISI and rate only; responses were never used to match.  "
             "Population lines use measured values only: days a neuron was not tracked are left out, nothing is replaced by 0.",
             fontsize=10.5, color=th["dim"], va="center")
    if save_path:
        fig.savefig(save_path, dpi=160, facecolor=th["bg"])
    plt.rcdefaults()
    return fig
