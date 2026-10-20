"""
Grant figures (Jason Maclean R01). Presentation only - no analysis happens here.

Everything is read from `peth/reach_peth_<subset>.npz`, which stage 6 already wrote. This
module crops, sorts and draws. If a number in a figure looks wrong, the fix belongs in
stage 6, not here.

One figure so far:

`stacked_peth_by_day` - reach-locked PETHs for every day side by side, CFA stacked above
thalamus, cropped to a short window around reach onset. How to describe it in one sentence:
*each row is one unit's firing rate around reach onset, z-scored against a shuffled null,
with units grouped as excited / suppressed / not modulated and sorted by when their response
peaks.*

Design follows the grant rules: >= 10 pt type, white background, no top/right spines on the
line plots, CFA blue / thalamus green throughout, PDF + 300 dpi PNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import SessionConfig
from .peth import THALAMUS, load_reach_peth, peak_change
from .units import load_units

CFA_BLUE = "#2196F3"
THAL_GREEN = "#4CAF50"
FONT = 10


def _sorted_rows(z, mod, peak_z, c, win):
    """
    Row order within a region: excited first, then suppressed, then not modulated; the two
    modulated groups sorted by the time of their largest |z| INSIDE THE DISPLAYED WINDOW
    (so the sort matches what the reader can see). Returns the order and the two boundaries.
    """
    idx = np.arange(len(z))
    w = (c >= win[0]) & (c <= win[1])
    lat = lambda ii: c[w][np.argmax(np.abs(z[ii][:, w]), axis=1)] if len(ii) else ii
    exc = idx[mod & (peak_z > 0)]
    sup = idx[mod & (peak_z <= 0)]
    non = idx[~mod]
    exc = exc[np.argsort(lat(exc))] if len(exc) else exc
    sup = sup[np.argsort(lat(sup))] if len(sup) else sup
    return np.concatenate([exc, sup, non]), len(exc), len(sup)


def stacked_peth_by_day(cfgs: dict, out_dir, subset: str = "all", win=(-1.0, 1.0),
                        vlim: float = 7.5, save=True):
    """
    cfgs: {"Sept 14": SessionConfig, ...} in the order the columns should appear.

    Columns = days; rows = CFA heatmap, thalamus heatmap, mean firing rate. One shared colour
    scale (+-vlim, clipped) so days are comparable by eye.

    Caveat the figure cannot show: z grows with the number of events, and the days differ
    (23 to 75 bouts), so compare the PATTERN across days, not the depth of colour. Use
    `run_matched_peth` if a quantitative across-day comparison is needed.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({"font.size": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT - 1,
                     "ytick.labelsize": FONT - 1, "legend.fontsize": FONT - 1,
                     "axes.titlesize": FONT, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})

    days = list(cfgs)
    data = {}
    for name, cfg in cfgs.items():
        r = load_reach_peth(cfg, subset)
        data[name] = r

    n = len(days)
    fig = plt.figure(figsize=(2.55 * n + 1.1, 7.6))
    gs = fig.add_gridspec(3, n + 1, height_ratios=[3, 3, 1.7],
                          width_ratios=[1] * n + [0.055],
                          hspace=0.14, wspace=0.12, left=0.075, right=0.945,
                          top=0.855, bottom=0.085)

    regions = [("CFA", ("CFA",), CFA_BLUE), ("Thalamus", THALAMUS, THAL_GREEN)]
    im, trace_axes = None, []
    for col, name in enumerate(days):
        r = data[name]
        c, z, reg, mod, pz = r["centers"], r["z"], r["region"], r["modulated"], r["peak_z"]
        keep = (c >= win[0]) & (c <= win[1])
        cw = c[keep]

        for row, (rname, members, colour) in enumerate(regions):
            ax = fig.add_subplot(gs[row, col])
            m = np.isin(reg, members)
            if not m.any():
                ax.set_axis_off()
                continue
            order, ne, ns = _sorted_rows(z[m], mod[m], pz[m], c, win)
            zz = z[m][order][:, keep]
            im = ax.imshow(zz, aspect="auto", extent=[cw[0], cw[-1], len(zz), 0],
                           cmap="RdBu_r", vmin=-vlim, vmax=vlim, interpolation="nearest")
            ax.axvline(0, color="k", lw=1.2, ls="--")
            if ne and ns:
                ax.axhline(ne, color="k", lw=0.7, ls=":", alpha=0.7)
            if 0 < ne + ns < len(zz):
                ax.axhline(ne + ns, color="k", lw=1.3)
            ax.set_xlim(win)
            ax.set_xticks([win[0], 0, win[1]])
            ax.set_xticklabels([])                      # only the bottom row carries the axis
            if row == 0:
                ax.set_title(f"{name}\n{len(r['event_times'])} reaches", pad=6)
            if col == 0:
                ax.set_ylabel(f"{rname}\nunits", color=colour, fontweight="bold")
            else:
                ax.set_yticklabels([])
            ax.text(0.03, 0.04, f"n={len(zz)} ({ne + ns} mod.)", transform=ax.transAxes,
                    fontsize=FONT - 2, va="bottom",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

        # mean |Z| of the modulated units: how strongly the average responsive unit departs
        # from its shuffled baseline, and when. Signed z would cancel excited against
        # suppressed; |z| does not, and keeps the latency comparison readable.
        ax = fig.add_subplot(gs[2, col], sharey=trace_axes[0] if trace_axes else None)
        trace_axes.append(ax)
        for rname, members, colour in regions:
            m = np.isin(reg, members) & mod
            if not m.any():
                continue
            a = np.abs(z[m][:, keep])
            mu, se = a.mean(0), a.std(0) / np.sqrt(m.sum())
            ax.plot(cw, mu, color=colour, lw=1.8, label=rname)   # per-day n is on the heatmaps
            ax.fill_between(cw, mu - se, mu + se, color=colour, alpha=0.25, lw=0)
        ax.axvline(0, color="k", lw=1.0, ls="--")
        ax.set_xlim(win); ax.set_xticks([win[0], 0, win[1]])
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_xlabel("Time from reach onset (s)")
        if col == 0:
            ax.set_ylabel("Mean |Z|\n(modulated units)")
            ax.legend(frameon=False, loc="upper left", fontsize=FONT - 2,
                      handlelength=1.2, borderpad=0.1, labelspacing=0.25)
        else:
            ax.tick_params(labelleft=False)

    cax = fig.add_subplot(gs[0:2, n])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"Z-score vs. shuffled (clipped at +-{vlim:g})")
    cb.outline.set_visible(False)

    zthr = {float(r["z_threshold"]) for r in data.values()}
    zt = f"{zthr.pop():.2f}" if len(zthr) == 1 else "per day"
    fig.suptitle("Reach-locked firing in motor cortex and motor thalamus, mouse 4103",
                 fontsize=FONT + 2, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, f"Each row is one unit, z-scored against a shuffled null. Units are "
                         f"grouped as excited / suppressed / not modulated (|Z| > {zt} for "
                         f">= 80 ms) and sorted by response latency.",
             ha="center", fontsize=FONT - 1, color="#444444")

    if save:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = out / f"stacked_peth_by_day_{subset}"
        fig.savefig(f"{stem}.png", dpi=300, facecolor="white")
        fig.savefig(f"{stem}.pdf", facecolor="white")
        print(f"saved {stem}.png and .pdf")
    return fig


# --------------------------------------------------------- change from baseline
def _strip(ax, x, vals, colour, width=0.3, rng=None):
    """Jittered points + median bar: show every unit, not a box that hides them."""
    rng = rng or np.random.default_rng(0)
    if len(vals) == 0:
        return
    j = rng.uniform(-width / 2, width / 2, size=len(vals))
    ax.scatter(x + j, vals, s=11, color=colour, alpha=0.55, lw=0, zorder=2)
    m = np.median(vals)
    ax.plot([x - width * 0.85, x + width * 0.85], [m, m], color="k", lw=2, zorder=3)


def change_from_baseline(cfgs: dict, out_dir, subset: str = "all", window=(-0.5, 1.0),
                         save=True):
    """
    Firing change from each unit's own baseline, session by session.

    Baseline = `null_mean`, the unit's rate at randomly placed event times, smoothed exactly
    like the real PSTH. For every unit the bin of largest |change| inside `window` gives
        delta Hz = PSTH - baseline      and      % change = 100 x delta Hz / baseline.

    Panels: (A) delta Hz per day, excited and suppressed shown separately because they have
    opposite sign; (B) the same as % of baseline; (C) the event-count caveat - both |delta Hz|
    and |Z| depend on how many reaches a session had, in OPPOSITE directions, so neither can
    be compared across sessions at face value.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"font.size": FONT, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})

    days = list(cfgs)
    rows = []
    for name, cfg in cfgs.items():
        r = load_reach_peth(cfg, subset)
        pc = peak_change(r, window)
        reg, mod, pz = r["region"], r["modulated"], r["peak_z"]
        rows.append(dict(name=name, n_ev=len(r["event_times"]), reg=reg, mod=mod, pz=pz,
                         dhz=pc["delta_hz"], pct=pc["pct_change"], z=r["peak_z"]))

    fig = plt.figure(figsize=(14.5, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1], left=0.055, right=0.985,
                          top=0.80, bottom=0.10, wspace=0.26)
    regions = [("CFA", ("CFA",), CFA_BLUE), ("Thalamus", THALAMUS, THAL_GREEN)]

    for p, (key, lab) in enumerate([("dhz", "Change from baseline (Hz)"),
                                    ("pct", "Change from baseline (%)")]):
        ax = fig.add_subplot(gs[0, p])
        ticks, labels = [], []
        for di, d in enumerate(rows):
            for ri, (rname, members, colour) in enumerate(regions):
                x = di * 2.4 + ri * 1.0
                m = np.isin(d["reg"], members) & d["mod"]
                _strip(ax, x, d[key][m & (d["pz"] > 0)], colour)
                _strip(ax, x, d[key][m & (d["pz"] <= 0)], colour)
                ticks.append(x); labels.append(rname[0])
            fig.text(0, 0, "")           # keep tight_layout happy on some backends
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=FONT - 1)
        for di, d in enumerate(rows):
            ax.text(di * 2.4 + 0.5, 1.035, f"{d['name']}\n{d['n_ev']} reaches",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=FONT - 2)
        ax.set_ylabel(lab)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.text(-0.13, 1.10, "AB"[p], transform=ax.transAxes, fontsize=FONT + 2,
                fontweight="bold", va="top")
        if key == "pct":
            shown = np.concatenate([d["pct"][d["mod"]] for d in rows])
            ax.set_ylim(np.percentile(shown, 2) * 1.15, np.percentile(shown, 98) * 1.15)

    # C: why these cannot be compared across sessions at face value
    ax = fig.add_subplot(gs[0, 2])
    N = np.array([d["n_ev"] for d in rows], float)
    mz = np.array([np.median(np.abs(d["z"][d["mod"]])) for d in rows])
    md = np.array([np.median(np.abs(d["dhz"][d["mod"]])) for d in rows])
    o = np.argsort(N)                       # join points along N, not along day order
    ax.plot(N[o], (mz / mz.mean())[o], "o-", color="#616161", lw=1.6, ms=6, label="median |Z|")
    ax.plot(N[o], (md / md.mean())[o], "s-", color="#C62828", lw=1.6, ms=6,
            label=r"median |$\Delta$ Hz|")
    for x, y, nm in zip(N, md / md.mean(), [d["name"] for d in rows]):
        ax.annotate(nm.replace("Sept ", ""), (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=FONT - 3, color="#C62828")
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("Reaches in the session")
    ax.set_ylabel("Metric / its mean across days")
    ax.legend(frameon=False, fontsize=FONT - 2, loc="lower left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(-0.18, 1.10, "C", transform=ax.transAxes, fontsize=FONT + 2,
            fontweight="bold", va="top")
    ax.set_title(f"r(N,|Z|)={np.corrcoef(N, mz)[0, 1]:+.2f},   "
                 f"r(N,|$\\Delta$Hz|)={np.corrcoef(N, md)[0, 1]:+.2f}\n"
                 f"opposite directions - match N before comparing days",
                 fontsize=FONT - 1)

    fig.suptitle("Reach-related change from each unit's own baseline, "
                 f"peak within {window[0]:+g} to {window[1]:+g} s",
                 fontsize=FONT + 2, fontweight="bold")
    fig.text(0.5, 0.925, "C = CFA, T = thalamus. One dot = one modulated unit, bar = median. "
                         "Excited units sit above zero, suppressed below. "
                         "Baseline = the unit's own rate at randomly placed event times.",
             ha="center", fontsize=FONT - 2, color="#444444")
    if save:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        stem = out / f"change_from_baseline_{subset}"
        fig.savefig(f"{stem}.png", dpi=300, facecolor="white")
        fig.savefig(f"{stem}.pdf", facecolor="white")
        print(f"saved {stem}.png and .pdf")
    return fig


# ------------------------------------------------------------- exemplar rasters
def _trial_order(cfg, ev, how):
    """
    Row order for a raster. 'time' = the order the reaches happened (safe default: it also
    shows drift over the session). 'amplitude' = by how far the paw got (max_y from
    bouts.npz), a BEHAVIOURAL covariate.

    Deliberately absent: sorting trials by the unit's own firing. Sorting units by latency in
    a population heatmap is standard, but sorting a single unit's trials by its own response
    is circular - it draws a diagonal out of pure noise, because the thing sorted on is the
    thing being displayed. Sort trials by behaviour, then the structure means something.
    """
    if how == "time":
        return np.arange(len(ev)), "reach # (chronological)"
    if how != "amplitude":
        raise ValueError(f"unknown trial sort {how!r} (use 'time' or 'amplitude')")
    b = np.load(cfg.analysis_dir / "peak_detection" / "bouts.npz")
    j = np.searchsorted(np.sort(b["onset_t"]), ev)       # ev came from the same bout onsets
    amp = b["max_y"][np.argsort(b["onset_t"])][np.clip(j, 0, len(b["max_y"]) - 1)]
    return np.argsort(amp), "reach, sorted by paw extent"


def exemplar_rasters(cfg: SessionConfig, out_dir, subset: str = "all", n_excited=10,
                     n_suppressed=8, win=(-1.0, 1.0), n_cols=10, trial_sort="time",
                     save=True):
    """
    Raster + PSTH for the clearest responders, in the reading order of Kevin's old
    `PETHforTopUnits` notebook: spikes on top, firing rate below.

    Units are picked by |peak Z| among the modulated ones, split excited / suppressed and
    balanced between CFA and thalamus so neither region dominates the panel. Each unit's
    baseline (its rate at random times) is drawn as a dashed line, so "change from baseline"
    is visible per unit rather than only in the summary.

    `trial_sort`: 'time' (default) or 'amplitude' - see `_trial_order` for why sorting trials
    by the neural response itself is not offered.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"font.size": FONT, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})

    r = load_reach_peth(cfg, subset)
    ev = np.sort(r["event_times"])
    c, z, reg, mod, pz = r["centers"], r["z"], r["region"], r["modulated"], r["peak_z"]
    ids = r["units"]
    keep = (c >= win[0]) & (c <= win[1])
    cw = c[keep]
    _, st, sc = load_units(cfg)
    order, ylab = _trial_order(cfg, ev, trial_sort)
    ev = ev[order]

    def pick(direction, k):
        """k strongest units of one direction, alternating CFA / thalamus."""
        sel = mod & ((pz > 0) if direction == "exc" else (pz <= 0))
        out = []
        pools = []
        for members in (("CFA",), THALAMUS):
            idx = np.where(sel & np.isin(reg, members))[0]
            pools.append(list(idx[np.argsort(-np.abs(pz[idx]))]))
        while len(out) < k and any(pools):
            for p in pools:
                if p and len(out) < k:
                    out.append(p.pop(0))
        return out

    blocks = [("Excited", pick("exc", n_excited)), ("Suppressed", pick("sup", n_suppressed))]
    n_rows = 2 * len(blocks)
    fig = plt.figure(figsize=(1.75 * n_cols, 2.4 * len(blocks) * 1.9))
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=[2, 1] * len(blocks),
                          hspace=0.45, wspace=0.3, left=0.045, right=0.99,
                          top=0.90, bottom=0.07)

    for b, (bname, idxs) in enumerate(blocks):
        for k, ui in enumerate(idxs[:n_cols]):
            uid = int(ids[ui])
            colour = CFA_BLUE if reg[ui] == "CFA" else THAL_GREEN
            train = st[sc == uid]

            # ---- raster ----
            ax = fig.add_subplot(gs[2 * b, k])
            for i, t0 in enumerate(ev):
                s = train[(train >= t0 + win[0]) & (train <= t0 + win[1])] - t0
                if len(s):
                    # thin, slightly transparent ticks: these units fire at 10-40 Hz and a
                    # fatter marker turns the raster into a solid block
                    ax.plot(s, np.full(len(s), i), "|", color="k", ms=1.7, mew=0.35,
                            alpha=0.85)
            ax.axvline(0, color="#D32F2F", lw=1.1, ls="--")
            ax.set_xlim(win); ax.set_ylim(-1, len(ev))
            ax.set_xticks([win[0], 0, win[1]]); ax.set_xticklabels([])
            ax.tick_params(labelsize=FONT - 2)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if k == 0:
                ax.set_ylabel(f"{bname}\n{ylab}", fontsize=FONT - 1)
            ax.set_title(f"u{uid}  {reg[ui]}\nZ={pz[ui]:+.1f}", fontsize=FONT - 1,
                         color=colour, pad=3)

            # ---- rate ----
            ax = fig.add_subplot(gs[2 * b + 1, k])
            ax.plot(cw, r["psth"][ui][keep], color=colour, lw=1.6)
            ax.plot(cw, r["null_mean"][ui][keep], color="#757575", lw=1.0, ls="--")
            ax.axvline(0, color="#D32F2F", lw=1.1, ls="--")
            ax.set_xlim(win); ax.set_xticks([win[0], 0, win[1]])
            ax.tick_params(labelsize=FONT - 2)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if k == 0:
                ax.set_ylabel("Rate (Hz)", fontsize=FONT - 1)
            if b == len(blocks) - 1:
                ax.set_xlabel("Time from\nreach onset (s)", fontsize=FONT - 1)

    fig.suptitle(f"Exemplar reach-modulated units - {cfg.session[:10]}, {len(ev)} reaches "
                 f"(blue = CFA, green = thalamus; grey dashed = baseline rate)",
                 fontsize=FONT + 2, fontweight="bold", y=0.975)
    if save:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        suffix = "" if trial_sort == "time" else f"_by{trial_sort}"
        stem = out / f"exemplar_rasters_{cfg.session[:10]}_{subset}{suffix}"
        fig.savefig(f"{stem}.png", dpi=300, facecolor="white")
        fig.savefig(f"{stem}.pdf", facecolor="white")
        print(f"saved {stem}.png and .pdf")
    return fig

