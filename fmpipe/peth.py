"""
Stage 6: reach-locked PETHs with a shuffle-calibrated null, split by depth/region.

Method (reproduces the Sept 16 'untrialized v4' cell):
  * events = bout onsets, PSTH bin 25 ms, window -2..+3 s, Gaussian smoothing (2 bins);
  * null: 200 sets of the same number of event times drawn uniformly over the recording,
    giving a per-unit, per-bin mean/SD of trial-averaged PSTHs; z = (real - null mean) / null SD;
  * a unit is modulated if |z| > Z for >= 80 ms continuously inside the response window;
  * Z is calibrated, not chosen: the lowest value on a grid (1.5..5.0) at which the same test
    applied to the shuffled PSTHs calls <= 10% of units modulated.

Speed: spikes are pre-binned at 5 ms once; each PSTH is a gather of count bins (event times are
quantised to 5 ms for the lookup, identically for real and shuffled events).

Outputs: peth/reach_peth_<subset>.npz, qc/reach_peth_<subset>.png, qc/reach_peth_depth_<subset>.png
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import SessionConfig, read_json, update_status, write_json
from .units import load_units

PETH_DEFAULTS = dict(
    win_pre=2.0, win_post=3.0,
    response_win=(-0.5, 2.0),
    bin_size=0.025, smooth_sigma_bins=2,
    min_consecutive_ms=80,
    min_baseline_hz=0.5,
    n_shuffles=200, target_fpr=0.10, seed=42,
    base_bin=0.005,
)
Z_GRID = np.arange(1.5, 5.01, 0.25)
REGION_COLORS = {"CFA": "#2196F3", "WM": "#9E9E9E", "THAL": "#14B8A6", "VAL": "#4CAF50", "other": "#E53935"}
THALAMUS = ("THAL", "VAL")          # everything >= 3000 um DV


# Per-subset parameter overrides. 'all_reaches' uses every reach (not just the first of a
# bout), so a shorter window is used because reaches follow each other quickly.
SUBSET_PARAMS = {
    "all_reaches": dict(win_pre=1.5, win_post=1.5, response_win=(-1.0, 1.0)),
}


def _events(cfg, subset):
    """-> (event times sorted, bout index of each event or None)."""
    a = cfg.analysis_dir / "peak_detection"
    if subset == "all_reaches":
        r = np.load(a / "reaches.npz")
        order = np.argsort(r["onset_t"])
        return r["onset_t"][order], r["bout_id"][order]
    b = np.load(a / "bouts.npz")
    t = b["onset_t"]
    if subset == "all":
        m = np.ones(len(t), bool)
    elif subset == "in_trial":
        m = b["trial_id"] >= 0
    elif subset == "first_in_trial":
        m = b["first_in_trial"]
    else:
        raise ValueError(f"unknown subset {subset!r}")
    return np.sort(t[m]), None


def _shuffled_events(rng, events, groups, t_lo, t_hi):
    """
    Null event times. Independent events: uniform random times. Events grouped in bouts
    (all_reaches): the bouts' internal timing is kept and only the bout start is randomised,
    so clustering of reaches is present in the null as well.
    """
    if groups is None:
        return np.sort(rng.uniform(t_lo, t_hi, size=len(events)))
    out = np.empty(len(events))
    for g in np.unique(groups):
        m = groups == g
        out[m] = events[m] - events[m].min() + rng.uniform(t_lo, t_hi)
    return np.sort(out)


class _Psth:
    """Pre-binned spike counts + fast trial-averaged PSTH for arbitrary event times."""

    def __init__(self, spike_trains, duration, P):
        self.P = P
        self.base = P["base_bin"]
        self.k = int(round(P["bin_size"] / self.base))                  # base bins per PSTH bin
        self.n_base = int(np.ceil(duration / self.base)) + 1
        self.offs = np.arange(int(round(-P["win_pre"] / self.base)), int(round(P["win_post"] / self.base)))
        self.centers = (np.arange(len(self.offs) // self.k) + 0.5) * P["bin_size"] - P["win_pre"]
        self.counts = np.zeros((len(spike_trains), self.n_base), np.uint8)
        for i, s in enumerate(spike_trains):
            c = np.bincount((s / self.base).astype(np.int64), minlength=self.n_base)[: self.n_base]
            self.counts[i] = np.minimum(c, 255)

    def mean_psth(self, events):
        idx0 = np.round(np.asarray(events) / self.base).astype(np.int64)
        tot = np.zeros((self.counts.shape[0], len(self.offs)), np.float64)
        for j in range(0, len(idx0), 32):                                # chunk to bound memory
            ix = np.clip(idx0[j:j + 32, None] + self.offs[None, :], 0, self.n_base - 1)
            tot += self.counts[:, ix].sum(axis=1)
        tot = tot[:, : (tot.shape[1] // self.k) * self.k].reshape(tot.shape[0], -1, self.k).sum(-1)
        hz = tot / (len(idx0) * self.P["bin_size"])
        return gaussian_filter1d(hz, sigma=self.P["smooth_sigma_bins"], axis=1)


def _modulated(z, resp_mask, min_bins, thr):
    """True where |z| exceeds thr for >= min_bins consecutive bins inside the response window."""
    zr = z[:, resp_mask]
    out = np.zeros(len(z), bool)
    if zr.shape[1] < min_bins:
        return out
    for arr in (zr > thr, zr < -thr):
        w = np.lib.stride_tricks.sliding_window_view(arr, min_bins, axis=1)
        out |= w.all(-1).any(-1)
    return out


def run_reach_peth(cfg: SessionConfig, subset: str = "all", **overrides) -> dict:
    P = dict(PETH_DEFAULTS); P.update(SUBSET_PARAMS.get(subset, {})); P.update(overrides)
    a = cfg.analysis_dir
    tb = read_json(a / "metadata" / "timebase.json")
    duration = tb["recording_duration_s"]
    events, groups = _events(cfg, subset)
    n_ev = len(events)
    if n_ev < 5:
        raise RuntimeError(f"Only {n_ev} events for subset '{subset}'; need >= 5.")

    u, st, sc = load_units(cfg)
    u = u[u.in_analysis & (u.rate_hz >= P["min_baseline_hz"])].copy()
    u = u.sort_values("dv_um").reset_index(drop=True)             # surface -> deep
    trains = [st[sc == i] for i in u.id.values]
    print(f"{n_ev} events ('{subset}'"
          + (f", in {len(np.unique(groups))} bouts, bout-preserving null" if groups is not None else "")
          + f"; window {-P['win_pre']}..+{P['win_post']} s, response {P['response_win']}), "
          f"{len(u)} eligible units ({u.region.value_counts().to_dict()})")

    ps = _Psth(trains, duration, P)
    centers = ps.centers
    resp = (centers >= P["response_win"][0]) & (centers < P["response_win"][1])
    min_bins = int(np.ceil(P["min_consecutive_ms"] / 1000.0 / P["bin_size"]))

    # ---- null ---------------------------------------------------------------
    rng = np.random.default_rng(P["seed"])
    t_lo, t_hi = P["win_pre"] + 1.0, duration - P["win_post"] - 1.0
    shuf = np.empty((P["n_shuffles"], len(u), len(centers)), np.float32)
    for s in range(P["n_shuffles"]):
        shuf[s] = ps.mean_psth(_shuffled_events(rng, events, groups, t_lo, t_hi))
    null_mean = shuf.mean(0)
    null_std = np.maximum(shuf.std(0), null_mean * 0.01 + 0.05)

    # ---- calibrate Z --------------------------------------------------------
    fpr = np.zeros((P["n_shuffles"], len(Z_GRID)))
    for s in range(P["n_shuffles"]):
        zs = (shuf[s] - null_mean) / null_std
        for j, thr in enumerate(Z_GRID):
            fpr[s, j] = _modulated(zs, resp, min_bins, thr).mean()
    mean_fpr = fpr.mean(0)
    ok = np.where(mean_fpr <= P["target_fpr"])[0]
    z_thr = float(Z_GRID[ok[0]] if len(ok) else Z_GRID[-1])
    fpr_at = float(mean_fpr[np.argmin(np.abs(Z_GRID - z_thr))])
    del shuf

    # ---- real ---------------------------------------------------------------
    real = ps.mean_psth(events)
    z = (real - null_mean) / null_std
    mod = _modulated(z, resp, min_bins, z_thr)
    zr = z[:, resp]
    pk = np.argmax(np.abs(zr), axis=1)
    peak_z = zr[np.arange(len(u)), pk]
    peak_lat = centers[resp][pk]
    u["modulated"] = mod
    u["peak_z"] = peak_z
    u["peak_latency_s"] = peak_lat
    u["direction"] = np.where(~mod, "none", np.where(peak_z > 0, "excited", "suppressed"))

    out = dict(units=u.id.values, z=z, psth=real, centers=centers, z_threshold=z_thr,
               fpr=fpr_at, event_times=events, null_mean=null_mean, null_std=null_std,
               modulated=mod, peak_z=peak_z, peak_latency=peak_lat,
               dv_um=u.dv_um.values, x_um=u.x_um.values, region=u.region.values.astype(str),
               response_win=np.array(P["response_win"], float))
    (a / "peth").mkdir(exist_ok=True)
    np.savez(a / "peth" / f"reach_peth_{subset}.npz", **out)
    u.to_csv(a / "peth" / f"reach_peth_{subset}_units.csv", index=False)

    summ = {}
    groups = {"CFA": ("CFA",), "WM": ("WM",), "THAL": ("THAL",), "VAL": ("VAL",), "THALAMUS": THALAMUS}
    for r, members in groups.items():
        m = u.region.isin(members).values
        summ[r] = dict(n=int(m.sum()), modulated=int((mod & m).sum()),
                       excited=int((mod & m & (peak_z > 0)).sum()), suppressed=int((mod & m & (peak_z <= 0)).sum()))
    write_json(a / "peth" / f"reach_peth_{subset}.json",
               dict(subset=subset, n_events=n_ev, z_threshold=z_thr, shuffled_fpr=fpr_at,
                    params={k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()}, regions=summ))
    print(f"Calibrated Z = {z_thr:.2f} (shuffled FPR {fpr_at:.1%});  modulated {int(mod.sum())}/{len(u)}")
    for r, s in summ.items():
        print(f"  {r:>3}: {s['modulated']:>3}/{s['n']:<3} modulated  ({s['excited']} excited, {s['suppressed']} suppressed)")
    update_status(cfg, f"peth_{subset}", True, n_events=n_ev, z_threshold=z_thr, fpr=fpr_at, regions=summ)
    out["table"] = u
    return out


def _null_stats(ps, n_ev, duration, P, rng, resp, min_bins):
    """Shuffle null for exactly `n_ev` random event times -> (null mean, null SD, calibrated Z, shuffled FPR at that Z)."""
    t_lo, t_hi = P["win_pre"] + 1.0, duration - P["win_post"] - 1.0
    shuf = np.empty((P["n_shuffles"], ps.counts.shape[0], len(ps.centers)), np.float32)
    for s in range(P["n_shuffles"]):
        shuf[s] = ps.mean_psth(np.sort(rng.uniform(t_lo, t_hi, size=n_ev)))
    null_mean = shuf.mean(0)
    null_std = np.maximum(shuf.std(0), null_mean * 0.01 + 0.05)
    fpr = np.zeros((P["n_shuffles"], len(Z_GRID)))
    for s in range(P["n_shuffles"]):
        zs = (shuf[s] - null_mean) / null_std
        for j, thr in enumerate(Z_GRID):
            fpr[s, j] = _modulated(zs, resp, min_bins, thr).mean()
    mean_fpr = fpr.mean(0)
    ok = np.where(mean_fpr <= P["target_fpr"])[0]
    thr = float(Z_GRID[ok[0]] if len(ok) else Z_GRID[-1])
    return null_mean, null_std, thr, float(mean_fpr[np.argmin(np.abs(Z_GRID - thr))])


def run_matched_peth(cfg: SessionConfig, n_events: int, subset: str = "all", n_boot: int = 200, seed: int = 7,
                     replace: bool = True, **overrides) -> dict:
    """
    PETH at a COMMON number of events so that z values are comparable between days (z grows with the number of events).
    Each of `n_boot` draws takes `n_events` events WITH replacement from the day's events (a bootstrap: it also gives the sampling
    uncertainty of the response measured with that many events); z = (PSTH - null mean) / null SD with the null built for exactly
    `n_events` random times, and the Z threshold calibrated at that N. Per unit and draw it records the net response (mean z in the
    response window), the signed peak z and whether the unit is modulated.
    """
    P = dict(PETH_DEFAULTS); P.update(SUBSET_PARAMS.get(subset, {})); P.update(overrides)
    a = cfg.analysis_dir
    duration = read_json(a / "metadata" / "timebase.json")["recording_duration_s"]
    events, _ = _events(cfg, subset)
    u, st, sc = load_units(cfg)
    u = u[u.in_analysis & (u.rate_hz >= P["min_baseline_hz"])].copy().sort_values("dv_um").reset_index(drop=True)
    ps = _Psth([st[sc == i] for i in u.id.values], duration, P)
    c = ps.centers
    resp = (c >= P["response_win"][0]) & (c < P["response_win"][1])
    min_bins = int(np.ceil(P["min_consecutive_ms"] / 1000.0 / P["bin_size"]))
    rng = np.random.default_rng(seed)
    null_mean, null_std, thr, fpr = _null_stats(ps, n_events, duration, P, rng, resp, min_bins)
    U, B = len(u), len(c)
    net = np.empty((n_boot, U), np.float32); peak = np.empty((n_boot, U), np.float32); mod = np.empty((n_boot, U), bool)
    zsum = np.zeros((U, B))
    for b in range(n_boot):
        ev = np.sort(rng.choice(events, size=n_events, replace=replace))
        z = (ps.mean_psth(ev) - null_mean) / null_std
        zr = z[:, resp]
        net[b] = zr.mean(1); pk = np.argmax(np.abs(zr), axis=1); peak[b] = zr[np.arange(U), pk]
        mod[b] = _modulated(z, resp, min_bins, thr); zsum += z
    out = dict(units=u.id.values, centers=c, z_mean=(zsum / n_boot).astype(np.float32), net=net, peak=peak, mod=mod, z_threshold=thr, fpr=fpr,
               n_events=n_events, n_events_available=len(events), dv_um=u.dv_um.values, region=u.region.values.astype(str), response_win=np.array(P["response_win"]))
    (a / "peth").mkdir(exist_ok=True)
    np.savez(a / "peth" / f"reach_peth_matched_{subset}_N{n_events}{'' if replace else '_norepl'}.npz", **out)
    print(f"{cfg.session[:10]}: matched N={n_events} of {len(events)} events ('{subset}', {'with' if replace else 'without'} replacement): Z={thr:.2f} (FPR {fpr:.1%}); "
          f"modulated in >=50% of draws: {int((mod.mean(0) >= 0.5).sum())}/{U}")
    out["table"] = u
    return out


def load_matched_peth(cfg: SessionConfig, n_events: int, subset: str = "all", replace: bool = True) -> dict:
    d = np.load(cfg.analysis_dir / "peth" / f"reach_peth_matched_{subset}_N{n_events}{'' if replace else '_norepl'}.npz")
    import pandas as pd
    out = {k: d[k] for k in d.files}
    u = pd.read_csv(cfg.analysis_dir / "units" / "units.csv").set_index("id").loc[out["units"]].reset_index()
    out["table"] = u
    return out


def load_reach_peth(cfg: SessionConfig, subset: str = "all") -> dict:
    d = np.load(cfg.analysis_dir / "peth" / f"reach_peth_{subset}.npz", allow_pickle=False)
    import pandas as pd
    r = {k: d[k] for k in d.files}
    r["table"] = pd.read_csv(cfg.analysis_dir / "peth" / f"reach_peth_{subset}_units.csv")
    return r


# --------------------------------------------------------------------- figures
def plot_reach_peth(cfg: SessionConfig, res: dict, subset: str = "all", save=True):
    """Region heatmaps (excited -> suppressed -> non-modulated, by latency/depth) + population means."""
    import matplotlib.pyplot as plt
    u, z, psth, c = res["table"], res["z"], res["psth"], res["centers"]
    rw = tuple(res.get("response_win", PETH_DEFAULTS["response_win"]))
    resp = (c >= rw[0]) & (c < rw[1])
    vmax = max(3.0, float(np.percentile(np.abs(z), 98)))
    mod, pz = res["modulated"], res["peak_z"]

    def order(mask):
        idx = np.where(mask)[0]
        exc = idx[mod[idx] & (pz[idx] > 0)]; sup = idx[mod[idx] & (pz[idx] <= 0)]; non = idx[~mod[idx]]
        lat = lambda ii: c[resp][np.argmax(np.abs(z[ii][:, resp]), axis=1)]
        exc = exc[np.argsort(lat(exc))] if len(exc) else exc
        sup = sup[np.argsort(lat(sup))] if len(sup) else sup
        return np.concatenate([exc, sup, non]), len(exc), len(sup)

    fig = plt.figure(figsize=(24, 11))
    panels = [("All regions", np.ones(len(u), bool), "k"),
              ("CFA", (u.region == "CFA").values, REGION_COLORS["CFA"]),
              ("Thalamus (>= 3000 um)", u.region.isin(THALAMUS).values, REGION_COLORS["THAL"]),
              ("VAL (>= 3500 um)", (u.region == "VAL").values, REGION_COLORS["VAL"])]
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1], hspace=0.25, wspace=0.3)
    for k, (name, mask, col) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        if mask.sum() == 0:
            ax.set_title(f"No {name} units"); continue
        idx, ne, ns = order(mask)
        im = ax.imshow(z[idx], aspect="auto", extent=[c[0], c[-1], len(idx), 0], cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.axvline(0, c="k", lw=1.5)
        if ne and ns: ax.axhline(ne, c="k", ls="--", lw=0.8, alpha=0.6)
        if 0 < ne + ns < len(idx): ax.axhline(ne + ns, c="k", lw=1.5)
        ax.set_xlabel("time from reach-bout onset (s)"); ax.set_ylabel("unit")
        ax.set_title(f"{name}: {len(idx)} units | {ne} exc + {ns} sup + {len(idx)-ne-ns} non", color=col, fontweight="bold")
        plt.colorbar(im, ax=ax, label="z", fraction=0.04, pad=0.02)
    axp = fig.add_subplot(gs[1, :])
    for r, members, col in (("CFA", ("CFA",), REGION_COLORS["CFA"]), ("Thalamus (>=3000)", THALAMUS, REGION_COLORS["THAL"]),
                            ("VAL (>=3500)", ("VAL",), REGION_COLORS["VAL"])):
        m = u.region.isin(members).values
        if m.sum():
            mu, se = psth[m].mean(0), psth[m].std(0) / np.sqrt(m.sum())
            axp.plot(c, mu, c=col, lw=2, label=f"{r} (n={m.sum()})", ls="--" if r.startswith("VAL") else "-")
            axp.fill_between(c, mu - se, mu + se, color=col, alpha=0.2)
    axp.axvline(0, c="k", ls="--"); axp.axvspan(*rw, color="gray", alpha=0.07)
    axp.set_xlabel("time from reach-bout onset (s)"); axp.set_ylabel("mean rate (Hz)"); axp.legend()
    fig.suptitle(f"{cfg.session[:10]} | reach-locked PETH ({subset}) | N={len(res['event_times'])} bouts | "
                 f"Z>{res['z_threshold']:.2f} (shuffled FPR {res['fpr']:.1%})", y=0.995)
    if save:
        fig.savefig(cfg.analysis_dir / "qc" / f"reach_peth_{subset}.png", dpi=130, bbox_inches="tight")
    return fig


THEMES = {
    "dark": dict(bg="#0b0f14", fg="#e6edf3", dim="#8b949e", nonmod="#6b7280", site="#ffffff", band_alpha=0.10,
                 exc="#ff7a2e", sup="#4da3ff", zero="#ffffff",
                 cmap=("dark_div", ["#4da3ff", "#1f4c9a", "#141a22", "#9a3412", "#ff7a2e"])),
    "light": dict(bg="#ffffff", fg="#111827", dim="#6b7280", nonmod="#b8bec7", site="#000000", band_alpha=0.10,
                  exc="#c62828", sup="#1565c0", zero="#000000", cmap=None),
}
REGION_BAND_COLORS = {"CFA": "#3b82f6", "WM": "#9ca3af", "THAL": "#14b8a6", "VAL": "#22c55e"}


def _theme_cmap(th):
    from matplotlib.colors import LinearSegmentedColormap
    if th["cmap"] is None:
        return plt_cmap("RdBu_r")
    return LinearSegmentedColormap.from_list(*th["cmap"])


def plt_cmap(name):
    import matplotlib.pyplot as plt
    return plt.get_cmap(name)


def plot_probe_modulation(cfg: SessionConfig, res: dict, subset: str = "all", theme: str = "dark",
                          vlim: float = 10.0, save=True, label=None):
    """
    Reach-locked modulation along the probe, drawn only from the data:
      probe map    each unit at its actual (x, depth); colour = signed peak z, size = |peak z|; no outlines
      depth PETH   every unit's z-scored PETH as a row at its true depth (blank where no unit was recorded)
      by depth     fraction of units excited / suppressed in 100 um bins
      latency      peak latency of each modulated unit against depth
    CFA / WM / VAL are marked by faint bands only. Colour limits are +-vlim for every day (values beyond are clipped
    in colour only, the colourbar says so).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    th = THEMES[theme]
    cmap = _theme_cmap(th)
    u, z, c = res["table"], res["z"], res["centers"]
    dv, xu = u.dv_um.values, u.x_um.values
    mod, pz, lat = res["modulated"], res["peak_z"], res["peak_latency"]
    rw = tuple(res.get("response_win", PETH_DEFAULTS["response_win"]))
    norm = TwoSlopeNorm(0, -vlim, vlim)
    lay = np.load(cfg.analysis_dir / "metadata" / "channel_layout.npz")
    site_dv = cfg.tip_dv_um - lay["y"]; site_x = lay["x"]
    lo, hi = float(site_dv.min()) - 50, float(site_dv.max()) + 50
    ev_name = {"all": "reach-bout onsets", "all_reaches": "every reach"}.get(subset, subset)

    plt.rcParams.update({"text.color": th["fg"], "axes.labelcolor": th["fg"], "xtick.color": th["dim"],
                         "ytick.color": th["dim"], "axes.edgecolor": th["dim"]})
    fig = plt.figure(figsize=(18, 11.5), facecolor=th["bg"])
    gs = fig.add_gridspec(1, 4, width_ratios=[0.85, 2.3, 1.0, 1.25], wspace=0.07, left=0.06, right=0.985, top=0.86, bottom=0.16)
    ax = [fig.add_subplot(gs[0, k]) for k in range(4)]
    for a in ax:
        a.set_facecolor(th["bg"])
        for name, (b0, b1) in cfg.region_bounds_dv_um.items():
            a.axhspan(b0, b1, color=REGION_BAND_COLORS.get(name, "#888"), alpha=th["band_alpha"], lw=0, zorder=0)
        a.set_ylim(hi, lo)
        for s in a.spines.values():
            s.set_alpha(0.4)
    for a in ax[1:]:
        a.tick_params(labelleft=False)

    # ---- A: probe map ----
    a = ax[0]
    a.scatter(site_x, site_dv, s=2.5, color=th["site"], alpha=0.10, lw=0, zorder=1)
    nm = ~mod
    a.scatter(xu[nm], dv[nm], s=16, color=th["nonmod"], alpha=0.65, lw=0, zorder=2)
    order = np.argsort(np.abs(pz[mod]))
    ii = np.where(mod)[0][order]
    a.scatter(xu[ii], dv[ii], s=22 + 15 * np.minimum(np.abs(pz[ii]), vlim), c=pz[ii], cmap=cmap, norm=norm,
              alpha=0.95, lw=0, zorder=3)
    a.set_xlim(0, 70); a.set_xlabel("x on probe (um)"); a.set_ylabel("DV depth below surface (um)", fontsize=11)
    a.set_title("units on the probe", color=th["fg"], fontsize=12, pad=8)
    for name, (b0, b1) in cfg.region_bounds_dv_um.items():
        m = (u.region == name).values
        if m.sum() == 0:
            continue
        a.text(0.04, max(b0, lo) + 18, f"{name}   {int((mod & m).sum())}/{int(m.sum())} modulated",
               transform=a.get_yaxis_transform(), va="top", ha="left", fontsize=10.5,
               fontweight="bold", color=REGION_BAND_COLORS.get(name, th["fg"]), zorder=6)

    # ---- B: depth-resolved PETH ----
    a = ax[1]
    dt = c[1] - c[0]
    edges = np.append(c - dt / 2, c[-1] + dt / 2)
    srt = np.argsort(dv); d_sorted = dv[srt]
    cap = 20.0                    # a unit's row spans up to half the gap to its neighbours, at most 20 um; larger gaps stay blank
    for rank, i in enumerate(srt):
        up = (d_sorted[rank] - d_sorted[rank - 1]) / 2 if rank > 0 else cap
        dn = (d_sorted[rank + 1] - d_sorted[rank]) / 2 if rank < len(srt) - 1 else cap
        a.pcolormesh(edges, [dv[i] - min(up, cap), dv[i] + min(dn, cap)], z[i][None, :], cmap=cmap, norm=norm,
                     shading="flat", rasterized=True, zorder=2)
    a.axvline(0, color=th["zero"], lw=1.1, alpha=0.9, zorder=4)
    a.axvspan(*rw, color=th["fg"], alpha=0.04, zorder=1)
    a.set_xlim(edges[0], edges[-1]); a.set_xlabel("time from reach onset (s)")
    a.set_title("z-scored PETH of every unit, at its depth", color=th["fg"], fontsize=12, pad=8)

    # ---- C: fraction modulated by depth ----
    a = ax[2]
    bins = np.arange(np.floor(lo / 100) * 100, hi + 100, 100)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (dv >= b0) & (dv < b1)
        if m.sum() == 0:
            continue
        fe = (mod & m & (pz > 0)).sum() / m.sum(); fs = (mod & m & (pz <= 0)).sum() / m.sum()
        a.barh((b0 + b1) / 2, fe, height=88, color=th["exc"], alpha=0.9, lw=0, zorder=3)
        a.barh((b0 + b1) / 2, -fs, height=88, color=th["sup"], alpha=0.9, lw=0, zorder=3)
        a.text(1.04, (b0 + b1) / 2, f"{int(m.sum())}", fontsize=7.5, va="center", color=th["dim"])
    a.axvline(0, color=th["dim"], lw=0.8, zorder=4); a.set_xlim(-1, 1.12)
    a.set_xlabel("fraction of units\n<- suppressed | excited ->")
    a.set_title("modulated vs depth (n per 100 um)", color=th["fg"], fontsize=12, pad=8)

    # ---- D: latency vs depth ----
    a = ax[3]
    a.axvline(0, color=th["zero"], lw=1.0, alpha=0.8, zorder=2)
    a.scatter(lat[ii], dv[ii], s=22 + 15 * np.minimum(np.abs(pz[ii]), vlim), c=pz[ii], cmap=cmap, norm=norm, alpha=0.95, lw=0, zorder=3)
    a.set_xlim(rw[0] - 0.05, rw[1] + 0.05); a.set_xlabel("latency of peak response (s)")
    a.set_title("when each modulated unit responds", color=th["fg"], fontsize=12, pad=8)

    # ---- colour bar + size key ----
    cax = fig.add_axes([0.14, 0.045, 0.26, 0.016])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label(f"z vs shuffled   (colour limits +-{vlim:g}; larger values clipped in colour only)", color=th["fg"], fontsize=9)
    cb.ax.tick_params(colors=th["dim"], labelsize=8); cb.outline.set_edgecolor(th["dim"])
    kax = fig.add_axes([0.62, 0.012, 0.30, 0.06]); kax.set_facecolor(th["bg"]); kax.axis("off")
    for k, zz in enumerate((3, 6, 10)):
        kax.scatter([k * 0.9], [0.5], s=22 + 15 * zz, color=th["dim"], lw=0)
        kax.text(k * 0.9, 0.05, f"|z|={zz}", ha="center", va="top", fontsize=8, color=th["dim"])
    kax.scatter([2.9], [0.5], s=16, color=th["nonmod"], alpha=0.65, lw=0)
    kax.text(2.9, 0.05, "not modulated", ha="center", va="top", fontsize=8, color=th["dim"])
    kax.set_xlim(-0.5, 3.5); kax.set_ylim(-0.4, 1)

    th_m = u.region.isin(THALAMUS).values
    tot = (f"{int(mod.sum())}/{len(mod)} modulated ({int((mod & (pz > 0)).sum())} excited, {int((mod & (pz <= 0)).sum())} suppressed)"
           f"   |   thalamus (>=3000 um): {int((mod & th_m).sum())}/{int(th_m.sum())}")
    fig.text(0.06, 0.955, f"Mouse 4103  |  {cfg.session[:10]}  |  reach-locked modulation along the probe", fontsize=18,
             fontweight="bold", color=th["fg"], va="center")
    fig.text(0.06, 0.915, f"{len(res['event_times'])} {label or ev_name}   |   shuffle-calibrated Z > {res['z_threshold']:.2f} "
             f"(shuffled false-positive rate {res['fpr']:.1%})   |   {tot}", fontsize=11.5, color=th["dim"], va="center")
    if save:
        fig.savefig(cfg.analysis_dir / "qc" / f"probe_modulation_{subset}_{theme}.png", dpi=170, facecolor=th["bg"])
    plt.rcdefaults()
    return fig


def plot_stacked_regions(cfg: SessionConfig, res: dict, subset: str = "all", n_nonmod=10,
                         regions=("CFA", "THAL", "VAL"), save=True):
    """
    One continuous heatmap: CFA units on top, a thin divider, VAL units below. Within each
    region: excited units by peak latency, then suppressed units by peak latency, then
    `n_nonmod` representative non-modulated units (evenly spaced in depth), so each region
    shows its own latency 'diagonal'.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    u, z, c = res["table"], res["z"], res["centers"]
    mod, pz = res["modulated"], res["peak_z"]
    rw = tuple(res.get("response_win", PETH_DEFAULTS["response_win"]))
    resp = (c >= rw[0]) & (c < rw[1])
    vmax = max(3.0, float(np.percentile(np.abs(z), 98)))
    lat = lambda ii: c[resp][np.argmax(np.abs(z[ii][:, resp]), axis=1)]

    rows, blocks = [], []          # blocks: (region, kind, start_row, n)
    for reg in regions:
        idx = np.where((u.region == reg).values)[0]
        exc = idx[mod[idx] & (pz[idx] > 0)]
        sup = idx[mod[idx] & (pz[idx] <= 0)]
        non = idx[~mod[idx]]
        exc = exc[np.argsort(lat(exc))] if len(exc) else exc
        sup = sup[np.argsort(lat(sup))] if len(sup) else sup
        non_sorted = non[np.argsort(u.dv_um.values[non])] if len(non) else non
        if len(non_sorted) > n_nonmod:
            non_sorted = non_sorted[np.unique(np.linspace(0, len(non_sorted) - 1, n_nonmod).round().astype(int))]
        for kind, ii in (("excited", exc), ("suppressed", sup), ("non", non_sorted)):
            blocks.append((reg, kind, len(rows), len(ii), len(non) if kind == "non" else len(ii)))
            rows.extend(ii.tolist())
    rows = np.array(rows, int)
    # vertical positions: a small gap between the two regions for the divider
    gap = 0.8
    ypos, y, last_reg = [], 0.0, None
    for i in rows:
        reg = u.region.values[i]
        if last_reg is not None and reg != last_reg:
            y += gap
        ypos.append(y); y += 1.0; last_reg = reg
    ypos = np.array(ypos)

    fig, ax = plt.subplots(figsize=(9, 13))
    dt = c[1] - c[0]
    edges = np.append(c - dt / 2, c[-1] + dt / 2)
    for r_i, i in enumerate(rows):
        ax.pcolormesh(edges, [ypos[r_i], ypos[r_i] + 1.0], z[i][None, :], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      shading="flat", rasterized=True)
    ax.axvline(0, c="k", lw=1.2)
    ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(y, 0)
    for reg, kind, start, n, n_total in blocks:
        if n == 0:
            continue
        y0 = ypos[start]; y1 = ypos[start + n - 1] + 1.0
        col = REGION_COLORS[reg]
        ax.add_patch(mp.Rectangle((edges[0] - 0.045 * (edges[-1] - edges[0]), y0), 0.03 * (edges[-1] - edges[0]), y1 - y0,
                                  color=col, clip_on=False))
        label = {"excited": f"exc ({n})", "suppressed": f"sup ({n})",
                 "non": f"non-mod ({n} of {n_total})"}[kind]
        ax.text(edges[-1] + 0.02 * (edges[-1] - edges[0]), (y0 + y1) / 2, label, va="center", fontsize=8,
                color="darkred" if kind == "excited" else "darkblue" if kind == "suppressed" else "gray")
        if kind != "excited" or start > 0:
            if start > 0 and u.region.values[rows[start]] == u.region.values[rows[start - 1]]:
                ax.axhline(y0 - 0.0, c="k", lw=0.6, ls="--" if kind == "suppressed" else "-", alpha=0.7)
    # thin divider between regions
    for k in range(1, len(rows)):
        if u.region.values[rows[k]] != u.region.values[rows[k - 1]]:
            yd = (ypos[k - 1] + 1.0 + ypos[k]) / 2
            ax.axhline(yd, c="k", lw=2.0)
            nxt = u.region.values[rows[k]]
            tag = {"THAL": "thalamus 3000-3500 um", "VAL": "VAL >= 3500 um (thalamus)"}.get(nxt, nxt)
            ax.text(edges[-1] + 0.02 * (edges[-1] - edges[0]), yd, f"▼ {tag} below",
                    fontsize=9, fontweight="bold", va="center", color=REGION_COLORS[u.region.values[rows[k]]])
    for reg in regions:                      # region names, rotated, inside the colour bars
        ii = [r_i for r_i, i in enumerate(rows) if u.region.values[i] == reg]
        if ii:
            ax.text(edges[0] - 0.03 * (edges[-1] - edges[0]), (ypos[ii[0]] + ypos[ii[-1]] + 1) / 2,
                    f"{reg}  (n={len(ii)} shown, {int((u.region == reg).sum())} total)", rotation=90, va="center", ha="center",
                    color="white", fontsize=10, fontweight="bold")
    ax.axvspan(*rw, color="k", alpha=0.03)
    ax.set_yticks([]); ax.set_xlabel("time from reach onset (s)")
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap="RdBu_r")
    plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.26, label="z vs shuffled")
    ax.set_title(f"{cfg.session[:10]} | {subset}: {len(res['event_times'])} events, Z>{res['z_threshold']:.2f}\n"
                 f"latency-sorted: excited, then suppressed, then {n_nonmod} representative non-modulated units")
    plt.tight_layout()
    if save:
        fig.savefig(cfg.analysis_dir / "qc" / f"reach_peth_stacked_{subset}.png", dpi=130, bbox_inches="tight")
    return fig


def compare_peth(cfg: SessionConfig, a_name: str = "all", b_name: str = "all_reaches", save=True):
    """Robustness check: same units analysed with two event definitions (e.g. bout onsets vs every reach)."""
    import matplotlib.pyplot as plt
    A, B = load_reach_peth(cfg, a_name), load_reach_peth(cfg, b_name)
    ta, tb_ = A["table"].set_index("id"), B["table"].set_index("id")
    ids = ta.index.intersection(tb_.index)
    ta, tb_ = ta.loc[ids], tb_.loc[ids]
    ma, mb = ta.modulated.values, tb_.modulated.values
    both, only_a, only_b = int((ma & mb).sum()), int((ma & ~mb).sum()), int((~ma & mb).sum())
    neither = int((~ma & ~mb).sum())
    same_dir = int(((ta.peak_z.values > 0) == (tb_.peak_z.values > 0))[ma & mb].sum())
    r = float(np.corrcoef(ta.peak_z.values, tb_.peak_z.values)[0, 1])
    print(f"{a_name} ({len(A['event_times'])} events, Z>{A['z_threshold']:.2f}) vs "
          f"{b_name} ({len(B['event_times'])} events, Z>{B['z_threshold']:.2f}) on {len(ids)} units:")
    print(f"  modulated in both {both}, only {a_name} {only_a}, only {b_name} {only_b}, neither {neither};  "
          f"same direction in {same_dir}/{both} of the shared ones;  peak-z correlation r={r:.2f}")
    fig, ax = plt.subplots(figsize=(6, 6))
    for reg in ("CFA", "WM", "THAL", "VAL"):
        m = (ta.region == reg).values
        ax.scatter(ta.peak_z[m], tb_.peak_z[m], s=22, color=REGION_COLORS[reg], alpha=0.8, label=reg)
    lim = float(np.nanpercentile(np.abs(np.r_[ta.peak_z, tb_.peak_z]), 99)) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7); ax.axhline(0, c="0.8", lw=0.6); ax.axvline(0, c="0.8", lw=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"peak z, {a_name}"); ax.set_ylabel(f"peak z, {b_name}")
    ax.set_title(f"{cfg.session[:10]}: r={r:.2f}; modulated both {both} / only {a_name} {only_a} / only {b_name} {only_b}", fontsize=9)
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(cfg.analysis_dir / "qc" / f"reach_peth_compare_{a_name}_vs_{b_name}.png", dpi=130)
    return dict(both=both, only_a=only_a, only_b=only_b, neither=neither, same_direction=same_dir, r=r)


def plot_peth_depth(cfg: SessionConfig, res: dict, subset: str = "all", save=True, max_band_um=14.0):
    """
    Each unit's reach PETH drawn at its true depth on the probe.
      A  z-scored PETH, one band per unit at its DV depth (blank = no unit there)
      B  where on the probe the units sit, coloured by response (red excited / blue suppressed)
      C  fraction of units modulated vs depth (excited right, suppressed left)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    u, z, c = res["table"], res["z"], res["centers"]
    dv = u.dv_um.values
    order = np.argsort(dv)
    rw = tuple(res.get("response_win", PETH_DEFAULTS["response_win"]))
    vmax = max(3.0, float(np.percentile(np.abs(z), 98)))
    norm = TwoSlopeNorm(0, -vmax, vmax)
    lo, hi = float(dv.min()) - 60, float(dv.max()) + 60

    fig, ax = plt.subplots(1, 3, figsize=(17, 10), sharey=True, gridspec_kw=dict(width_ratios=[2.2, 1.0, 1.0]))
    for name, (b0, b1) in cfg.region_bounds_dv_um.items():
        for a_ in ax:
            a_.axhspan(b0, b1, color=REGION_COLORS.get(name, "0.8"), alpha=0.08, lw=0)
        ax[0].text(c[0] + 0.03, max(b0, lo) + 25, name, fontsize=11, fontweight="bold", color=REGION_COLORS.get(name, "k"), va="top")
    # A: bands
    edges = np.append(c - (c[1] - c[0]) / 2, c[-1] + (c[1] - c[0]) / 2)
    d_sorted = dv[order]
    for rank, i in enumerate(order):
        up = (d_sorted[rank] - d_sorted[rank - 1]) / 2 if rank > 0 else max_band_um
        dn = (d_sorted[rank + 1] - d_sorted[rank]) / 2 if rank < len(order) - 1 else max_band_um
        y0, y1 = dv[i] - min(up, max_band_um), dv[i] + min(dn, max_band_um)
        ax[0].pcolormesh(edges, [y0, y1], z[i][None, :], cmap="RdBu_r", norm=norm, shading="flat", rasterized=True)
    ax[0].axvline(0, c="k", lw=1.2)
    ax[0].axvspan(*rw, color="k", alpha=0.04)
    ax[0].set_xlim(edges[0], edges[-1]); ax[0].set_ylim(hi, lo)
    ax[0].set_xlabel("time from reach-bout onset (s)"); ax[0].set_ylabel("DV depth below surface (um)")
    ax[0].set_title("A  z-scored reach PETH of each unit at its depth")
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"); plt.colorbar(sm, ax=ax[0], fraction=0.03, pad=0.01, label="z vs shuffled")

    # B: probe schematic
    mod, pz = res["modulated"], res["peak_z"]
    ax[1].scatter(u.x_um[~mod], dv[~mod], s=14, c="0.7", label=f"not modulated ({(~mod).sum()})", zorder=2)
    sc = ax[1].scatter(u.x_um[mod], dv[mod], s=30 + 22 * np.abs(pz[mod]), c=pz[mod], cmap="RdBu_r", norm=norm,
                       edgecolors="k", linewidths=0.4, zorder=3)
    ax[1].set_xlabel("x on probe (um)"); ax[1].set_title("B  units on the probe\n(colour = peak z, size = |z|)")
    ax[1].legend(fontsize=8, loc="lower right")

    # C: modulated fraction by depth
    bins = np.arange(np.floor(lo / 100) * 100, hi + 100, 100)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (dv >= b0) & (dv < b1)
        if m.sum() == 0: continue
        fe = (mod & m & (pz > 0)).sum() / m.sum(); fs = (mod & m & (pz <= 0)).sum() / m.sum()
        ax[2].barh((b0 + b1) / 2, fe, height=90, color="#c62828")
        ax[2].barh((b0 + b1) / 2, -fs, height=90, color="#1565c0")
        ax[2].text(1.02, (b0 + b1) / 2, f"n={m.sum()}", fontsize=7, va="center")
    ax[2].axvline(0, c="k", lw=0.8); ax[2].set_xlim(-1, 1.25)
    ax[2].set_xlabel("fraction of units  <- suppressed | excited ->"); ax[2].set_title("C  modulated fraction vs depth (100 um)")
    fig.suptitle(f"{cfg.session[:10]}: reach-locked responses along the probe  |  N={len(res['event_times'])} bouts, "
                 f"Z>{res['z_threshold']:.2f}", y=0.995)
    plt.tight_layout()
    if save:
        fig.savefig(cfg.analysis_dir / "qc" / f"reach_peth_depth_{subset}.png", dpi=130, bbox_inches="tight")
    return fig
