"""
Stage 7: paw kinematics and their relation to firing.

Stage 6 asks *whether* a unit is modulated around a reach. This stage asks *how the firing
relates to the movement itself*: the paw's trajectory in camera pixels, its speed, and the
coupling between each unit's rate and that speed.

Three products, all built from outputs that already exist (dlc_alignment/, units/,
peak_detection/, peth/) -- nothing here re-runs stages 0-6 or touches region labels:

  * reach-aligned trajectories: x, y and speed of a marker on a common grid around bout
    (or reach) onsets, so the movement can be plotted next to the PETH on one time axis;
  * instantaneous firing rate on the camera's own frame grid, so rate and kinematics are
    sampled identically and a trajectory can be coloured by a unit's rate;
  * speed coupling: Pearson r between a unit's rate and paw speed as a function of lag,
    with a circular-shift null, giving a best lag (negative = the unit leads the paw).

Tracking caveat that drives the design: the paw is only in view during reaching, so DLC
likelihood clears 0.5 on roughly half the frames. Everything below is computed on
confidently tracked frames only; x/y are smoothed and differentiated WITHIN contiguous
tracked segments, never across a dropout, and the fraction of frames used is reported.
The circular-shift null is computed under the same validity mask, so the (highly
non-random) time structure of tracking is present in the null as well.

Outputs: kinematics/reach_kinematics_<subset>.npz, kinematics/speed_coupling.csv,
         kinematics/kinematics_meta.json, qc/kinematics_<subset>.png
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binomtest

from .config import SessionConfig, read_json, update_status, write_json
from .dlc import load_dlc
from .peth import PETH_DEFAULTS, THALAMUS, REGION_COLORS
from .units import load_units

KIN_DEFAULTS = dict(
    marker="paw_center",
    likelihood_min=0.5,        # same gate stage 5 uses to call a frame tracked
    max_jump_px=15.0,          # a confident frame this far from the previous one is a DLC
                               # mis-detection, not motion (15 px at 200 Hz = 3000 px/s)
    min_segment_frames=10,     # a tracked run shorter than this is dropped (50 ms at 200 Hz)
    smooth_ms=15.0,            # Gaussian on x/y before differentiating, within a segment
    win_pre=0.5, win_post=1.0,  # trajectory window around each event
    rate_smooth_ms=25.0,       # Gaussian on spike counts binned at the frame rate
    coupling_bin_s=0.020,      # rate/speed are decimated to this bin for the lag scan
    lag_max_s=0.30,            # lag scan runs -lag_max..+lag_max
    n_shifts=200, seed=11,     # circular-shift / permutation null for the coupling
    min_baseline_hz=PETH_DEFAULTS["min_baseline_hz"],
    # the trial-wise test correlates one unit's rate against speed event by event; a unit
    # that fires a handful of spikes across all events cannot support a correlation, and
    # produces large unstable r values, so it is left out of that test (not of the PETH).
    min_spikes_per_event=2.0, min_spikes_window=150,
    perm_batch=25,             # permutations evaluated per matrix op (memory/speed trade)
)


def _params(overrides):
    P = dict(KIN_DEFAULTS)
    P.update(overrides)
    return P


def _runs(mask):
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(padded)
    return np.where(d == 1)[0], np.where(d == -1)[0]        # [start, stop)


# --------------------------------------------------------------------- kinematics
def kinematics(cfg: SessionConfig, view: str | None = None, **overrides) -> dict:
    """
    Marker position and speed on the camera frame grid.

    x/y are NaN wherever the marker is not confidently tracked. Within each tracked
    segment they are Gaussian-smoothed and differentiated with the real frame times;
    speed is in pixels/s. No smoothing or differentiation crosses a dropout.
    """
    P = _params(overrides)
    d = load_dlc(cfg, view)
    t = d["canonical_time"].astype(float)
    bp = P["marker"]
    if f"{bp}_x" not in d:
        raise KeyError(f"marker {bp!r} not in the DLC file (have: "
                       f"{sorted(k[:-2] for k in d if k.endswith('_x'))})")
    x_raw, y_raw = d[f"{bp}_x"].astype(float), d[f"{bp}_y"].astype(float)
    tracked = d[f"{bp}_likelihood"] >= P["likelihood_min"]

    # DLC occasionally places the marker confidently on the wrong thing, which shows up as a
    # single-frame jump of tens of pixels (hundreds of times faster than a real reach). Those
    # frames are dropped, which also splits the tracked run so nothing is smoothed or
    # differentiated across the jump.
    n_jump = 0
    if P["max_jump_px"]:
        step = np.hypot(np.diff(x_raw), np.diff(y_raw))
        bad = np.zeros(len(t), bool)
        bad[1:] = tracked[:-1] & tracked[1:] & (step > P["max_jump_px"])
        n_jump = int(bad.sum())
        tracked = tracked & ~bad

    dt = float(np.median(np.diff(t)))
    sigma = (P["smooth_ms"] / 1000.0) / dt
    x = np.full(len(t), np.nan)
    y = np.full(len(t), np.nan)
    speed = np.full(len(t), np.nan)
    seg_id = np.full(len(t), -1, np.int32)      # which tracked segment a frame belongs to

    starts, stops = _runs(tracked)
    keep = (stops - starts) >= P["min_segment_frames"]
    starts, stops = starts[keep], stops[keep]
    for s, (a, b) in enumerate(zip(starts, stops)):
        xs = gaussian_filter1d(x_raw[a:b], sigma, mode="nearest")
        ys = gaussian_filter1d(y_raw[a:b], sigma, mode="nearest")
        x[a:b], y[a:b] = xs, ys
        speed[a:b] = np.hypot(np.gradient(xs, t[a:b]), np.gradient(ys, t[a:b]))
        seg_id[a:b] = s

    valid = np.isfinite(speed)
    return dict(t=t, x=x, y=y, speed=speed, valid=valid, seg_id=seg_id, dt=dt, marker=bp,
                n_segments=int(len(starts)), frac_tracked=float(tracked.mean()),
                frac_valid=float(valid.mean()), n_jump_frames=n_jump)


def frame_rate(train: np.ndarray, t: np.ndarray, dt: float, smooth_ms: float) -> np.ndarray:
    """One unit's firing rate (Hz) on the camera frame grid, Gaussian-smoothed."""
    edges = np.concatenate([t - dt / 2, [t[-1] + dt / 2]])
    counts = np.histogram(train, edges)[0] / dt
    return gaussian_filter1d(counts, (smooth_ms / 1000.0) / dt)


def _events(cfg, subset):
    a = cfg.analysis_dir / "peak_detection"
    if subset == "all_reaches":
        r = np.load(a / "reaches.npz")
        return np.sort(r["onset_t"])
    b = np.load(a / "bouts.npz")
    t = b["onset_t"]
    m = (b["trial_id"] >= 0) if subset == "in_trial" else \
        b["first_in_trial"] if subset == "first_in_trial" else np.ones(len(t), bool)
    return np.sort(t[m])


def align(trace: np.ndarray, t: np.ndarray, events, P) -> tuple:
    """
    -> (n_events, n_samples) of `trace` sampled at the frame nearest each event + grid offset,
    and the grid. NaNs in `trace` (untracked frames) are preserved. Events whose window
    does not fit inside the video are dropped.
    """
    dt = float(np.median(np.diff(t)))
    grid = np.arange(-P["win_pre"], P["win_post"], dt)
    ev = np.asarray(events, float)
    ev = ev[(ev - P["win_pre"] >= t[0]) & (ev + P["win_post"] <= t[-1])]
    return gather(trace, t, ev, grid), grid, ev


def gather(trace: np.ndarray, t: np.ndarray, ev, grid, shift: float = 0.0) -> np.ndarray:
    """`trace` at (event + shift + grid) for a FIXED event list, no dropping (indices clipped)."""
    idx = np.searchsorted(t, np.asarray(ev, float)[:, None] + shift + grid[None, :])
    return trace[np.clip(idx, 0, len(t) - 1)]


# ------------------------------------------------------------------ speed coupling
def _decimate(v, k, how="mean"):
    import warnings
    n = (len(v) // k) * k
    r = v[:n].reshape(-1, k)
    if how != "mean":
        return r.all(axis=1)
    with warnings.catch_warnings():           # all-NaN bins are expected (untracked)
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(r, axis=1)


def speed_coupling(cfg: SessionConfig, kin: dict, units=None, **overrides):
    """
    Pearson r between each unit's rate and paw speed as a function of lag, on confidently
    tracked bins only. r(lag) correlates rate at time i with speed at time i+lag, so a
    NEGATIVE best lag means the unit's rate leads the paw.

    The null is `n_shifts` circular shifts of the rate against the same speed and the same
    validity mask: it preserves each unit's rate autocorrelation and the time structure of
    tracking, which a shuffle of bins would destroy. z = (max|r| - mean null max|r|) / SD.
    """
    P = _params(overrides)
    u, st, sc = load_units(cfg)
    u = u[u.in_analysis & (u.rate_hz >= P["min_baseline_hz"])].copy()
    if units is not None:
        u = u[u.id.isin(np.asarray(units))].copy()
    u = u.sort_values("dv_um").reset_index(drop=True)

    t, dt = kin["t"], kin["dt"]
    k = max(1, int(round(P["coupling_bin_s"] / dt)))
    speed = _decimate(kin["speed"], k)
    valid = np.isfinite(speed)
    nb = len(speed)

    R = np.empty((len(u), nb), np.float32)
    for i, cid in enumerate(u.id.values):
        R[i] = _decimate(frame_rate(st[sc == cid], t, dt, P["rate_smooth_ms"]), k)

    max_lag = int(round(P["lag_max_s"] / P["coupling_bin_s"]))
    lags = np.arange(-max_lag, max_lag + 1)

    def r_at(shift):
        """Pearson r of every unit's rate with speed rolled by `shift` bins."""
        s = np.roll(speed, -shift)
        m = valid & np.roll(valid, -shift)
        sm = s[m]
        sm = sm - sm.mean()
        Rm = R[:, m].astype(np.float64)
        Rm -= Rm.mean(axis=1, keepdims=True)
        denom = np.sqrt((Rm ** 2).sum(axis=1) * (sm ** 2).sum())
        return np.divide(Rm @ sm, denom, out=np.zeros(len(Rm)), where=denom > 0)

    r_lag = np.column_stack([r_at(int(s)) for s in lags])            # (n_units, n_lags)
    best = np.argmax(np.abs(r_lag), axis=1)
    peak_r = r_lag[np.arange(len(u)), best]
    best_lag = lags[best] * P["coupling_bin_s"]

    rng = np.random.default_rng(P["seed"])
    lo = max(2 * max_lag, int(round(5.0 / P["coupling_bin_s"])))      # keep shifts well clear of 0
    shifts = rng.integers(lo, nb - lo, size=P["n_shifts"])
    null = np.column_stack([r_at(int(s)) for s in shifts])
    null_abs = np.abs(null)
    z = (np.abs(peak_r) - null_abs.mean(axis=1)) / np.maximum(null_abs.std(axis=1), 1e-9)
    p = (null_abs >= np.abs(peak_r)[:, None]).mean(axis=1)

    u["peak_r"] = peak_r
    u["best_lag_s"] = best_lag
    u["coupling_z"] = z
    u["coupling_p"] = p
    u["coupled"] = p < 0.05
    return dict(table=u, lags=lags * P["coupling_bin_s"], r_lag=r_lag,
                n_bins_used=int(valid.sum()), n_bins=nb, bin_s=P["coupling_bin_s"],
                null_abs_mean=null_abs.mean(axis=1), n_shifts=P["n_shifts"])


# ------------------------------------------------- trial-wise (residual) coupling
def _masked_center(A, M):
    """Centre A on its across-event mean per time bin, using only entries in M; 0 outside M."""
    n = np.maximum(M.sum(axis=-2), 1)
    mean = np.where(M, A, 0.0).sum(axis=-2) / n
    return np.where(M, A - mean[..., None, :], 0.0)


def trialwise_coupling(cfg: SessionConfig, kin: dict, events, units=None, **overrides):
    """
    Does firing track HOW the paw moved on a given reach, beyond the stereotyped reach?

    The session-wide `speed_coupling` cannot answer this: the paw is only tracked while it
    reaches, so a unit that simply fires more during reaches correlates with speed. Here the
    across-event mean is removed from both speed and rate at every time bin, leaving only
    trial-to-trial deviations, and the two residuals are correlated across all (event, bin)
    pairs with a confidently tracked speed.

    Null: 200 permutations of which event's rate residual pairs with which event's speed
    residual, taking the max |r| over the same lag grid, so the best-lag search is included
    in the null rather than being free.
    """
    P = _params(overrides)
    u, st, sc = load_units(cfg)
    u = u[u.in_analysis & (u.rate_hz >= P["min_baseline_hz"])].copy()
    if units is not None:
        u = u[u.id.isin(np.asarray(units))].copy()
    u = u.sort_values("dv_um").reset_index(drop=True)

    t, dt = kin["t"], kin["dt"]
    S, grid, ev = align(kin["speed"], t, events, P)
    M = np.isfinite(S)
    Sc = _masked_center(S, M)
    s_ss = float((Sc ** 2).sum())

    # drop units with too few spikes in the analysis window to support a correlation
    lo, hi = ev + grid[0], ev + grid[-1]
    n_sp = np.array([int((np.searchsorted(st[sc == cid], hi)
                          - np.searchsorted(st[sc == cid], lo)).sum()) for cid in u.id.values])
    enough = (n_sp >= P["min_spikes_window"]) & (n_sp / max(len(ev), 1) >= P["min_spikes_per_event"])
    u = u[enough].copy().reset_index(drop=True)
    n_sp = n_sp[enough]
    u["tw_n_spikes_window"] = n_sp

    max_lag = int(round(P["lag_max_s"] / P["coupling_bin_s"]))
    lags = np.arange(-max_lag, max_lag + 1) * P["coupling_bin_s"]
    rng = np.random.default_rng(P["seed"])
    perms = np.array([rng.permutation(len(ev)) for _ in range(P["n_shifts"])])

    def corr(Rc):
        """r of each (..., E, T) centred rate block against the centred speed."""
        dn = np.sqrt((Rc ** 2).sum(axis=(-2, -1)) * s_ss)
        return np.divide((Rc * Sc).sum(axis=(-2, -1)), dn, out=np.zeros(dn.shape), where=dn > 0)

    r_lag = np.empty((len(u), len(lags)))
    null_max = np.empty((len(u), len(perms)))
    for i, cid in enumerate(u.id.values):
        rate = frame_rate(st[sc == cid], t, dt, P["rate_smooth_ms"])
        R = np.stack([gather(rate, t, ev, grid, shift=float(l)) for l in lags])   # (L,E,T)
        r_lag[i] = corr(_masked_center(R, M))
        for j in range(0, len(perms), P["perm_batch"]):
            blk = perms[j:j + P["perm_batch"]]
            Rp = _masked_center(R[:, blk].transpose(1, 0, 2, 3), M)   # (B, L, E, T)
            null_max[i, j:j + len(blk)] = np.abs(corr(Rp)).max(axis=1)

    best = np.argmax(np.abs(r_lag), axis=1)
    peak_r = r_lag[np.arange(len(u)), best]
    u["tw_peak_r"] = peak_r
    u["tw_best_lag_s"] = lags[best]
    u["tw_p"] = (null_max >= np.abs(peak_r)[:, None]).mean(axis=1)
    u["tw_coupled"] = u.tw_p < 0.05
    return dict(table=u, lags=lags, r_lag=r_lag, null_max=null_max, grid=grid,
                n_events=int(len(ev)), n_pairs=int(M.sum()),
                n_excluded=int((~enough).sum()), n_tested=int(len(u)))


# ------------------------------------------------------------------------- driver
def run_kinematics(cfg: SessionConfig, subset: str = "all", view: str | None = None,
                   **overrides) -> dict:
    """Compute and save reach-aligned kinematics + speed coupling for one session."""
    P = _params(overrides)
    kin = kinematics(cfg, view, **overrides)
    ev = _events(cfg, subset)
    X, grid, ev_used = align(kin["x"], kin["t"], ev, P)
    Y, _, _ = align(kin["y"], kin["t"], ev, P)
    S, _, _ = align(kin["speed"], kin["t"], ev, P)
    SEG, _, _ = align(kin["seg_id"], kin["t"], ev, P)

    print(f"{cfg.session[:10]} '{subset}': {len(ev_used)}/{len(ev)} events with a full "
          f"{-P['win_pre']:+.2f}..{P['win_post']:+.2f} s window; marker {kin['marker']} tracked on "
          f"{kin['frac_tracked']:.1%} of frames, {kin['n_segments']} usable segments "
          f"({kin['frac_valid']:.1%} of frames give a speed); "
          f"{kin['n_jump_frames']} frames rejected as DLC jumps >{P['max_jump_px']:.0f} px")
    with np.errstate(invalid="ignore"):
        cov = np.isfinite(S).mean(axis=0)
    print(f"  tracking coverage across the window: {cov.min():.0%} (worst bin) to "
          f"{cov.max():.0%} (best, at {grid[np.argmax(cov)]:+.2f} s)")

    cp = speed_coupling(cfg, kin, **overrides)
    tb = cp["table"]
    print(f"  session-wide speed coupling on {cp['n_bins_used']}/{cp['n_bins']} bins of "
          f"{cp['bin_s'] * 1000:.0f} ms: {int(tb.coupled.sum())}/{len(tb)} units beat the "
          f"circular-shift null (p<0.05, {cp['n_shifts']} shifts)")
    print("    NOTE tracked frames are mostly reaching, so this largely restates reach "
          "modulation; the trial-wise test below is the one that isolates speed.")
    for r, members in (("CFA", ("CFA",)), ("THALAMUS", THALAMUS)):
        m = tb.region.isin(members)
        if m.any():
            c = m & tb.coupled
            lead = tb.best_lag_s[c]
            print(f"    {r:>8}: {int(c.sum())}/{int(m.sum())} coupled, median |r| "
                  f"{np.abs(tb.peak_r[c]).median() if c.any() else float('nan'):.2f}, "
                  f"median best lag {lead.median() if c.any() else float('nan'):+.3f} s")

    tw = trialwise_coupling(cfg, kin, ev, **overrides)
    tw_tb = tw["table"]
    cols = ["id", "tw_peak_r", "tw_best_lag_s", "tw_p", "tw_coupled", "tw_n_spikes_window"]
    tb = tb.merge(tw_tb[cols], on="id", how="left")      # left: keep units the test skipped
    tb["tw_coupled"] = tb.tw_coupled.notna() & (tb.tw_coupled == True)   # NaN = not tested
    tb["tw_tested"] = tb.tw_p.notna()
    exp = 0.05 * tw["n_tested"]
    n_hit = int(tw_tb.tw_coupled.sum())
    tw_binom = binomtest(n_hit, tw["n_tested"], 0.05, alternative="greater").pvalue
    print(f"  trial-wise residual coupling ({tw['n_events']} events, {tw['n_pairs']} tracked "
          f"(event,bin) pairs; {tw['n_excluded']} units skipped for <"
          f"{P['min_spikes_per_event']:.0f} spikes/event): "
          f"{n_hit}/{tw['n_tested']} units beat the event-permutation "
          f"null (p<0.05; {exp:.1f} expected by chance, binomial p={tw_binom:.1e})")
    for r, members in (("CFA", ("CFA",)), ("THALAMUS", THALAMUS)):
        m = tw_tb.region.isin(members)
        if m.any():
            c = m & tw_tb.tw_coupled
            print(f"    {r:>8}: {int(c.sum())}/{int(m.sum())} coupled, median |r| "
                  f"{np.abs(tw_tb.tw_peak_r[c]).median() if c.any() else float('nan'):.2f}, "
                  f"median best lag "
                  f"{tw_tb.tw_best_lag_s[c].median() if c.any() else float('nan'):+.3f} s")

    out_dir = cfg.analysis_dir / "kinematics"
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / f"reach_kinematics_{subset}.npz", grid=grid, x=X, y=Y, speed=S,
             seg_id=SEG, event_times=ev_used, coverage=cov, lags=cp["lags"], r_lag=cp["r_lag"],
             tw_lags=tw["lags"], tw_r_lag=tw["r_lag"],
             units=tb.id.values, region=tb.region.values.astype(str), dv_um=tb.dv_um.values,
             peak_r=tb.peak_r.values, best_lag_s=tb.best_lag_s.values,
             coupling_z=tb.coupling_z.values, coupling_p=tb.coupling_p.values,
             tw_peak_r=tb.tw_peak_r.values, tw_best_lag_s=tb.tw_best_lag_s.values,
             tw_p=tb.tw_p.values)
    tb.to_csv(out_dir / "speed_coupling.csv", index=False)
    write_json(out_dir / "kinematics_meta.json", dict(
        subset=subset, marker=kin["marker"], n_events=int(len(ev_used)), n_events_total=int(len(ev)),
        frac_tracked=kin["frac_tracked"], frac_valid=kin["frac_valid"], n_segments=kin["n_segments"],
        n_bins_used=cp["n_bins_used"], n_bins=cp["n_bins"],
        n_coupled=int(tb.coupled.sum()), n_units=int(len(tb)),
        n_tw_coupled=int(tb.tw_coupled.sum()), n_tw_pairs=tw["n_pairs"],
        n_tw_tested=tw["n_tested"], n_tw_excluded=tw["n_excluded"],
        n_jump_frames=kin["n_jump_frames"],
        coverage_min=float(cov.min()), coverage_max=float(cov.max()),
        params={k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()}))
    update_status(cfg, f"kinematics_{subset}", True, n_events=int(len(ev_used)),
                  marker=kin["marker"], frac_tracked=kin["frac_tracked"],
                  n_coupled=int(tb.coupled.sum()), n_tw_coupled=int(tb.tw_coupled.sum()),
                  n_units=int(len(tb)))
    return dict(kin=kin, grid=grid, x=X, y=Y, speed=S, seg_id=SEG, event_times=ev_used,
                coverage=cov, coupling=cp, trialwise=tw, table=tb, subset=subset)


def load_kinematics(cfg: SessionConfig, subset: str = "all") -> dict:
    import pandas as pd
    d = np.load(cfg.analysis_dir / "kinematics" / f"reach_kinematics_{subset}.npz")
    out = {k: d[k] for k in d.files}
    out["table"] = pd.read_csv(cfg.analysis_dir / "kinematics" / "speed_coupling.csv")
    return out


# ------------------------------------------------------------------------ figures
def _nan_mean_sem(A):
    with np.errstate(invalid="ignore"):
        n = np.isfinite(A).sum(axis=0)
        m = np.nanmean(A, axis=0)
        sd = np.nanstd(A, axis=0)
    return m, np.divide(sd, np.sqrt(np.maximum(n, 1)), out=np.zeros_like(sd), where=n > 1), n


def _paths(X, Y, SEG, values=None):
    """
    Line segments for a LineCollection, broken wherever the paw is untracked OR the two
    samples come from different tracked segments. Without the second condition the paw
    appears to travel in a straight line across a dropout, which it never did.
    """
    segs, vals = [], []
    for i in range(len(X)):
        ok = (np.isfinite(X[i][:-1]) & np.isfinite(X[i][1:])
              & np.isfinite(Y[i][:-1]) & np.isfinite(Y[i][1:])
              & (SEG[i][:-1] == SEG[i][1:]) & (SEG[i][:-1] >= 0))
        if not ok.any():
            continue
        pts = np.column_stack([X[i], Y[i]]).reshape(-1, 1, 2)
        segs.append(np.concatenate([pts[:-1], pts[1:]], axis=1)[ok])
        vals.append((values[i][:-1] if values.ndim == 2 else values[:-1])[ok])
    if not segs:
        return np.empty((0, 2, 2)), np.empty(0)
    return np.concatenate(segs), np.concatenate(vals)


def _limits(X, Y):
    return (np.nanpercentile(X, 0.5), np.nanpercentile(X, 99.5),
            np.nanpercentile(Y, 99.5), np.nanpercentile(Y, 0.5))


def plot_kinematics(cfg: SessionConfig, res: dict, save=True, min_coverage=0.4):
    """
    The movement (A), its time course with tracking coverage (B), the population response on
    the same time axis (C), the two coupling tests (D-F) and two example units' trajectories
    coloured by their own firing rate (G, H).

    Bins where fewer than `min_coverage` of events have a tracked paw are shaded in B and C:
    there the mean is an average over a small, self-selected subset of reaches.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from .peth import load_reach_peth

    grid, X, Y, S, SEG = res["grid"], res["x"], res["y"], res["speed"], res["seg_id"]
    cov, tb, cp, tw = res["coverage"], res["table"], res["coupling"], res["trialwise"]
    zone = cfg.reach_zone or {}
    x0, x1, y0, y1 = _limits(X, Y)
    thin = cov < min_coverage

    def shade(ax):
        """Grey the time bins where too few events contribute to the mean."""
        a = None
        for i, bad in enumerate(np.append(thin, False)):
            if bad and a is None:
                a = grid[i]
            elif not bad and a is not None:
                ax.axvspan(a, grid[i - 1], color="#BDBDBD", alpha=0.22, lw=0, zorder=0)
                a = None

    fig = plt.figure(figsize=(17, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    # A -- trajectories in camera pixels, coloured by time from onset
    ax = fig.add_subplot(gs[0, 0])
    tgrid = np.tile(grid, (len(X), 1))
    segs, vals = _paths(X, Y, SEG, tgrid)
    norm = plt.Normalize(grid[0], grid[-1])
    lc = LineCollection(segs, cmap="viridis", norm=norm, lw=0.8, alpha=0.55)
    lc.set_array(vals)
    ax.add_collection(lc)
    if zone:
        ax.axvline(zone["x_min"], color="#F5A623", ls="--", lw=1)
        ax.axhline(zone["y_min"], color="#F5A623", ls="--", lw=1)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax,
                 label="time from onset (s)", fraction=0.046)
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.set_title(f"A   paw trajectories, {len(X)} '{res['subset']}' events\n"
                 f"orange dashed = reach zone; gaps = untracked", fontsize=10)

    # B -- x, y, speed vs time from onset, with tracking coverage
    ax = fig.add_subplot(gs[0, 1])
    shade(ax)
    for arr, lab, col in ((X, "x", "#2196F3"), (Y, "y", "#E53935")):
        m, se, _ = _nan_mean_sem(arr)
        ax.plot(grid, m, color=col, lw=1.6, label=lab)
        ax.fill_between(grid, m - se, m + se, color=col, alpha=0.25, lw=0)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("position (px)"); ax.set_xlabel("time from onset (s)")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    axs = ax.twinx()
    m, se, _ = _nan_mean_sem(S)
    axs.plot(grid, m, color="#111111", lw=1.8)
    axs.fill_between(grid, m - se, m + se, color="#111111", alpha=0.2, lw=0)
    axs.plot(grid, cov * axs.get_ylim()[1], color="#7E57C2", lw=1.1, ls="-.")
    axs.set_ylabel("speed (px/s, black)   coverage (purple, 0-1 scaled)")
    ax.set_title(f"B   mean +- SEM kinematics; grey = <{min_coverage:.0%} of events tracked",
                 fontsize=10)

    # C -- population response on the same time axis as the speed
    ax = fig.add_subplot(gs[0, 2])
    shade(ax)
    try:
        pe = load_reach_peth(cfg, res["subset"])
        c, z, mod, reg = pe["centers"], pe["z"], pe["modulated"], pe["region"]
        for lab, members, col in (("CFA", ("CFA",), REGION_COLORS["CFA"]),
                                  ("thalamus", THALAMUS, REGION_COLORS["THAL"])):
            m = np.isin(reg, members) & mod
            if m.sum():
                ax.plot(c, np.abs(z[m]).mean(axis=0), color=col, lw=1.9,
                        label=f"{lab} (n={int(m.sum())})")
        ax.set_ylabel("mean |z| of modulated units")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    except FileNotFoundError:
        ax.text(0.5, 0.5, f"no PETH for subset '{res['subset']}'", ha="center",
                transform=ax.transAxes)
    sm, _, _ = _nan_mean_sem(S)
    axb = ax.twinx()
    axb.plot(grid, sm, color="#111111", lw=1.4, ls="--")
    axb.set_ylabel("paw speed (px/s, dashed)")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlim(grid[0], grid[-1]); ax.set_xlabel("time from onset (s)")
    ax.set_title("C   population response vs the movement", fontsize=10)

    # D -- session-wide r(lag) on tracked frames
    ax = fig.add_subplot(gs[1, 0])
    for lab, members, col in (("CFA", ("CFA",), REGION_COLORS["CFA"]),
                              ("thalamus", THALAMUS, REGION_COLORS["THAL"])):
        m = tb.region.isin(members).values & tb.coupled.values
        if m.sum():
            ax.plot(cp["lags"], cp["r_lag"][m].T, color=col, lw=0.5, alpha=0.2)
            ax.plot(cp["lags"], np.abs(cp["r_lag"][m]).mean(axis=0), color=col, lw=2.3,
                    label=f"{lab} mean |r| (n={int(m.sum())})")
    ax.axvline(0, color="k", lw=0.8, ls=":"); ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("lag (s)   negative = unit leads paw")
    ax.set_ylabel("r (thin, signed)   mean |r| (thick)")
    ax.legend(fontsize=8)
    ax.set_title("D   session-wide coupling on tracked frames\n"
                 "(confounded with reach modulation -- see F)", fontsize=10)

    # E -- trial-wise residual r(lag): the test that isolates speed
    ax = fig.add_subplot(gs[1, 1])
    twt = tw["table"]                       # only the units the trial-wise test could run on
    for lab, members, col in (("CFA", ("CFA",), REGION_COLORS["CFA"]),
                              ("thalamus", THALAMUS, REGION_COLORS["THAL"])):
        m = twt.region.isin(members).values & twt.tw_coupled.values
        if m.sum():
            ax.plot(tw["lags"], tw["r_lag"][m].T, color=col, lw=0.5, alpha=0.25)
            ax.plot(tw["lags"], np.abs(tw["r_lag"][m]).mean(axis=0), color=col, lw=2.3,
                    label=f"{lab} mean |r| (n={int(m.sum())})")
        else:
            ax.plot([], [], color=col, lw=2.3, label=f"{lab} (none)")
    ax.axvline(0, color="k", lw=0.8, ls=":"); ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("lag (s)   negative = unit leads paw")
    ax.set_ylabel("residual r (thin, signed)   mean |r| (thick)")
    ax.legend(fontsize=8)
    ax.set_title("E   trial-wise coupling, stereotyped reach removed", fontsize=10)

    # F -- the two tests against each other
    ax = fig.add_subplot(gs[1, 2])
    t2 = tb[tb.tw_tested]
    col = [REGION_COLORS.get(r, REGION_COLORS["other"]) for r in t2.region]
    ax.scatter(np.abs(t2.peak_r), np.abs(t2.tw_peak_r), s=26, c=col, alpha=0.85, lw=0)
    sig = t2.tw_coupled.values
    ax.scatter(np.abs(t2.peak_r[sig]), np.abs(t2.tw_peak_r[sig]), s=70, facecolors="none",
               edgecolors="k", lw=0.9, label=f"trial-wise p<0.05 (n={int(sig.sum())})")
    lim = max(np.abs(t2.peak_r).max(), np.abs(t2.tw_peak_r).max()) * 1.08
    ax.plot([0, lim], [0, lim], color="#999999", lw=0.9, ls="--")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("|r| session-wide (tracked frames)")
    ax.set_ylabel("|r| trial-wise (residuals)")
    ax.legend(fontsize=7.5)
    ax.set_title("F   how much survives removing the mean reach", fontsize=10)

    # G/H -- example units: trajectory coloured by that unit's own rate
    ex = tb[tb.tw_coupled] if tb.tw_coupled.any() else tb[tb.coupled]
    key = "tw_peak_r" if tb.tw_coupled.any() else "peak_r"
    ex = ex.reindex(ex[key].abs().sort_values(ascending=False).index)
    picks = []
    for members in (("CFA",), THALAMUS):
        sel = ex[ex.region.isin(members)]
        if len(sel):
            picks.append(sel.iloc[0])
    kin = res["kin"]
    _, st, sc = load_units(cfg)
    P = _params({})
    for j, row in enumerate(picks[:2]):
        ax = fig.add_subplot(gs[2, j])
        rate = frame_rate(st[sc == int(row.id)], kin["t"], kin["dt"], P["rate_smooth_ms"])
        Rev = gather(rate, kin["t"], res["event_times"], grid)
        vmin, vmax = np.percentile(Rev, [5, 95])
        norm = plt.Normalize(vmin, vmax)
        segs, vals = _paths(X, Y, SEG, Rev)
        lc = LineCollection(segs, cmap="magma", norm=norm, lw=1.1, alpha=0.8)
        lc.set_array(vals)
        ax.add_collection(lc)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="magma"), ax=ax,
                     label="rate (Hz)", fraction=0.046)
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
        ax.set_title(f"{'GH'[j]}   unit {int(row.id)}  {row.region} {row.dv_um:.0f} um   "
                     f"trial-wise r={row.tw_peak_r:+.2f} at {row.tw_best_lag_s:+.2f} s",
                     fontsize=10)

    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    meta = read_json(cfg.analysis_dir / "kinematics" / "kinematics_meta.json")
    pr = meta["params"]
    ax.text(0, 1, "\n".join([
        f"{cfg.session[:10]}   experiment {cfg.experiment}   subset '{meta['subset']}'",
        f"marker {meta['marker']}, tracked on {meta['frac_tracked']:.1%} of frames",
        f"{meta['n_events']}/{meta['n_events_total']} events with a full window",
        f"coverage across the window {meta['coverage_min']:.0%} to {meta['coverage_max']:.0%}",
        "",
        f"{meta['n_jump_frames']} frames rejected as DLC jumps "
        f">{pr['max_jump_px']:.0f} px/frame",
        "",
        f"session-wide: {meta['n_coupled']}/{meta['n_units']} units p<0.05",
        f"   ({pr['n_shifts']} circular shifts, {meta['n_bins_used']} bins of "
        f"{pr['coupling_bin_s'] * 1000:.0f} ms)",
        f"trial-wise:   {meta['n_tw_coupled']}/{meta['n_tw_tested']} units p<0.05 "
        f"({0.05 * meta['n_tw_tested']:.1f} expected,",
        f"               binomial p="
        f"{binomtest(meta['n_tw_coupled'], meta['n_tw_tested'], 0.05, alternative='greater').pvalue:.1e})",
        f"   ({pr['n_shifts']} event permutations, {meta['n_tw_pairs']} tracked pairs,",
        "    max |r| over all lags taken in the null;",
        f"    {meta['n_tw_excluded']} units skipped for too few spikes)",
        "",
        "The paw is only tracked while reaching, so the session-wide",
        "test cannot separate speed coding from reach modulation.",
        "Panel E/F is the test that can: it keeps only trial-to-trial",
        "deviations from the average reach.",
    ]), va="top", ha="left", fontsize=8.8, family="monospace", transform=ax.transAxes)

    fig.suptitle(f"Paw kinematics and firing -- {cfg.session[:10]} ({res['subset']})",
                 fontsize=14)
    if save:
        (cfg.analysis_dir / "qc").mkdir(exist_ok=True)
        p = cfg.analysis_dir / "qc" / f"kinematics_{res['subset']}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        print(f"  saved {p}")
    return fig
