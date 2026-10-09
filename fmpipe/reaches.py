"""
Stage 5: reach detection from DLC (absolute pixel zone) + per-day zone calibration.

A reach is a run of >= min_frames_in_zone consecutive frames in which paw_center OR digit2
(likelihood >= likelihood_min) is inside the zone  x > x_min AND y > y_min.
Reaches closer than merge_window_s are merged into bouts. This reproduces the
Sept 16 'untrialized' detector of the old notebook.

Camera/slit position changes between days, so the zone is a per-session config value.
`propose_zones` suggests one for each day by locating where the paw ends its reaches
after pellet presentation (back-beam trials) and shifting a reference day's zone by the
difference. It is a starting point: look at `plot_reach_zone` and edit cfg.reach_zone.

Trial structure (back beam) is only an annotation: trial_id / first_in_trial are
stored, but detection does not depend on it, and the (unreliable) pellet beam is only tagged.
"""

from __future__ import annotations

import numpy as np

from .config import SessionConfig, read_json, update_status, write_json
from .dlc import load_dlc
from .sync import load_ttl

REACH_DEFAULTS = dict(
    markers=("paw_center", "digit2"),
    likelihood_min=0.5,
    min_frames_in_zone=5,        # 25 ms at 200 Hz
    merge_window_s=0.8,
    trial_window_s=12.0,         # a trial = back-beam event + this window
    beam_tol_s=0.5,
)


def _params(cfg):
    p = dict(REACH_DEFAULTS)
    p.update(cfg.reach_params)
    return p


def _runs(mask):
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(padded)
    return np.where(d == 1)[0], np.where(d == -1)[0] - 1


def trial_starts(cfg: SessionConfig, t_lo, t_hi, window):
    """Back-beam events whose whole trial window lies inside the video."""
    b = load_ttl(cfg, cfg.trial_pin)
    return b[(b > t_lo) & (b < t_hi - window)]


# ------------------------------------------------------------------ detection
def detect_reaches(cfg: SessionConfig) -> dict:
    if not cfg.reach_zone:
        raise RuntimeError("cfg.reach_zone is not set (needs {'x_min':..., 'y_min':...}). "
                           "Use propose_zones() / plot_reach_zone() to choose one for this day.")
    P = _params(cfg)
    xz, yz = float(cfg.reach_zone["x_min"]), float(cfg.reach_zone["y_min"])
    d = load_dlc(cfg)
    t = d["canonical_time"]
    n = len(t)
    lik_min = P["likelihood_min"]

    in_zone = np.zeros(n, bool)
    xs, ys, ok = {}, {}, {}
    for bp in P["markers"]:
        if f"{bp}_x" not in d:
            print(f"  WARNING marker {bp} not in DLC file")
            continue
        xs[bp], ys[bp] = d[f"{bp}_x"].astype(float), d[f"{bp}_y"].astype(float)
        ok[bp] = d[f"{bp}_likelihood"] >= lik_min
        in_zone |= ok[bp] & (xs[bp] > xz) & (ys[bp] > yz)

    # keep only runs of >= min_frames_in_zone consecutive frames
    s, e = _runs(in_zone)
    keep = (e - s + 1) >= P["min_frames_in_zone"]
    s, e = s[keep], e[keep]
    in_zone_f = np.zeros(n, bool)
    for a, b in zip(s, e):
        in_zone_f[a:b + 1] = True

    onset_i, peak_i, max_y = s.copy(), np.zeros(len(s), int), np.zeros(len(s))
    for k, (a, b) in enumerate(zip(s, e)):
        best = np.full(b - a + 1, -np.inf)
        for bp in xs:
            best = np.maximum(best, np.where(ok[bp][a:b + 1], ys[bp][a:b + 1], -np.inf))
        j = int(np.argmax(best))
        peak_i[k], max_y[k] = a + j, best[j]
    onset_t, peak_t = t[onset_i], t[peak_i]

    # bouts: a reach starting within merge_window of the previous reach's peak joins its bout
    bout_id = np.zeros(len(s), int)
    for k in range(1, len(s)):
        bout_id[k] = bout_id[k - 1] + (0 if onset_t[k] - peak_t[k - 1] <= P["merge_window_s"] else 1)
    n_bouts = int(bout_id.max()) + 1 if len(s) else 0

    # annotations: pellet beam near the reach peak; back-beam trial the reach falls in
    beam = load_ttl(cfg, cfg.beam_pin)
    broke = np.array([bool(len(beam) and np.any(np.abs(beam - tp) < P["beam_tol_s"])) for tp in peak_t], bool)
    starts = load_ttl(cfg, cfg.trial_pin)
    def trial_of(times):
        idx = np.searchsorted(starts, times, side="right") - 1
        inside = (idx >= 0) & (times < starts[np.clip(idx, 0, None)] + P["trial_window_s"]) if len(starts) else np.zeros(len(times), bool)
        return np.where(inside, idx, -1)
    reach_trial = trial_of(onset_t)

    b_onset = np.array([onset_t[bout_id == b].min() for b in range(n_bouts)])
    b_peak = np.array([peak_t[bout_id == b][np.argmax(max_y[bout_id == b])] for b in range(n_bouts)])
    b_amp = np.array([max_y[bout_id == b].max() for b in range(n_bouts)])
    b_n = np.array([(bout_id == b).sum() for b in range(n_bouts)])
    b_beam = np.array([broke[bout_id == b].any() for b in range(n_bouts)], bool)
    b_trial = trial_of(b_onset)
    first = np.zeros(n_bouts, bool)
    seen = set()
    for k in range(n_bouts):
        if b_trial[k] >= 0 and b_trial[k] not in seen:
            first[k] = True
            seen.add(int(b_trial[k]))

    a = cfg.analysis_dir / "peak_detection"
    a.mkdir(exist_ok=True)
    np.savez(a / "reaches.npz", onset_t=onset_t, peak_t=peak_t, max_y=max_y, frame_onset=onset_i,
             frame_peak=peak_i, bout_id=bout_id, trial_id=reach_trial, broke_beam=broke)
    np.savez(a / "bouts.npz", onset_t=b_onset, peak_t=b_peak, max_y=b_amp, n_reaches=b_n,
             trial_id=b_trial, first_in_trial=first, broke_beam=b_beam)
    np.save(a / "in_zone_mask.npy", in_zone_f)

    enrich = trial_enrichment(cfg, b_onset, t, P)
    meta = dict(method="absolute_pixel_zone", reach_zone=dict(x_min=xz, y_min=yz), **{
        k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()},
        n_reaches=int(len(s)), n_bouts=n_bouts, n_trials=int(len(starts)),
        n_trials_with_bout=int(len(seen)), n_beam_tagged_bouts=int(b_beam.sum()),
        enrichment=enrich)
    write_json(a / "reaches_meta.json", meta)

    print(f"Zone x>{xz:.0f}, y>{yz:.0f}: {len(s)} reaches in {n_bouts} bouts "
          f"({int(b_beam.sum())} beam-tagged)")
    print(f"  back-beam trials with a bout: {len(seen)}/{len(starts)}; "
          f"bout-onset rate inside trial windows {enrich['p_trial']:.2f} vs random windows "
          f"{enrich['p_random']:.2f}  (enrichment {enrich['ratio']:.1f}x)")
    update_status(cfg, "reaches", True, n_reaches=int(len(s)), n_bouts=n_bouts,
                  zone=dict(x_min=xz, y_min=yz), enrichment=enrich["ratio"])
    return dict(onset_t=onset_t, peak_t=peak_t, bout_onset_t=b_onset, bout_peak_t=b_peak, meta=meta)


def trial_enrichment(cfg, bout_onsets, frame_t, P, n_random=2000, seed=0):
    """
    Sanity measure for a zone: reaches should follow pellet presentation. Fraction of
    back-beam trial windows containing a bout onset vs the same for random windows that
    do not overlap any trial.
    """
    W = P["trial_window_s"]
    starts = trial_starts(cfg, frame_t[0], frame_t[-1], W)
    if len(starts) == 0:
        return dict(p_trial=float("nan"), p_random=float("nan"), ratio=float("nan"), n_trials=0)
    allstarts = load_ttl(cfg, cfg.trial_pin)
    has = lambda a0: np.searchsorted(bout_onsets, a0 + W) - np.searchsorted(bout_onsets, a0) > 0
    p_trial = float(has(starts).mean())
    rng = np.random.default_rng(seed)
    cand = rng.uniform(frame_t[0], frame_t[-1] - W, size=n_random * 3)
    near = np.searchsorted(allstarts, cand + W) - np.searchsorted(allstarts, cand - W) > 0   # overlaps a trial
    cand = cand[~near][:n_random]
    p_rand = float(has(cand).mean()) if len(cand) else float("nan")
    ratio = p_trial / p_rand if p_rand and p_rand > 0 else float("inf")
    return dict(p_trial=p_trial, p_random=p_rand, ratio=float(ratio), n_trials=int(len(starts)))


# ------------------------------------------------------------ zone calibration
def trial_endpoints(cfg: SessionConfig, marker="paw_center"):
    """
    Where the paw ends its reach: for each back-beam trial, the confident frame with the
    largest x in the 12 s after the arm comes up. Returns (n_trials, 2) of (x, y).
    """
    P = _params(cfg)
    d = load_dlc(cfg)
    t, x, y = d["canonical_time"], d[f"{marker}_x"], d[f"{marker}_y"]
    lik = d[f"{marker}_likelihood"]
    pts = []
    for b in trial_starts(cfg, t[0], t[-1], P["trial_window_s"]):
        i0, i1 = np.searchsorted(t, [b, b + P["trial_window_s"]])
        ok = np.where(lik[i0:i1] >= P["likelihood_min"])[0]
        if len(ok) >= 20:
            j = i0 + ok[np.argmax(x[i0:i1][ok])]
            pts.append((x[j], y[j]))
    return np.array(pts)


def propose_zones(cfgs: dict, ref: str, ref_zone=(400.0, 380.0), use_ref_endpoint=None):
    """
    cfgs: {name: SessionConfig}. Proposal for each day = ref_zone + (median endpoint of the day
    - median endpoint of the reference day). `use_ref_endpoint` (x, y) overrides the reference
    day's endpoint, e.g. when the reference day's own endpoints are noisy.
    """
    ends = {k: trial_endpoints(c) for k, c in cfgs.items()}
    r = np.median(ends[ref], axis=0) if use_ref_endpoint is None else np.asarray(use_ref_endpoint, float)
    out = {}
    for k, e in ends.items():
        med = np.median(e, axis=0)
        shift = med - r
        out[k] = dict(x_min=float(round(ref_zone[0] + shift[0])), y_min=float(round(ref_zone[1] + shift[1])),
                      endpoint=med.tolist(), n_trials=int(len(e)), shift=shift.tolist())
    return out


# ------------------------------------------------------------------- plotting
def reference_frame(cfg: SessionConfig):
    """A video frame at a typical reach endpoint, cached as qc/reference_frame.png (portable)."""
    import matplotlib.image as mpimg
    p = cfg.analysis_dir / "qc" / "reference_frame.png"
    if p.exists():
        return mpimg.imread(p)
    import cv2
    fa = read_json(cfg.analysis_dir / "metadata" / "frame_alignment.json")[cfg.primary_view]
    mp4 = f"{cfg.video_dir}/{cfg.video_date_prefix}-{fa['video']}_{cfg.primary_view}.mp4"
    ep = trial_endpoints(cfg)
    d = load_dlc(cfg)
    x, y, lik = d["paw_center_x"], d["paw_center_y"], d["paw_center_likelihood"]
    target = np.median(ep, axis=0) if len(ep) else np.array([np.median(x), np.median(y)])
    cand = np.where(lik >= 0.9)[0]
    j = cand[np.argmin(np.hypot(x[cand] - target[0], y[cand] - target[1]))]
    cap = cv2.VideoCapture(mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {j} from {mp4}")
    cv2.imwrite(str(p), fr)
    return mpimg.imread(p)


def plot_reach_zone(cfg: SessionConfig, save=True):
    """Frame + paw positions + the zone, and the reaches found, so the zone can be judged by eye."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    P = _params(cfg)
    d = load_dlc(cfg)
    t = d["canonical_time"]
    xz, yz = cfg.reach_zone["x_min"], cfg.reach_zone["y_min"]
    img = reference_frame(cfg)
    a = cfg.analysis_dir / "peak_detection"
    bouts = np.load(a / "bouts.npz") if (a / "bouts.npz").exists() else None
    reaches = np.load(a / "reaches.npz") if (a / "reaches.npz").exists() else None

    fig = plt.figure(figsize=(17, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img)
    rng = np.random.default_rng(0)
    for bp, col in zip(P["markers"], ("cyan", "orange")):
        ok = np.where(d[f"{bp}_likelihood"] >= P["likelihood_min"])[0]
        pick = rng.choice(ok, min(6000, len(ok)), replace=False)
        ax.scatter(d[f"{bp}_x"][pick], d[f"{bp}_y"][pick], s=1.5, alpha=0.3, c=col, label=bp)
    if reaches is not None and len(reaches["frame_peak"]):
        fp = reaches["frame_peak"]
        ax.scatter(d["paw_center_x"][fp], d["paw_center_y"][fp], s=22, facecolors="none",
                   edgecolors="red", lw=0.8, label=f"reach peaks (n={len(fp)})")
    ep = trial_endpoints(cfg)
    if len(ep):
        ax.scatter(ep[:, 0], ep[:, 1], s=26, c="lime", marker="x", lw=1, label="per-trial endpoint")
    ax.add_patch(mp.Rectangle((xz, yz), 4000, 4000, fc="yellow", ec="yellow", alpha=0.12))
    ax.axvline(xz, color="yellow", ls="--", lw=1.2); ax.axhline(yz, color="yellow", ls="--", lw=1.2)
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
    ax.set_title(f"{cfg.session[:10]}  reach zone x>{xz:.0f}, y>{yz:.0f}  (edit cfg.reach_zone to move it)")
    ax.legend(fontsize=8, loc="lower left")

    # signed distance from the zone corner over time, with reaches and back-beam trial starts
    ax2 = fig.add_subplot(gs[0, 1])
    best = np.full(len(t), -np.inf)
    for bp in P["markers"]:
        okm = d[f"{bp}_likelihood"] >= P["likelihood_min"]
        dist = np.hypot(d[f"{bp}_x"] - xz, d[f"{bp}_y"] - yz) * np.where((d[f"{bp}_x"] > xz) & (d[f"{bp}_y"] > yz), 1, -1)
        best = np.maximum(best, np.where(okm, dist, -np.inf))
    best[~np.isfinite(best)] = np.nan
    sub = slice(None, None, 10)
    ax2.plot(t[sub], best[sub], lw=0.3, c="k")
    ax2.axhline(0, c="red", ls="--", lw=1)
    for tb in load_ttl(cfg, cfg.trial_pin):
        ax2.axvspan(tb, tb + P["trial_window_s"], color="tab:green", alpha=0.12, lw=0)
    if bouts is not None:
        ax2.plot(bouts["onset_t"], np.full(len(bouts["onset_t"]), np.nanmax(best) * 1.02), "v", c="tab:blue", ms=3)
    ax2.set_ylabel("signed distance to zone corner (px)"); ax2.set_xlabel("canonical time (s)")
    ax2.set_title("green = back-beam trial windows, blue = detected bouts")

    ax3 = fig.add_subplot(gs[1, 1])
    if bouts is not None and len(bouts["onset_t"]):
        # bout-onset rate around trial starts
        starts = load_ttl(cfg, cfg.trial_pin)
        rel = np.concatenate([bouts["onset_t"] - s0 for s0 in starts])
        ax3.hist(rel, bins=np.arange(-20, 40, 1.0), color="tab:blue")
        ax3.axvspan(0, P["trial_window_s"], color="tab:green", alpha=0.12)
        ax3.set_xlabel("time from back-beam event (s)"); ax3.set_ylabel("bout onsets")
        m = read_json(a / "reaches_meta.json")["enrichment"]
        ax3.set_title(f"reaches should follow pellet presentation  (enrichment {m['ratio']:.1f}x)")
    plt.tight_layout()
    if save:
        fig.savefig(cfg.analysis_dir / "qc" / "reach_zone.png", dpi=110)
    return fig
