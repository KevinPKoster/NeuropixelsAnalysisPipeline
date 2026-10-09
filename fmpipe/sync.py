"""
Stage 1: common timebase + NI TTLs.

Canonical time = seconds since the start of the Neuropixels stream (t0).
Open Ephys timestamps NP and NI on one clock; the 1 Hz sync pulses recorded on
both are used to VERIFY that (slope ~1, offset ~0, tiny residuals), and the
stage raises if they do not agree. Spikes, TTLs, frames and DLC are all later
expressed as `absolute_time - t0`, so nothing downstream needs the fit.
"""

from __future__ import annotations

import numpy as np

from .config import SessionConfig, update_status, write_json

# acceptance limits for the NP<->NI sync check
MAX_SLOPE_ERR = 1e-6          # |a - 1|
MAX_OFFSET_S = 1e-3           # |b|
MAX_RESID_STD_S = 5e-4        # residual std
MIN_MATCH_FRACTION = 0.99     # fraction of sync pulses (shorter stream) that must pair up
PAIR_TOL_S = 5e-3             # a pulse pairs with its nearest partner within this


def rising_edges(full_words: np.ndarray, timestamps: np.ndarray, pin: int) -> np.ndarray:
    """Times at which digital line `pin` goes low -> high."""
    on = (full_words & (1 << pin)) > 0
    prev = np.empty_like(on)
    prev[0] = False
    prev[1:] = on[:-1]
    return timestamps[on & ~prev]


def _first_last(path):
    """First and last value of a big .npy without loading it."""
    m = np.load(path, mmap_mode="r")
    return float(m[0]), float(m[-1]), len(m)


def _pair_sync(t_np, t_ni):
    """
    Pair each pulse of the shorter train with its nearest partner in the other.
    Assumes the two clocks agree to within PAIR_TOL_S (verified by the fit that follows).
    """
    if len(t_np) <= len(t_ni):
        short, long_, swap = t_np, t_ni, False
    else:
        short, long_, swap = t_ni, t_np, True
    idx = np.clip(np.searchsorted(long_, short), 1, len(long_) - 1)
    nearest = np.where(np.abs(long_[idx - 1] - short) <= np.abs(long_[idx] - short), idx - 1, idx)
    ok = np.abs(long_[nearest] - short) <= PAIR_TOL_S
    s, l = short[ok], long_[nearest[ok]]
    return (l, s, ok.mean()) if swap else (s, l, ok.mean())   # -> (np, ni, frac)


def run_sync(cfg: SessionConfig) -> dict:
    a = cfg.analysis_dir

    # ---- stream spans (cheap: first/last timestamp only) --------------------
    np_start, np_end, _ = _first_last(cfg.np_cont_dir / "timestamps.npy")
    ni_start, ni_end, _ = _first_last(cfg.ni_cont_dir / "timestamps.npy")
    t0 = np_start
    duration = np_end - t0

    # ---- events -------------------------------------------------------------
    ni_ts = np.load(cfg.ni_ttl_dir / "timestamps.npy")
    ni_fw = np.load(cfg.ni_ttl_dir / "full_words.npy")
    np_ts = np.load(cfg.np_ttl_dir / "timestamps.npy")
    np_fw = np.load(cfg.np_ttl_dir / "full_words.npy")

    sync_ni = rising_edges(ni_fw, ni_ts, cfg.ni_sync_bit)
    sync_np = rising_edges(np_fw, np_ts, cfg.np_sync_bit)
    if len(sync_ni) < 10 or len(sync_np) < 10:
        raise RuntimeError(f"Too few sync pulses (NI bit {cfg.ni_sync_bit}: {len(sync_ni)}, "
                           f"NP bit {cfg.np_sync_bit}: {len(sync_np)}).")

    p_np, p_ni, frac = _pair_sync(sync_np, sync_ni)
    if frac < MIN_MATCH_FRACTION:
        raise RuntimeError(
            f"Only {frac:.1%} of sync pulses could be paired between NP and NI "
            f"(need {MIN_MATCH_FRACTION:.0%}). The two clocks do not agree; check sync wiring/bits.")
    A = np.vstack([p_np - t0, np.ones_like(p_np)]).T
    slope, off = np.linalg.lstsq(A, p_ni - t0, rcond=None)[0]
    resid = (p_ni - t0) - (slope * (p_np - t0) + off)
    fit = dict(
        slope=float(slope), offset_s=float(off), n_paired=int(len(p_np)),
        n_ni_pulses=int(len(sync_ni)), n_np_pulses=int(len(sync_np)),
        resid_std_s=float(resid.std()), resid_max_abs_s=float(np.abs(resid).max()),
    )
    problems = []
    if abs(slope - 1) > MAX_SLOPE_ERR: problems.append(f"slope {slope:.9f}")
    if abs(off) > MAX_OFFSET_S: problems.append(f"offset {off*1e3:.3f} ms")
    if resid.std() > MAX_RESID_STD_S: problems.append(f"residual std {resid.std()*1e3:.3f} ms")
    if problems:
        raise RuntimeError("NP<->NI clocks disagree: " + "; ".join(problems))

    # ---- TTLs on all 8 NI lines, canonical seconds -------------------------
    ttl_dir = a / "ttls"
    for stale in ttl_dir.glob("TTL_Pin_*.npy"):
        stale.unlink()
    counts = {}
    for pin in range(8):
        t = rising_edges(ni_fw, ni_ts, pin) - t0
        np.save(ttl_dir / f"TTL_Pin_{pin}.npy", t)
        counts[pin] = int(len(t))

    tb = dict(
        t0_abs=t0, np_end_s=duration,
        ni_start_s=ni_start - t0, ni_end_s=ni_end - t0,
        recording_duration_s=duration,
        sync=fit, ttl_counts=counts,
        pin_labels={str(k): v for k, v in cfg.pin_labels.items()},
    )
    write_json(a / "metadata" / "timebase.json", tb)

    print(f"t0 = NP start = {t0:.6f} s (OE clock); duration {duration:.1f} s; "
          f"NI spans {tb['ni_start_s']:+.3f} -> {tb['ni_end_s']:.1f} s")
    print(f"Sync: {fit['n_paired']} pulses paired (NI {fit['n_ni_pulses']}, NP {fit['n_np_pulses']}); "
          f"slope-1={slope-1:+.2e}, offset={off*1e6:.1f} us, resid std={resid.std()*1e6:.1f} us")
    for pin in range(8):
        print(f"  pin {pin} {cfg.pin_labels.get(pin, ''):<20s} {counts[pin]:>9d} rising edges")
    update_status(cfg, "sync", True, resid_std_us=float(resid.std() * 1e6),
                  duration_s=duration, ttl_counts=counts)
    return tb


def load_ttl(cfg: SessionConfig, pin: int) -> np.ndarray:
    return np.load(cfg.analysis_dir / "ttls" / f"TTL_Pin_{pin}.npy")
