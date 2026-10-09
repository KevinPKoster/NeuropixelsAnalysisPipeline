"""
Stage 2: camera frames -> canonical time.

Each camera's ExposureActive line produces one NI pulse per exposure. But the
line also pulses whenever a camera free-runs (Live Preview / Test Snap in
acquire_pyspin.py) with no frames saved. So the pulse stream of one camera
contains

    * real recordings: continuous trains at exactly the trigger rate (200.0 Hz),
    * stray free-run bursts: a different rate, no counters CSV.

A real train is identified by (a) its pulse rate and (b) a counters CSV whose
counter span matches its pulse count. The chosen train gives
    frame_time = train_ttl[ counter - first_counter ]
for every saved frame (dropped frames simply have no entry).

Everything is checked loudly: nothing is guessed silently.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import SessionConfig, update_status, write_json
from .sync import load_ttl

MAX_GAP_S = 0.3             # a gap longer than this ends a train
MIN_TRAIN_TTLS = 500        # shorter trains are noise
RATE_TOL = 0.02             # median interval must be within 2% of 1/trigger_hz
MAX_TRAILING_TTLS = 200     # NI pulses after the camera's last counter (cam stops first)
WARN_TRAILING_TTLS = 50
MAX_TTL_INVERSIONS = 5      # tolerated out-of-order NI timestamps per chosen train ...
MAX_INVERSION_FRAC = 0.25   # ... each smaller than this fraction of the frame interval


def find_trains(t: np.ndarray, max_gap_s: float = MAX_GAP_S):
    """Split pulse times into contiguous trains -> list of (i_start, i_end_inclusive)."""
    if len(t) == 0:
        return []
    cut = np.where(np.diff(t) > max_gap_s)[0]
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, len(t) - 1]
    return list(zip(starts.tolist(), ends.tolist()))


def describe_trains(t, trigger_hz):
    out = []
    for i0, i1 in find_trains(t):
        n = i1 - i0 + 1
        med = float(np.median(np.diff(t[i0:i1 + 1]))) if n > 1 else float("nan")
        triggered = bool(n >= MIN_TRAIN_TTLS and abs(med * trigger_hz - 1) <= RATE_TOL)
        out.append(dict(i0=i0, i1=i1, n=n, t_start=float(t[i0]), t_end=float(t[i1]),
                        median_interval_ms=med * 1e3,
                        rate_hz=(1 / med if med == med and med > 0 else None),
                        triggered=triggered))
    return out


def find_counter_csvs(cfg: SessionConfig, view: str):
    """All non-empty counters CSVs for this view recorded on the session's date."""
    import os
    pat = re.compile(rf"^{cfg.video_date_prefix}-(\d+)_{re.escape(view)}_counters\.csv$")
    found = []
    for name in os.listdir(cfg.video_dir):
        m = pat.match(name)
        if not m:
            continue
        path = os.path.join(cfg.video_dir, name)
        df = pd.read_csv(path, usecols=["counter", "saved"])
        if len(df) == 0:
            continue
        ctr = df["counter"].to_numpy(dtype=np.int64)
        found.append(dict(video=int(m.group(1)), path=path, n_rows=len(df),
                          first=int(ctr[0]), last=int(ctr[-1]),
                          span=int(ctr[-1] - ctr[0] + 1),
                          n_saved=int((df["saved"].to_numpy() == 1).sum()),
                          counter=ctr, saved=df["saved"].to_numpy(dtype=np.int64)))
    return sorted(found, key=lambda d: d["video"])


def match_trains_to_csvs(trains, csvs):
    """
    Pair triggered trains with counters CSVs: the CSV's counter span must not exceed the
    train's pulse count and may fall short by at most MAX_TRAILING_TTLS. Closest first;
    each CSV and each train is used once.
    """
    cands = []
    for ti, tr in enumerate(trains):
        if not tr["triggered"]:
            continue
        for cj, c in enumerate(csvs):
            trailing = tr["n"] - c["span"]
            if 0 <= trailing <= MAX_TRAILING_TTLS:
                cands.append((trailing, ti, cj))
    cands.sort()
    used_t, used_c, pairs = set(), set(), []
    for trailing, ti, cj in cands:
        if ti in used_t or cj in used_c:
            continue
        used_t.add(ti); used_c.add(cj); pairs.append((ti, cj, trailing))
    return pairs


def _align_one_view(cfg, view, spec, required):
    t_all = load_ttl(cfg, spec["pin"])
    trains = describe_trains(t_all, spec["trigger_hz"])
    csvs = find_counter_csvs(cfg, view)

    print(f"\n[{view}] NI pin {spec['pin']}: {len(t_all)} pulses in {len(trains)} train(s); "
          f"{len(csvs)} counters CSV(s) on {cfg.video_date_prefix}")
    for i, tr in enumerate(trains):
        kind = "TRIGGERED" if tr["triggered"] else "ignored (rate/length not a real recording)"
        rate = f"{tr['rate_hz']:.1f} Hz" if tr["rate_hz"] else "n/a"
        print(f"   train {i}: n={tr['n']:>8d}  {tr['t_start']:9.2f}->{tr['t_end']:9.2f} s  {rate:>9s}  {kind}")

    pairs = match_trains_to_csvs(trains, csvs)
    for ti, cj, trailing in pairs:
        print(f"   train {ti} <-> video {csvs[cj]['video']} ({csvs[cj]['n_rows']} rows), trailing pulses {trailing}")
    unmatched_csv = [c["video"] for j, c in enumerate(csvs) if j not in {p[1] for p in pairs}]
    unmatched_tr = [i for i, t in enumerate(trains) if t["triggered"] and i not in {p[0] for p in pairs}]
    if unmatched_csv:
        print(f"   counters CSV(s) with no pulse train (recorded outside this OE recording?): videos {unmatched_csv}")
    if unmatched_tr:
        print(f"   WARNING triggered train(s) with no counters CSV: {unmatched_tr}")

    if not pairs:
        msg = f"[{view}] no pulse train matches a counters CSV."
        if required:
            raise RuntimeError(msg + " See train/CSV table above.")
        print("   " + msg + " Skipping this view.")
        return None

    if cfg.segment == "longest":
        ti, cj, trailing = max(pairs, key=lambda p: trains[p[0]]["n"])
    else:
        sel = [p for p in pairs if csvs[p[1]]["video"] == int(cfg.segment)]
        if not sel:
            raise RuntimeError(f"[{view}] segment {cfg.segment} is not among matched videos "
                               f"{[csvs[p[1]]['video'] for p in pairs]}")
        ti, cj, trailing = sel[0]
    tr, c = trains[ti], csvs[cj]
    if trailing > WARN_TRAILING_TTLS:
        print(f"   WARNING {trailing} pulses after the camera's last counter (camera stopped early?)")

    ttl = t_all[tr["i0"]: tr["i1"] + 1]
    nominal_s = 1.0 / spec["trigger_hz"]

    # Open Ephys occasionally writes two adjacent event timestamps out of order by a
    # fraction of a millisecond. Tolerate a handful of those (logged), fail on anything else.
    bad = np.where(np.diff(ttl) <= 0)[0]
    repairs = []
    if len(bad):
        size = np.abs(np.diff(ttl)[bad])
        if len(bad) > MAX_TTL_INVERSIONS or size.max() > MAX_INVERSION_FRAC * nominal_s:
            raise RuntimeError(
                f"[{view}] {len(bad)} out-of-order pulse timestamps in the chosen train "
                f"(largest {size.max()*1e3:.3f} ms); not a small glitch, refusing to repair.")
        for i in bad:
            repairs.append(dict(pulse_index=int(i), t_s=float(ttl[i]), inversion_ms=float((ttl[i + 1] - ttl[i]) * 1e3)))
            print(f"   NOTE out-of-order NI timestamps at pulse {i} (t={ttl[i]:.4f} s, "
                  f"{(ttl[i + 1] - ttl[i]) * 1e3:+.3f} ms): sorted. Affects that frame by < 1 ms.")
        ttl = np.sort(ttl)

    trig_idx = c["counter"] - c["first"]
    saved = c["saved"] == 1
    saved_idx = trig_idx[saved]
    if saved_idx.max() >= len(ttl):
        raise RuntimeError(f"[{view}] counter index {saved_idx.max()} beyond train length {len(ttl)}")
    frame_t = ttl[saved_idx]
    if not np.all(np.diff(frame_t) > 0):
        raise RuntimeError(f"[{view}] frame times are not strictly increasing; counter log inconsistent.")

    iv = np.diff(frame_t) * 1e3
    nominal_ms = 1e3 / spec["trigger_hz"]
    big = int((iv > 3 * nominal_ms).sum())

    a = cfg.analysis_dir / "camera_alignment"
    np.save(a / f"{view}_frame_times.npy", frame_t)          # row i <-> DLC row i <-> i-th saved video frame
    np.save(a / f"{view}_frame_trigger_idx.npy", saved_idx)
    np.save(a / f"{view}_ttl_times.npy", ttl)

    print(f"   USING video {c['video']}: {c['n_saved']} saved frames of {c['n_rows']} "
          f"({c['n_rows'] - c['n_saved']} dropped), {frame_t[0]:.2f}->{frame_t[-1]:.2f} s, "
          f"interval {iv.mean():.4f}+-{iv.std():.4f} ms, gaps>{3*nominal_ms:.0f} ms: {big}")
    return dict(
        view=view, video=c["video"], counters_csv=c["path"], pin=spec["pin"],
        n_ttls_in_train=tr["n"], n_rows=c["n_rows"], n_saved=c["n_saved"],
        n_dropped=c["n_rows"] - c["n_saved"], trailing_ttls=int(trailing),
        first_counter=c["first"], last_counter=c["last"],
        t_first_frame_s=float(frame_t[0]), t_last_frame_s=float(frame_t[-1]),
        interval_mean_ms=float(iv.mean()), interval_std_ms=float(iv.std()),
        n_large_gaps=big, ttl_timestamp_repairs=repairs,
        trains=[{k: v for k, v in t.items()} for t in trains],
        matched=[dict(train=ti_, video=csvs[cj_]["video"], trailing=int(tr_)) for ti_, cj_, tr_ in pairs],
        csvs_without_train=unmatched_csv,
    )


def run_frames(cfg: SessionConfig) -> dict:
    out = {}
    for view, spec in cfg.cameras.items():
        res = _align_one_view(cfg, view, spec, required=(view == cfg.primary_view))
        if res is not None:
            out[view] = res
    write_json(cfg.analysis_dir / "metadata" / "frame_alignment.json", out)
    update_status(cfg, "frames", True,
                  views={v: dict(video=r["video"], n_saved=r["n_saved"], n_dropped=r["n_dropped"])
                         for v, r in out.items()})
    return out


def load_frame_times(cfg: SessionConfig, view: str | None = None) -> np.ndarray:
    return np.load(cfg.analysis_dir / "camera_alignment" / f"{view or cfg.primary_view}_frame_times.npy")
