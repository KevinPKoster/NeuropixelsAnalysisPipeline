"""
Stage 4: DeepLabCut output -> canonical time.

DLC row i is the i-th frame of the MP4, which is the i-th SAVED frame of the camera
ledger, so   canonical_time[i] = frame_times[i]   (written by stage 2).

Checks (all raise):
  * the .h5 belongs to the video that stage 2 aligned (DDMMYYYY-N in the filename),
  * the .h5 has exactly as many rows as there are saved frames. The only exception is a
    declared tail shortfall (cfg.dlc_max_tail_missing), for acquisitions that stopped
    abnormally and never wrote the last frames to disk.

Writes dlc_alignment/<view>_dlc.npz with canonical_time, bodyparts and, for every
bodypart, <bp>_x, <bp>_y, <bp>_likelihood (float32).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SessionConfig, read_json, update_status, write_json
from .frames import load_frame_times

REPORT_PARTS = ("paw_center", "digit2", "nose")


def read_dlc_h5(path):
    """-> (bodyparts list, dict bp -> (x, y, likelihood)) from a DeepLabCut .h5."""
    df = pd.read_hdf(path)
    bps = list(dict.fromkeys(df.columns.get_level_values("bodyparts")))
    scorers = list(dict.fromkeys(df.columns.get_level_values("scorer")))
    if len(scorers) != 1:
        raise RuntimeError(f"{path}: expected one DLC scorer, found {scorers}")
    out = {bp: tuple(df[(scorers[0], bp, c)].to_numpy(dtype=np.float32) for c in ("x", "y", "likelihood"))
           for bp in bps}
    return bps, out


def run_dlc(cfg: SessionConfig) -> dict:
    if not cfg.dlc_h5:
        print("No DLC .h5 configured (cfg.dlc_h5 is empty): stage skipped.")
        update_status(cfg, "dlc", False, reason="no dlc_h5 configured")
        return {}
    fa = read_json(cfg.analysis_dir / "metadata" / "frame_alignment.json")
    summary = {}
    for view, path in cfg.dlc_h5.items():
        if view not in fa:
            raise RuntimeError(f"[{view}] DLC given but stage 2 produced no alignment for this view.")
        path = Path(path)
        m = re.match(r"^(\d{8})-(\d+)_", path.name)
        if not m or m.group(1) != cfg.video_date_prefix or int(m.group(2)) != fa[view]["video"]:
            raise RuntimeError(
                f"[{view}] DLC file {path.name} does not belong to the aligned video "
                f"{cfg.video_date_prefix}-{fa[view]['video']}.")

        ft = load_frame_times(cfg, view)
        bps, data = read_dlc_h5(path)
        n_rows = len(data[bps[0]][0])
        missing = len(ft) - n_rows
        allowed = int(cfg.dlc_max_tail_missing.get(view, 0))
        if missing != 0:
            if 0 < missing <= allowed:
                print(f"[{view}] NOTE DLC has {missing} fewer rows than saved frames; declared tail "
                      f"shortfall (<= {allowed}), using the first {n_rows} frames.")
                ft = ft[:n_rows]
            else:
                raise RuntimeError(
                    f"[{view}] DLC rows ({n_rows}) != saved frames ({len(ft)}) (difference {missing}). "
                    f"If the acquisition lost the last frames, declare it with "
                    f"cfg.dlc_max_tail_missing={{'{view}': {max(missing, 0)}}}.")

        arrays = {"canonical_time": ft, "bodyparts": np.array(bps)}
        for bp, (x, y, lik) in data.items():
            arrays[f"{bp}_x"], arrays[f"{bp}_y"], arrays[f"{bp}_likelihood"] = x, y, lik
        np.savez(cfg.analysis_dir / "dlc_alignment" / f"{view}_dlc.npz", **arrays)

        stats = {}
        for bp in REPORT_PARTS:
            if bp in data:
                lik = data[bp][2]
                stats[bp] = dict(median_likelihood=float(np.median(lik)),
                                 frac_ge_0p5=float((lik >= 0.5).mean()),
                                 frac_ge_0p9=float((lik >= 0.9).mean()))
        model = re.search(r"(shuffle\d+_snapshot_best-\d+)", path.name)
        summary[view] = dict(h5=str(path), model=model.group(1) if model else None, n_rows=n_rows,
                             n_saved_frames=len(ft) + max(missing, 0), tail_missing=max(missing, 0),
                             t_first_s=float(ft[0]), t_last_s=float(ft[-1]), tracking=stats)
        print(f"[{view}] DLC {summary[view]['model']}: {n_rows} rows <-> frames "
              f"{ft[0]:.2f}->{ft[-1]:.2f} s")
        for bp, s in stats.items():
            print(f"   {bp:<11s} median likelihood {s['median_likelihood']:.3f}   "
                  f">=0.5: {s['frac_ge_0p5']:.1%}   >=0.9: {s['frac_ge_0p9']:.1%}")

    write_json(cfg.analysis_dir / "metadata" / "dlc_alignment.json", summary)
    update_status(cfg, "dlc", True, views={v: dict(model=s["model"], n_rows=s["n_rows"],
                                                   tail_missing=s["tail_missing"]) for v, s in summary.items()})
    return summary


def load_dlc(cfg: SessionConfig, view: str | None = None) -> dict:
    d = np.load(cfg.analysis_dir / "dlc_alignment" / f"{view or cfg.primary_view}_dlc.npz")
    return {k: d[k] for k in d.files}
