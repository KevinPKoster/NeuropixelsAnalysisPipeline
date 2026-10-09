"""
Create a per-session analysis notebook from the standard template.

    python tools/new_session_notebook.py 2026-09-16_13-21-12 --experiment 2

Writes 'Anchored analysis/Freely moving analysis/Mouse 4103/<session-date>.ipynb'.
Only the CONFIG cell differs between days.
"""

import argparse
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "Anchored analysis" / "Freely moving analysis" / "Mouse 4103"

MD_TITLE = """# {session}  (experiment {experiment})  -  freely moving, mouse 4103

Each stage reads only what the previous stage wrote into the analysis folder, and
writes exactly what the next stage reads. Raw data is never modified.

| stage | function | writes |
|---|---|---|
| 0 | `run_layout` | `metadata/channel_layout.*` (what was actually recorded) |
| 1 | `run_sync` | `metadata/timebase.json`, `ttls/TTL_Pin_*.npy` |
| 2 | `run_frames` | `camera_alignment/<view>_frame_times.npy`, `metadata/frame_alignment.json` |
| 3 | `run_units` | `units/units.csv` (label, best channel, depth, DV, region), `units/spike_times.npy` |
| 4 | `run_dlc` | `dlc_alignment/<view>_dlc.npz` (all bodyparts on the canonical clock) |
| 5 | `detect_reaches` | `peak_detection/reaches.npz`, `bouts.npz`, `reaches_meta.json` |
| 6 | `run_reach_peth` | `peth/reach_peth_<subset>.npz/.json/_units.csv`, figures in `qc/` |

Canonical time = seconds since the start of the Neuropixels stream."""

CODE_SETUP = '''# --- setup: find the repo so `import fmpipe` works on any machine -------------
import sys
from pathlib import Path
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "fmpipe").is_dir())
sys.path.insert(0, str(REPO))

import numpy as np
import matplotlib.pyplot as plt
from fmpipe import (SessionConfig, run_layout, run_sync, run_frames, run_units, plot_probe_units,
                    run_dlc, load_dlc, detect_reaches, plot_reach_zone,
                    run_reach_peth, plot_reach_peth, plot_peth_depth, plot_stacked_regions, compare_peth,
                    plot_probe_modulation,
                    load_ttl, load_frame_times, read_json)'''

CODE_CONFIG = '''# --- CONFIG: the only cell you edit ------------------------------------------
cfg = SessionConfig(
    data_root = r"{data_root}",      # folder that holds the session folders (any drive letter)
    session   = "{session}",
    experiment = {experiment},                  # Open Ephys experiment number
    video_dir = r"S:\\MP4_BACKUP",               # counters CSVs / MP4s
    cameras   = {cameras},
    segment   = "longest",                      # or a video number, e.g. 3
    # kilosort4 copies can live on several drives with different curation; the copy with the
    # newest curation is used (missing roots are skipped, so this is safe on any machine)
    ks_search_roots = (r"C:\\Users\\Shermanlab\\Documents\\Open Ephys", "Z:/"),
    dlc_h5    = {dlc},                          # view -> DLC .h5 (leave {{}} until DLC is done)
    dlc_max_tail_missing = {tail},              # only if acquisition lost the last frames (disk full)
    # Reach zone (pixels, primary camera): a reach = paw_center/digit2 with x > x_min AND y > y_min.
    # The camera moves between days. This was PROPOSED automatically from where the paw ends its
    # reaches after pellet presentation: look at the plot below and edit if it looks wrong.
    reach_zone = {zone},
)
cfg.init_analysis()
cfg.save()
print("analysis folder:", cfg.analysis_dir)'''

CODE_STAGES = '''# --- stages 0-3 ----------------------------------------------------------------
run_layout(cfg)
run_sync(cfg)
run_frames(cfg)
units = run_units(cfg)
plot_probe_units(cfg, units); plt.show()
run_dlc(cfg)

# --- stage 5: reaches ----------------------------------------------------------
if cfg.reach_zone:
    detect_reaches(cfg)
    plot_reach_zone(cfg); plt.show()

    # --- stage 6: reach-locked PETH (shuffle-calibrated), by region and by depth ---------
    res = run_reach_peth(cfg, subset="all")      # or "first_in_trial" / "in_trial"
    plot_reach_peth(cfg, res); plt.show()
    plot_peth_depth(cfg, res); plt.show()
    plot_stacked_regions(cfg, res, "all"); plt.show()   # CFA on top, VAL below, latency-sorted
    plot_probe_modulation(cfg, res, "all", theme="dark"); plt.show()      # responses along the probe
    plot_probe_modulation(cfg, res, "all", theme="light")

    # --- robustness: EVERY reach (not only the first of a bout), window -1.5..+1.5 s, response -1..+1 s.
    # The null keeps each bout's internal reach timing and randomises only where bouts fall.
    res_all = run_reach_peth(cfg, subset="all_reaches")
    plot_stacked_regions(cfg, res_all, "all_reaches"); plt.show()
    plot_probe_modulation(cfg, res_all, "all_reaches", theme="dark"); plt.show()
    plot_probe_modulation(cfg, res_all, "all_reaches", theme="light")
    compare_peth(cfg, "all", "all_reaches"); plt.show()'''

CODE_SANITY = '''# --- sanity overview -----------------------------------------------------------
tb = read_json(cfg.analysis_dir / "metadata" / "timebase.json")
fa = read_json(cfg.analysis_dir / "metadata" / "frame_alignment.json")
view = cfg.primary_view
ft = load_frame_times(cfg, view)

fig, ax = plt.subplots(3, 1, figsize=(14, 8))
for pin in range(8):
    t = load_ttl(cfg, pin)
    if len(t) and pin != cfg.cameras.get(view, {}).get("pin"):
        ax[0].plot(t, np.full_like(t, pin), "|", ms=8, label=f"{pin} {cfg.pin_labels[pin]}")
ax[0].axvspan(ft[0], ft[-1], color="tab:blue", alpha=0.08, label=f"{view} frames")
ax[0].set_yticks(range(8)); ax[0].set_xlabel("canonical time (s)")
ax[0].set_title("NI TTL lines (camera line omitted) and extent of the aligned video")
ax[0].legend(fontsize=7, ncol=4, loc="upper right")

iv = np.diff(ft) * 1e3
ax[1].hist(iv, bins=np.linspace(iv.min(), min(iv.max(), 10), 120), color="k")
ax[1].set_yscale("log"); ax[1].set_xlabel("frame interval (ms)")
ax[1].set_title(f"{view}: {len(ft)} frames, interval {iv.mean():.4f} +- {iv.std():.4f} ms")

for v, r in fa.items():
    for i, tr in enumerate(r["trains"]):
        ax[2].plot([tr["t_start"], tr["t_end"]], [list(fa).index(v)] * 2, lw=6,
                   color="tab:green" if tr["triggered"] else "tab:red", solid_capstyle="butt")
ax[2].set_yticks(range(len(fa))); ax[2].set_yticklabels(list(fa))
ax[2].set_xlabel("canonical time (s)")
ax[2].set_title("camera pulse trains: green = real recording, red = ignored (free-run bursts / strays)")
plt.tight_layout(); plt.show()

print("sync residual std: %.1f us" % (tb["sync"]["resid_std_s"] * 1e6))
print("recording duration: %.1f s" % tb["recording_duration_s"])
for v, r in fa.items():
    print(f"{v}: video {r['video']}, {r['n_saved']} frames, {r['n_dropped']} dropped")'''


def build(session, experiment, data_root, cameras, dlc="{}", tail="{}", zone="None"):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(MD_TITLE.format(session=session, experiment=experiment)),
        nbf.v4.new_code_cell(CODE_SETUP),
        nbf.v4.new_code_cell(CODE_CONFIG.format(session=session, experiment=experiment,
                                                data_root=data_root, cameras=cameras,
                                                dlc=dlc, tail=tail, zone=zone)),
        nbf.v4.new_code_cell(CODE_STAGES),
        nbf.v4.new_code_cell(CODE_SANITY),
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--experiment", type=int, default=1)
    ap.add_argument("--data-root", default="K:/")
    ap.add_argument("--second-cam-hz", type=float, default=120.19)
    ap.add_argument("--dlc", default=None, help="DLC .h5 for the primary view (paw_angled)")
    ap.add_argument("--dlc-tail-missing", type=int, default=0,
                    help="frames the acquisition lost at the end (e.g. disk full)")
    ap.add_argument("--zone", default=None, help="reach zone corner 'x_min,y_min' in pixels")
    a = ap.parse_args()
    zone = "None"
    if a.zone:
        zx, zy = [float(v) for v in a.zone.split(",")]
        zone = f'{{"x_min": {zx:.0f}, "y_min": {zy:.0f}}}'
    cams = ('{"paw_angled": dict(pin=0, trigger_hz=200.0),\n'
            f'                 "bottom": dict(pin=1, trigger_hz={a.second_cam_hz})}}')
    dlc = f'{{"paw_angled": r"{a.dlc}"}}' if a.dlc else "{}"
    tail = f'{{"paw_angled": {a.dlc_tail_missing}}}' if a.dlc_tail_missing else "{}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{a.session[:10]}.ipynb"
    nbf.write(build(a.session, a.experiment, a.data_root, cams, dlc, tail, zone), out)
    print("wrote", out)
