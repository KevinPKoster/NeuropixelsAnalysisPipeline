"""
Session registry for mouse 4103 (freely moving). One row per analysed recording.

Used by `regenerate_notebooks.py` to (re)create the per-day notebooks. After a notebook
exists, edit its CONFIG cell directly; the registry is only the starting point.

reach_zone values were PROPOSED automatically (fmpipe.reaches.propose_zones: Sept 16 zone
shifted by the difference in where the paw ends its reaches). Adjust after looking at plots.
"""

DLC = r"C:\Users\Shermanlab\Desktop\4103-Kevin-2026-09-22\videos"

SESSIONS = [
    dict(session="2026-09-14_15-40-28", experiment=1, second_cam_hz=120.19,
         dlc=rf"{DLC}\14092026-2_paw_angledDLC_Resnet50_4103Sep22shuffle3_snapshot_best-130.h5",
         tail_missing=0, zone=(411, 321)),
    dict(session="2026-09-15_14-22-37", experiment=1, second_cam_hz=120.19,
         dlc=rf"{DLC}\15092026-1_paw_angledDLC_Resnet50_4103Sep22shuffle2_snapshot_best-100.h5",
         tail_missing=0, zone=(311, 328)),
    dict(session="2026-09-16_13-21-12", experiment=2, second_cam_hz=120.19,
         dlc=rf"{DLC}\16092026-3_paw_angledDLC_Resnet50_4103Sep22shuffle1_snapshot_best-150.h5",
         tail_missing=0, zone=(400, 380)),          # Kevin's zone from watching the video
    dict(session="2026-09-17_12-22-54", experiment=1, second_cam_hz=120.19,
         dlc=rf"{DLC}\17092026-3_paw_angledDLC_Resnet50_4103Sep22shuffle3_snapshot_best-130.h5",
         tail_missing=0, zone=(287, 318)),
    dict(session="2026-09-18_11-22-28", experiment=1, second_cam_hz=200.0,
         dlc=rf"{DLC}\18092026-2_paw_angledDLC_Resnet50_4103Sep22shuffle3_snapshot_best-130.h5",
         tail_missing=100, zone=(281, 321)),        # disk filled: last 77 frames never written
]
