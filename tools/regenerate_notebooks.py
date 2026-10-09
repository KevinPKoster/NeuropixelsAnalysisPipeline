"""
(Re)create the per-day notebooks from tools/sessions.py.

    python tools/regenerate_notebooks.py            # only creates notebooks that do not exist
    python tools/regenerate_notebooks.py --force    # overwrite (discards edits made in Jupyter!)
    python tools/regenerate_notebooks.py --only 2026-09-16
"""

import argparse
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).parent))
from new_session_notebook import OUT_DIR, build        # noqa: E402
from sessions import SESSIONS                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true")
ap.add_argument("--only", default=None, help="session date prefix, e.g. 2026-09-16")
a = ap.parse_args()

OUT_DIR.mkdir(parents=True, exist_ok=True)
for s in SESSIONS:
    if a.only and not s["session"].startswith(a.only):
        continue
    out = OUT_DIR / f"{s['session'][:10]}.ipynb"
    if out.exists() and not a.force:
        print(f"skip (exists, use --force to overwrite): {out.name}")
        continue
    cams = ('{"paw_angled": dict(pin=0, trigger_hz=200.0),\n'
            f'                 "bottom": dict(pin=1, trigger_hz={s["second_cam_hz"]})}}')
    dlc = f'{{"paw_angled": r"{s["dlc"]}"}}' if s.get("dlc") else "{}"
    tail = f'{{"paw_angled": {s["tail_missing"]}}}' if s.get("tail_missing") else "{}"
    zone = f'{{"x_min": {s["zone"][0]}, "y_min": {s["zone"][1]}}}' if s.get("zone") else "None"
    nb = build(s["session"], s["experiment"], "K:/", cams, dlc, tail, zone)
    nbf.write(nb, out)
    print("wrote", out.name)
