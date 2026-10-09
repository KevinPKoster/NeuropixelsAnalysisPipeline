"""
Create the cross-day matching notebook:

    python tools/make_matching_notebook.py [--force]

Writes 'Anchored analysis/Freely moving analysis/Mouse 4103/cross_day_matching.ipynb'.
The matching itself uses only waveforms, positions and spike trains (fmpipe.matching never
imports behaviour/PETH code). Reach responses of tracked neurons are looked at only in the
separate notebook section marked DOWNSTREAM, after confidence levels are fixed.
"""

import argparse
import sys
from pathlib import Path

import nbformat as nbf

OUT = (Path(__file__).resolve().parents[1] / "Anchored analysis" / "Freely moving analysis" / "Mouse 4103"
       / "cross_day_matching.ipynb")

MD0 = """# Cross-day unit matching - mouse 4103, Sept 15-18 (training window)

**What this does.** For every pair of days it asks, for every good unit, "is this the same neuron as any
unit on the other day?", and returns a probability. Neurons are then followed across several days at
chosen confidence levels (posterior of every link >= 0.6 / 0.7 / 0.8 / 0.9 / 0.95).

**What it uses** (nothing behavioural): the spatial waveform footprint on the physical electrodes
(raw waveforms, two halves of each recording), depth and lateral position after drift correction,
autocorrelogram and ISI shape, firing rate.

**How the probabilities are made honest.**
* *Different neuron* reference = pairs of different units from the same recording that sit close together on the probe.
* *Same neuron* is learned from the cross-day data (EM), starting from obvious matches; each unit gets at most one partner.
* **Decoy test:** the matching is repeated with the other day displaced by 160-320 um; any "match" it finds there is false,
  which measures the false-match rate at each level directly (panel C of the overview).

**Reading the plots.** Pair pages show the two units' best-channel waveforms, amplitude along the probe, autocorrelograms
and probe positions overlaid, plus bars showing which features argued for (green) or against (red) the match."""

SETUP = '''import sys
from pathlib import Path
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "fmpipe").is_dir())
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt
import pandas as pd
from fmpipe import SessionConfig
from fmpipe.matching import (extract_waveforms, match_all, load_matching, plot_pair_pages,
                             plot_chain_pages, plot_overview)'''

CONFIG = '''# --- CONFIG -------------------------------------------------------------------
ROOTS = (r"C:\\Users\\Shermanlab\\Documents\\Open Ephys", "Z:/")
def cfg(session, experiment=1):
    return SessionConfig(data_root="K:/", session=session, experiment=experiment, ks_search_roots=ROOTS)

DAYS = {                                   # label -> session (in chronological order)
    # Sept 14 is left out: a short, differently structured (static-shaping) session with only 112 good units;
    # Sept 15-18 are the training window analysed here.
    "09-15": cfg("2026-09-15_14-22-37"),
    "09-16": cfg("2026-09-16_13-21-12", experiment=2),
    "09-17": cfg("2026-09-17_12-22-54"),
    "09-18": cfg("2026-09-18_11-22-28"),
}
OUT_DIR = r"D:\\Neuropixels data-motor project\\unit_matching\\4103_sept15-18"   # tables, PDFs, workbook, pickle
LEVELS = (0.6, 0.7, 0.8, 0.9, 0.95)
SHOW_LEVEL = 0.8                           # level used for the example pages below'''

STEP1 = '''# --- step 1: raw mean waveforms (two halves) of every good unit (~4 min per day, skipped if done) ---
for d, c in DAYS.items():
    extract_waveforms(c)'''

STEP2 = '''# --- step 2: match every pair of days, estimate false-match rates, track neurons across days ---
out = match_all(DAYS, OUT_DIR, levels=LEVELS)
days, profiles, results = out["days"], out["profiles"], out["results"]
out["summary"]'''

STEP3 = '''# --- overview: matches per pair of days, tracked neurons per confidence level, calibration, drift ---
plot_overview(results, days, out["summary"], out["fdr"], level_for_matrix=SHOW_LEVEL,
              save_path=OUT_DIR + r"\\overview.png"); plt.show()'''

STEP4 = '''# --- false-match rate at each confidence level (decoy test), per pair of days ---
fdr = pd.concat({f"{a} vs {b}": f for (a, b), f in out["fdr"].items()}, names=["days"]).reset_index(level=0)
fdr.pivot(index="days", columns="level", values="est_fdr").round(3)'''

STEP5 = '''# --- example matched pairs for one pair of days (the full set is saved as a PDF) ---
A, B = "09-15", "09-16"
figs = plot_pair_pages(results[(A, B)], profiles[A], profiles[B], pdf_path=OUT_DIR + rf"\\pairs_{A}_vs_{B}.pdf",
                       min_p=SHOW_LEVEL, per_page=3)
for f in figs[:2]:
    plt.figure(f.number); plt.show()'''

STEP6 = '''# --- neurons tracked across >= 3 days at SHOW_LEVEL (full set saved as PDF) ---
ch = out["chains"][SHOW_LEVEL]
print(ch.groupby("n_days").size().rename("neurons tracked").to_frame().T.to_string())
figs = plot_chain_pages(ch, profiles, results, days, pdf_path=OUT_DIR + rf"\\tracked_neurons_p{SHOW_LEVEL:.2f}.pdf", min_days=3)
for f in figs[:3]:
    plt.figure(f.number); plt.show()'''

STEP7 = '''# --- the neurons followed across ALL days, at the most permissive level that finds any ---
for L in LEVELS:
    five = out["chains"][L]
    five = five[five.n_days == len(days)] if len(five) else five
    print(f"level {L}: {len(five)} neurons found on all {len(days)} days")
    if len(five):
        figs = plot_chain_pages(out["chains"][L], profiles, results, days,
                                pdf_path=OUT_DIR + rf"\\neurons_on_all_days_p{L:.2f}.pdf", min_days=len(days))
        for f in figs[:3]:
            plt.figure(f.number); plt.show()
        break'''

STEP8 = '''# --- SEQUENTIAL tracking: follow each neuron day -> next day (a one-day gap is allowed). Slow changes are tolerated,
# non-adjacent pairs are not required to pass the level (their direct posterior is shown on the pages). ---
from fmpipe.matching.chains import sequential_summary
seq_tabs, seq_sum = sequential_summary(results, days, LEVELS, max_gap=1)
for L, t in seq_tabs.items():
    if len(t):
        t2 = t.copy(); t2["members"] = [{d: int(profiles[d].unit_ids[i]) for d, i in m.items()} for m in t.members]
        t2.to_csv(OUT_DIR + rf"\\tracked_sequential_p{L:.2f}.csv", index=False)
seq_sum'''

STEP9 = '''# --- neurons tracked on all 5 days by the sequential rule (look at the right-hand matrix: weak non-adjacent pairs
# mean the neuron looks different at the two ends, which is either slow change or a wrong link) ---
L = 0.8
figs = plot_chain_pages(seq_tabs[L], profiles, results, days, pdf_path=OUT_DIR + rf"\\sequential_5day_p{L:.2f}.pdf", min_days=len(days))
for f in figs[:3]:
    plt.figure(f.number); plt.show()'''

STEP10 = '''# --- Excel workbook: summary, matches per pair, false-match rates, drift, one sheet per rule and level ---
from fmpipe.matching import write_datasheets
xlsx = write_datasheets(out, DAYS, OUT_DIR + r"\\\\tracked_neurons_sept15-18.xlsx", levels=LEVELS)
print("workbook:", xlsx)'''

STEP11 = '''# --- DOWNSTREAM: reach responses of the tracked neurons (never used for matching) ---
from fmpipe import plot_tracked_probe
for rule, tabs in (("strict", out["chains"]), ("sequential", seq_tabs)):
    for L in (0.8,):
        for theme in ("dark", "light"):
            f = plot_tracked_probe(DAYS, profiles, results, days, tabs[L], theme=theme, min_days=3,
                                   title=f"Mouse 4103 | neurons tracked on >= 3 days ({rule} rule, level {L})",
                                   save_path=OUT_DIR + rf"\\\\tracked_probe_{rule}_p{L:.2f}_{theme}.png")
            if theme == "dark":
                plt.figure(f.number); plt.show()'''

MD_DOWN = """## DOWNSTREAM (after the confidence levels above are fixed)
Anything that looks at reach responses of the tracked neurons belongs below this line and must not feed back into
the matching. The tables `tracked_neurons_p*.csv` in the output folder give, for every tracked neuron, its unit id on each day;
`peth/reach_peth_*_units.csv` in each day's analysis folder has the responses."""


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(MD0), nbf.v4.new_code_cell(SETUP), nbf.v4.new_code_cell(CONFIG),
                nbf.v4.new_code_cell(STEP1), nbf.v4.new_code_cell(STEP2), nbf.v4.new_code_cell(STEP3),
                nbf.v4.new_code_cell(STEP4), nbf.v4.new_code_cell(STEP5), nbf.v4.new_code_cell(STEP6),
                nbf.v4.new_code_cell(STEP7), nbf.v4.new_code_cell(STEP8), nbf.v4.new_code_cell(STEP9),
                nbf.v4.new_code_cell(STEP10), nbf.v4.new_markdown_cell(MD_DOWN), nbf.v4.new_code_cell(STEP11)]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if OUT.exists() and not a.force:
        sys.exit(f"{OUT.name} exists; use --force to overwrite (discards edits made in Jupyter).")
    nbf.write(build(), OUT)
    print("wrote", OUT)
