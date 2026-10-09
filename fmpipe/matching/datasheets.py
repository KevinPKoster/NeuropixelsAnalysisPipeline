"""
Cross-day matching, step 7: an Excel workbook with everything needed to work from the tables.

Sheets
  README                         definitions of every column and rule
  Summary                        neurons tracked on >= k days per confidence level (strict and sequential rules)
  Matches_per_pair               matched neurons per pair of days per confidence level
  Decoy_false_match_rate         decoy-estimated false-match rate per pair and level
  Drift                          estimated depth drift per pair of days
  Strict_p<level> / Seq_p<level> one row per tracked neuron: unit id, region, depth, rate on each day
  ..._resp                       the same rows plus each day's reach response (DOWNSTREAM information that was
                                 never used for matching)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .chains import chains_at_levels, sequential_summary

README = [
    "Cross-day unit matching, mouse 4103",
    "",
    "Matching uses ONLY: raw waveform footprint on physical electrodes, amplitude, depth and lateral position (after drift",
    "correction), autocorrelogram and ISI shape, firing rate. No behaviour (reaches, PETH) enters the matching.",
    "",
    "posterior: probability that two clusters on different days are the same neuron (naive-Bayes mixture, one partner per unit).",
    "level: minimum posterior required of the links (see rules).",
    "Strict rule (sheets Strict_p*): EVERY pair of days in the chain has posterior >= level (chains pruned to a mutually",
    "  consistent set). min_pair_p is the weakest pair; n_days_before_pruning shows how many days were linked before pruning.",
    "Sequential rule (sheets Seq_p*): consecutive days are linked (one-day gaps allowed: contiguous = False) with posterior",
    "  >= level; non-adjacent pairs are not required to pass. min_link_p = weakest link; min_nonadjacent_p = weakest direct",
    "  posterior among non-adjacent pairs (low values mean the two ends look different: slow change or an accumulated error).",
    "Decoy false-match rate: matching repeated with the other day displaced by 160-320 um (these pairs cannot be the same",
    "  neuron); the fraction of decoy 'matches' relative to real matches estimates the false-match rate at that level.",
    "Columns per day: <day>_unit (Phy cluster id), _region (CFA/WM/VAL from recorded depth), _dv_um, _rate_hz.",
    "Sheets ending in _resp add per day: _direction (excited/suppressed/none), _peak_z, _peak_latency_s from the reach-bout PETH",
    "  (shuffle-calibrated). These were NOT used for matching.",
]


def tracked_table(chains, profiles, days, resp=None):
    rows = []
    for _, ch in chains.iterrows():
        r = dict(chain=int(ch.chain), n_days=int(ch.n_days), days=ch.days)
        for col in ("n_days_before_pruning", "contiguous", "min_pair_p", "mean_pair_p", "min_link_p",
                    "min_nonadjacent_p", "mean_nonadjacent_p"):
            if col in ch.index:
                r[col] = ch[col] if not isinstance(ch[col], float) else round(ch[col], 4)
        dirs = []
        for d in days:
            P = profiles[d]
            if d in ch.members:
                i = ch.members[d]
                uid = int(P.unit_ids[i])
                r[f"{d}_unit"] = uid; r[f"{d}_region"] = P.region[i]
                r[f"{d}_dv_um"] = round(float(P.dv_um[i])); r[f"{d}_rate_hz"] = round(float(P.rate[i].mean()), 2)
                if resp is not None:
                    t = resp[d]
                    if uid in t.index:
                        r[f"{d}_direction"] = t.loc[uid, "direction"]; r[f"{d}_peak_z"] = round(float(t.loc[uid, "peak_z"]), 2)
                        r[f"{d}_peak_latency_s"] = round(float(t.loc[uid, "peak_latency_s"]), 3)
                        dirs.append(t.loc[uid, "direction"])
            else:
                for k in ("unit", "region", "dv_um", "rate_hz") + (("direction", "peak_z", "peak_latency_s") if resp is not None else ()):
                    r[f"{d}_{k}"] = None
        if resp is not None:
            mod = [x for x in dirs if x != "none"]
            r["n_days_modulated"] = len(mod)
            r["direction_consistent"] = (len(set(mod)) == 1) if len(mod) >= 2 else None
        rows.append(r)
    return pd.DataFrame(rows)


def write_datasheets(out: dict, cfgs: dict, path, levels=(0.6, 0.7, 0.8, 0.9, 0.95), subset="all"):
    days, profiles, results = out["days"], out["profiles"], out["results"]
    resp = {d: pd.read_csv(c.analysis_dir / "peth" / f"reach_peth_{subset}_units.csv").set_index("id") for d, c in cfgs.items()}

    strict_tabs, strict_sum = chains_at_levels(results, days, levels)
    seq_tabs, seq_sum = sequential_summary(results, days, levels, max_gap=1)
    seq0_tabs, seq0_sum = sequential_summary(results, days, levels, max_gap=0)
    summary = pd.concat([strict_sum.assign(rule="strict (every pair passes)"),
                         seq0_sum.assign(rule="sequential (consecutive days, no gap)"),
                         seq_sum.assign(rule="sequential (consecutive days, 1-day gap allowed)")], ignore_index=True)
    summary = summary[["rule"] + [c for c in summary.columns if c != "rule"]]

    pair_rows, fdr_rows, drift_rows = [], [], []
    for (a, b), r in results.items():
        a_ = r.pairs[r.pairs.assigned]
        pair_rows.append(dict(pair=f"{a} vs {b}", units_a=r.n_a, units_b=r.n_b,
                              **{f"matches p>={L}": int((a_.p >= L).sum()) for L in levels}))
        f = out["fdr"][(a, b)]
        for _, row in f.iterrows():
            fdr_rows.append(dict(pair=f"{a} vs {b}", level=row.level, matches=int(row.n_matches),
                                 mean_decoy_matches=round(row.mean_decoy_matches, 2), decoy_false_match_rate=round(row.est_fdr, 4),
                                 model_claimed_false_match_rate=round(row.model_fdr, 4)))
        drift_rows.append(dict(pair=f"{a} vs {b}", drift_um=round(r.drift_um, 1), prior_match_fraction=round(r.pi, 4)))

    path = Path(path)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"README": README}).to_excel(xw, "README", index=False)
        summary.to_excel(xw, "Summary", index=False)
        pd.DataFrame(pair_rows).to_excel(xw, "Matches_per_pair", index=False)
        pd.DataFrame(fdr_rows).to_excel(xw, "Decoy_false_match_rate", index=False)
        pd.DataFrame(drift_rows).to_excel(xw, "Drift", index=False)
        for L in levels:
            for name, tabs in (("Strict", strict_tabs), ("Seq", seq_tabs)):
                t = tabs[L]
                if len(t) == 0:
                    continue
                tracked_table(t, profiles, days).to_excel(xw, f"{name}_p{L:.2f}", index=False)
                tracked_table(t, profiles, days, resp).to_excel(xw, f"{name}_p{L:.2f}_resp", index=False)
        for ws in xw.book.worksheets:                       # readable: bold header, frozen, widths
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:60])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60 if ws.title != "README" else 120)
    return path
