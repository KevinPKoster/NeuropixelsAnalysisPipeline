"""
Cross-day matching, step 6: run everything for a set of days.

    out = match_all(cfgs, out_dir)

* builds profiles for every day (raw waveforms must exist: `extract_waveforms`)
* learns the 'different neuron' reference from all days, matches EVERY pair of days
* estimates false-match rates with decoys, builds tracked neurons at several confidence levels
* saves tables (CSV/JSON) and a pickle that the plotting functions can reload
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from ..config import write_json
from .chains import chains_at_levels
from .features import build_profiles
from .model import decoy_fdr, match_day_pair, reference_densities

LEVELS = (0.6, 0.7, 0.8, 0.9, 0.95)


def match_all(cfgs: dict, out_dir, levels=LEVELS, verbose=True) -> dict:
    """cfgs: {day name: SessionConfig}, in chronological order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    days = list(cfgs)
    profiles = {d: build_profiles(c, d) for d, c in cfgs.items()}
    f0, f_same, lo, hi, neg, pos = reference_densities(list(profiles.values()))
    if verbose:
        print(f"reference: {len(neg)} different-unit neighbour pairs, {len(pos)} same-unit split-half pairs")

    results, fdr = {}, {}
    for ia, a in enumerate(days):
        for b in days[ia + 1:]:
            r = match_day_pair(profiles[a], profiles[b], f0, lo, hi)
            results[(a, b)] = r
            fdr[(a, b)] = decoy_fdr(r, levels)
            n8 = int((r.pairs.assigned & (r.pairs.p >= 0.8)).sum())
            if verbose:
                print(f"{a} vs {b}: drift {r.drift_um:+6.1f} um | {n8:3d} matches at p>=0.8 "
                      f"({int((r.pairs.assigned & (r.pairs.p >= 0.6)).sum())} at >=0.6)")
            r.pairs.assign(day_a=a, day_b=b,
                           unit_a=profiles[a].unit_ids[r.pairs.i.values],
                           unit_b=profiles[b].unit_ids[r.pairs.j.values]).to_csv(out_dir / f"pairs_{a}_vs_{b}.csv", index=False)
            fdr[(a, b)].to_csv(out_dir / f"fdr_{a}_vs_{b}.csv", index=False)

    tables, summary = chains_at_levels(results, days, levels)
    for L, t in tables.items():
        if len(t):
            t2 = t.copy()
            t2["members"] = [{d: int(profiles[d].unit_ids[i]) for d, i in m.items()} for m in t.members]
            t2.to_csv(out_dir / f"tracked_neurons_p{L:.2f}.csv", index=False)
    summary.to_csv(out_dir / "tracked_summary.csv", index=False)
    write_json(out_dir / "matching_summary.json", dict(
        days=days, levels=list(levels),
        drift_um={f"{a}__{b}": r.drift_um for (a, b), r in results.items()},
        prior_pi={f"{a}__{b}": r.pi for (a, b), r in results.items()},
        n_units={d: p.n for d, p in profiles.items()},
        tracked=summary.to_dict(orient="records")))
    out = dict(days=days, profiles=profiles, results=results, fdr=fdr, chains=tables, summary=summary,
               reference=dict(neg=neg, pos=pos))
    with open(out_dir / "matching_results.pkl", "wb") as f:
        pickle.dump(out, f)
    return out


def load_matching(out_dir) -> dict:
    with open(Path(out_dir) / "matching_results.pkl", "rb") as f:
        return pickle.load(f)
