"""
Cross-day matching, step 4: tracking units across several days at a chosen confidence level.

Pairwise one-to-one matches (every pair of days) with posterior >= `level` are linked into
chains, strongest link first, and a link is refused if it would put two units of the same day
into one chain. Each chain reports

  n_days        days the neuron was found on (after pruning to a mutually consistent set)
  n_days_before_pruning   days linked before pruning (shows how often links alone over-extend a chain)
  min_link_p    weakest direct link used to build the chain
  min_pair_p    weakest posterior among ALL day pairs of the chain (including pairs not used to build it);
                by construction >= the confidence level: EVERY pair of days in the chain passes it
  mean_pair_p   mean of those posteriors
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def _pair_lookup(results: dict):
    """(day_a, i, day_b, j) -> posterior for every candidate pair (not only assigned ones)."""
    look = {}
    for (a, b), r in results.items():
        for i, j, p in zip(r.pairs.i.values, r.pairs.j.values, r.pairs.p.values):
            look[(a, int(i), b, int(j))] = float(p)
            look[(b, int(j), a, int(i))] = float(p)
    return look


def build_chains(results: dict, days: list, level: float) -> pd.DataFrame:
    edges = []
    for (a, b), r in results.items():
        t = r.pairs[r.pairs.assigned & (r.pairs.p >= level)]
        edges += [(float(p), (a, int(i)), (b, int(j))) for i, j, p in zip(t.i, t.j, t.p)]
    edges.sort(key=lambda e: -e[0])

    parent, members, min_link = {}, {}, {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, u, v in edges:
        for x in (u, v):
            if x not in parent:
                parent[x] = x; members[x] = {x}; min_link[x] = 1.0
    for p, u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            min_link[ru] = min(min_link[ru], p)
            continue
        days_u = {d for d, _ in members[ru]}; days_v = {d for d, _ in members[rv]}
        if days_u & days_v:                      # would put two units of one day in one chain
            continue
        parent[rv] = ru
        members[ru] |= members[rv]
        min_link[ru] = min(min_link[ru], min_link[rv], p)

    look = _pair_lookup(results)

    def pair_ps(m):
        return {(x, y): look.get((x[0], x[1], y[0], y[1]), 0.0) for x, y in itertools.combinations(m, 2)}

    rows = []
    for root in {find(x) for x in parent}:
        m = sorted(members[root], key=lambda t: days.index(t[0]))
        n_built = len(m)
        # Every pair of days in a chain must itself reach `level`. A chain built link by link can contain
        # days that do not match each other, so drop the member with the weakest agreement until it is consistent.
        while len(m) > 1:
            pp = pair_ps(m)
            if min(pp.values()) >= level:
                break
            support = {x: sum(v for (a, b), v in pp.items() if x in (a, b)) / (len(m) - 1) for x in m}
            m.remove(min(m, key=lambda x: support[x]))
        if len(m) < 2:
            continue
        pp = list(pair_ps(m).values())
        rows.append(dict(n_days=len(m), n_days_before_pruning=n_built, days=",".join(d for d, _ in m),
                         members={d: i for d, i in m}, min_link_p=min_link[root],
                         min_pair_p=float(min(pp)), mean_pair_p=float(np.mean(pp))))
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["n_days", "min_pair_p"], ascending=False).reset_index(drop=True)
        out.insert(0, "chain", np.arange(len(out)))
    return out


def build_sequential_chains(results: dict, days: list, level: float, max_gap: int = 0) -> pd.DataFrame:
    """
    Tracking that leans on CONSECUTIVE days: a neuron is followed day -> next day, so slow changes in its
    waveform or probe position are tolerated even if the first and last day look less alike.
    Links are one-to-one matches with posterior >= level between consecutive days (and, with max_gap=1, between
    days two apart when the day in between has no match: a one-day gap). Non-adjacent pairs are NOT required to
    pass the level; their direct posterior is reported (`min_nonadjacent_p`) as supporting evidence only.
    """
    idx = {d: k for k, d in enumerate(days)}
    edges = []
    for (a, b), r in results.items():
        gap = abs(idx[b] - idx[a]) - 1
        if gap > max_gap:
            continue
        t = r.pairs[r.pairs.assigned & (r.pairs.p >= level)]
        edges += [(gap, -float(p), (a, int(i)), (b, int(j))) for i, j, p in zip(t.i, t.j, t.p)]
    edges.sort(key=lambda e: (e[0], e[1]))           # adjacent days first, then strongest first

    parent, members, links = {}, {}, {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for _, _, u, v in edges:
        for x in (u, v):
            if x not in parent:
                parent[x] = x; members[x] = {x}; links[x] = []
    for gap, negp, u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if {d for d, _ in members[ru]} & {d for d, _ in members[rv]}:
            continue
        parent[rv] = ru; members[ru] |= members[rv]; links[ru] = links[ru] + links[rv] + [(gap, -negp)]

    look = _pair_lookup(results)
    rows = []
    for root in {find(x) for x in parent}:
        m = sorted(members[root], key=lambda t: idx[t[0]])
        ds = [idx[d] for d, _ in m]
        non_adj = [look.get((x[0], x[1], y[0], y[1]), 0.0) for x, y in itertools.combinations(m, 2)
                   if idx[y[0]] - idx[x[0]] > 1]
        rows.append(dict(n_days=len(m), days=",".join(d for d, _ in m), members={d: i for d, i in m},
                         contiguous=bool(ds == list(range(ds[0], ds[-1] + 1))),
                         span_days=ds[-1] - ds[0] + 1,
                         min_link_p=float(min(p for _, p in links[root])) if links[root] else np.nan,
                         min_nonadjacent_p=float(min(non_adj)) if non_adj else np.nan,
                         mean_nonadjacent_p=float(np.mean(non_adj)) if non_adj else np.nan))
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["n_days", "min_link_p"], ascending=False).reset_index(drop=True)
        out.insert(0, "chain", np.arange(len(out)))
    return out


def sequential_summary(results: dict, days: list, levels=(0.6, 0.7, 0.8, 0.9, 0.95), max_gap=0):
    tables, rows = {}, []
    for L in levels:
        c = build_sequential_chains(results, days, L, max_gap)
        tables[L] = c
        rows.append(dict(level=L, **{f">={k} days": int((c.n_days >= k).sum()) if len(c) else 0
                                     for k in range(2, len(days) + 1)}))
    return tables, pd.DataFrame(rows)


def chains_at_levels(results: dict, days: list, levels=(0.6, 0.7, 0.8, 0.9, 0.95)):
    """-> (dict level -> chains table, summary table of how many neurons are tracked over >= k days)."""
    tables, rows = {}, []
    for L in levels:
        c = build_chains(results, days, L)
        tables[L] = c
        rows.append(dict(level=L, **{f">={k} days": int((c.n_days >= k).sum()) if len(c) else 0
                                     for k in range(2, len(days) + 1)}))
    return tables, pd.DataFrame(rows)
