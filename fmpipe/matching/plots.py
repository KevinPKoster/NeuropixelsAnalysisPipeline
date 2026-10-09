"""
Cross-day matching, step 5: figures in the style of the old matching notebook, rebuilt for the
probabilistic matcher. None of them shows or uses reach/PETH data.

  plot_pair_pages     per matched pair: waveform, spatial footprint, ACG, probe position, evidence bars
  plot_chain_pages    per tracked neuron: waveforms/ACGs on every day, depth over days, posterior matrix
  plot_overview       how many matches / tracked neurons at each confidence level, calibration, drift
"""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .features import HALF_ROWS, ROW_UM, DayProfiles
from .model import FEATURES, feature_evidence

DAY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
FEATURE_LABELS = {
    "wf_cos": "waveform footprint", "amp_abs_logratio": "amplitude", "dy_abs": "depth offset",
    "dx_abs": "lateral offset", "acg_mid_r": "ACG shape", "isi_r": "ISI shape",
    "rate_abs_logratio": "firing rate",
}


def _dv_center(P: DayProfiles):
    """Depth (DV um) of each unit's amplitude-weighted centre."""
    return P.dv_um - (P.cent_y.mean(1) - P.peak_row * ROW_UM)


def _footprint(P, i, drift=0.0):
    """Amplitude (peak-to-peak) along depth for unit i: (DV um positions, amplitude), drift-shifted into day A's frame."""
    w = P.win[i].mean(0)                                  # (17, 2, T)
    amp = np.ptp(w, axis=-1).max(axis=1)                  # (17,)
    k = np.arange(-HALF_ROWS, HALF_ROWS + 1)
    dv = P.dv_um[i] - k * ROW_UM + drift
    return dv, amp


def _label(P, i):
    return f"{P.name} unit {P.unit_ids[i]} ({P.region[i]}, DV {P.dv_um[i]:.0f})"


def _pair_row(fig, gs_row, A, ia, B, jb, res, row, ev, level_txt):
    t_ms = (np.arange(A.peak_ch_trace.shape[-1]) - 40) / 30.0
    colA, colB = DAY_COLORS[0], DAY_COLORS[1]
    ax = fig.add_subplot(gs_row[0])
    ax.plot(t_ms, A.peak_ch_trace[ia].mean(0), color=colA, label=A.name)
    ax.plot(t_ms, B.peak_ch_trace[jb].mean(0), color=colB, label=B.name)
    ax.axhline(0, c="0.8", lw=0.5); ax.set_title("waveform, each unit's best channel", fontsize=8)
    ax.set_xlabel("ms", fontsize=7); ax.set_ylabel("uV", fontsize=7); ax.legend(fontsize=6); ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs_row[1])
    dvA, aA = _footprint(A, ia); dvB, aB = _footprint(B, jb, res.drift_um)
    ax.plot(aA, dvA, "-o", ms=3, color=colA); ax.plot(aB, dvB, "-o", ms=3, color=colB)
    ax.invert_yaxis(); ax.set_title("amplitude along the probe", fontsize=8)
    ax.set_xlabel("peak-to-peak (uV)", fontsize=7); ax.set_ylabel("DV (um)", fontsize=7); ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs_row[2])
    lag = np.arange(1, 101)
    ax.plot(lag, A.acg_mid[ia].mean(0), color=colA); ax.plot(lag, B.acg_mid[jb].mean(0), color=colB)
    ax.set_title(f"autocorrelogram (r = {row.acg_mid_r:.2f})", fontsize=8)
    ax.set_xlabel("lag (ms)", fontsize=7); ax.set_ylabel("Hz", fontsize=7); ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs_row[3])
    dvc_A, dvc_B = _dv_center(A), _dv_center(B) + res.drift_um
    cA, cB = dvc_A[ia], dvc_B[jb]
    mA = np.abs(dvc_A - cA) < 200; mB = np.abs(dvc_B - cA) < 200
    ax.scatter(A.cent_x.mean(1)[mA], dvc_A[mA], s=14, facecolors="none", edgecolors=colA, alpha=0.5, label=f"{A.name}")
    ax.scatter(B.cent_x.mean(1)[mB], dvc_B[mB], s=14, marker="x", color=colB, alpha=0.5, label=f"{B.name}")
    xa, xb = A.cent_x.mean(1)[ia], B.cent_x.mean(1)[jb]
    ax.plot([xa, xb], [cA, cB], "k-", lw=1.2)
    ax.scatter([xa], [cA], s=70, color=colA, zorder=5); ax.scatter([xb], [cB], s=70, marker="X", color=colB, zorder=5)
    ax.set_ylim(cA + 200, cA - 200); ax.set_xlim(0, 70)
    ax.set_title(f"probe position (offset {row.dy_abs:.1f} um)", fontsize=8)
    ax.set_xlabel("x (um)", fontsize=7); ax.set_ylabel("DV (um)", fontsize=7); ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs_row[4])
    colors = ["#2e7d32" if v > 0 else "#c62828" for v in ev]
    ax.barh(range(len(ev)), ev, color=colors)
    ax.set_yticks(range(len(ev))); ax.set_yticklabels([FEATURE_LABELS[FEATURES[c]] for c in res.f1.cols], fontsize=6)
    ax.invert_yaxis(); ax.axvline(0, c="k", lw=0.6)
    ax.set_title("evidence for 'same neuron'\n(log-likelihood ratio)", fontsize=8); ax.tick_params(labelsize=6)
    return fig.axes[-5:]


def _row_header(fig, axes, text, pad=0.052):
    """Bold header centred above a row of axes (anchored to the row, so it never collides with titles)."""
    top = max(a.get_position().y1 for a in axes)
    fig.text(0.5, top + pad, text, ha="center", va="bottom", fontsize=9, fontweight="bold")


def plot_pair_pages(res, A: DayProfiles, B: DayProfiles, pdf_path=None, min_p=0.6, per_page=3, max_pairs=None):
    """One row per assigned pair (strongest first). Returns the list of figures; also saves a PDF if given."""
    sel = res.pairs[res.pairs.assigned & (res.pairs.p >= min_p)].sort_values("p", ascending=False)
    if max_pairs:
        sel = sel.head(max_pairs)
    if len(sel) == 0:
        return []
    ev = feature_evidence(res, sel)
    figs, pdf = [], (PdfPages(pdf_path) if pdf_path else None)
    for page in range(int(np.ceil(len(sel) / per_page))):
        part = sel.iloc[page * per_page:(page + 1) * per_page]
        fig = plt.figure(figsize=(17, 3.6 * per_page))
        outer = fig.add_gridspec(per_page, 1, hspace=0.95, top=0.86, bottom=0.07)
        for r, (k, row) in enumerate(part.iterrows()):
            gs = outer[r].subgridspec(1, 5, wspace=0.42, width_ratios=[1, 1, 1, 1, 1.15])
            axes = _pair_row(fig, gs, A, int(row.i), B, int(row.j), res, row, ev[page * per_page + r], "")
            _row_header(fig, axes, f"{_label(A, int(row.i))}  <->  {_label(B, int(row.j))}   |   posterior {row.p:.3f}   "
                        f"| waveform cos {row.wf_cos:.2f}, depth offset {row.dy_abs:.1f} um, ACG r {row.acg_mid_r:.2f}")
        fig.suptitle(f"{A.name} vs {B.name}: matched pairs {page * per_page + 1}-{page * per_page + len(part)} of {len(sel)} "
                     f"(drift {res.drift_um:+.0f} um)", y=0.985, fontsize=12)
        if pdf is not None:
            pdf.savefig(fig, bbox_inches="tight")
        figs.append(fig)
    if pdf is not None:
        pdf.close()
    return figs


def plot_chain_pages(chains, profiles: dict, results: dict, days: list, pdf_path=None, min_days=3, max_chains=None, per_page=2):
    """One row per tracked neuron: how it looks on every day it was found, and how confident each pairing is."""
    sel = chains[chains.n_days >= min_days]
    if max_chains:
        sel = sel.head(max_chains)
    figs, pdf = [], (PdfPages(pdf_path) if pdf_path else None)
    if len(sel) == 0:
        return figs
    look = {}
    for (a, b), r in results.items():
        for i, j, p in zip(r.pairs.i, r.pairs.j, r.pairs.p):
            look[(a, int(i), b, int(j))] = look[(b, int(j), a, int(i))] = float(p)
    t_ms = (np.arange(121) - 40) / 30.0
    for page in range(int(np.ceil(len(sel) / per_page))):
        part = sel.iloc[page * per_page:(page + 1) * per_page]
        fig = plt.figure(figsize=(17, 4.1 * per_page))
        outer = fig.add_gridspec(per_page, 1, hspace=0.85, top=0.88, bottom=0.07)
        for r, (_, ch) in enumerate(part.iterrows()):
            gs = outer[r].subgridspec(1, 5, wspace=0.4)
            n_before = len(fig.axes)
            mem = ch.members
            ds = [d for d in days if d in mem]
            first = ds[0]
            ax = fig.add_subplot(gs[0])
            for d in ds:
                P = profiles[d]; ax.plot(t_ms, P.peak_ch_trace[mem[d]].mean(0), color=DAY_COLORS[days.index(d)], label=d)
            ax.set_title("best-channel waveform", fontsize=8); ax.legend(fontsize=6); ax.set_xlabel("ms", fontsize=7); ax.tick_params(labelsize=6)
            ax = fig.add_subplot(gs[1])
            for d in ds:
                P = profiles[d]; drift = results[(first, d)].drift_um if (first, d) in results else 0.0
                dv, amp = _footprint(P, mem[d], drift)
                ax.plot(amp, dv, "-o", ms=3, color=DAY_COLORS[days.index(d)])
            ax.invert_yaxis(); ax.set_title("amplitude along the probe", fontsize=8); ax.set_xlabel("uV", fontsize=7); ax.tick_params(labelsize=6)
            ax = fig.add_subplot(gs[2])
            for d in ds:
                P = profiles[d]; ax.plot(np.arange(1, 101), P.acg_mid[mem[d]].mean(0), color=DAY_COLORS[days.index(d)])
            ax.set_title("autocorrelogram", fontsize=8); ax.set_xlabel("lag (ms)", fontsize=7); ax.tick_params(labelsize=6)
            ax = fig.add_subplot(gs[3])
            for d in ds:
                P = profiles[d]; drift = results[(first, d)].drift_um if (first, d) in results else 0.0
                ax.plot([days.index(d)], [_dv_center(P)[mem[d]] + drift], "o", ms=8, color=DAY_COLORS[days.index(d)])
            ax.set_xticks(range(len(days))); ax.set_xticklabels(days, fontsize=6); ax.invert_yaxis()
            ax.set_title(f"depth over days (day-{first} frame)", fontsize=8); ax.tick_params(labelsize=6)
            ax = fig.add_subplot(gs[4])
            M = np.full((len(ds), len(ds)), np.nan)
            for a_, b_ in itertools.combinations(range(len(ds)), 2):
                M[a_, b_] = M[b_, a_] = look.get((ds[a_], mem[ds[a_]], ds[b_], mem[ds[b_]]), 0.0)
            ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
            for a_, b_ in itertools.product(range(len(ds)), repeat=2):
                if a_ != b_:
                    ax.text(b_, a_, f"{M[a_, b_]:.2f}", ha="center", va="center", fontsize=7, color="w" if M[a_, b_] < 0.6 else "k")
            ax.set_xticks(range(len(ds))); ax.set_xticklabels(ds, fontsize=6); ax.set_yticks(range(len(ds))); ax.set_yticklabels(ds, fontsize=6)
            ax.set_title("posterior between every pair of days", fontsize=8)
            units_txt = "  ".join(f"{d}: u{profiles[d].unit_ids[mem[d]]} ({profiles[d].region[mem[d]]})" for d in ds)
            if "min_pair_p" in ch.index:
                strength = f"every pair of days >= {ch.min_pair_p:.2f}"
            else:                                  # sequential chains: consecutive days linked, others only reported
                strength = f"weakest consecutive link {ch.min_link_p:.2f}, weakest non-adjacent pair {ch.min_nonadjacent_p:.2f}"
            _row_header(fig, fig.axes[n_before:], f"tracked neuron {ch.chain}: found on {ch.n_days} days | {strength} | {units_txt}")
        fig.suptitle("Neurons tracked across days", y=0.985, fontsize=12)
        if pdf is not None:
            pdf.savefig(fig, bbox_inches="tight")
        figs.append(fig)
    if pdf is not None:
        pdf.close()
    return figs


def plot_overview(results: dict, days: list, summary, fdr_tables: dict, level_for_matrix=0.8, save_path=None):
    """Matches per day pair, tracked neurons vs confidence level, calibration check, drift."""
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    # A: matrix of matches at a level
    M = np.full((len(days), len(days)), np.nan)
    for (a, b), r in results.items():
        n = int((r.pairs.assigned & (r.pairs.p >= level_for_matrix)).sum())
        M[days.index(a), days.index(b)] = M[days.index(b), days.index(a)] = n
    im = ax[0, 0].imshow(M, cmap="Blues")
    for i, j in itertools.product(range(len(days)), repeat=2):
        if i != j:
            ax[0, 0].text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=11)
    ax[0, 0].set_xticks(range(len(days))); ax[0, 0].set_xticklabels(days); ax[0, 0].set_yticks(range(len(days))); ax[0, 0].set_yticklabels(days)
    ax[0, 0].set_title(f"A  matched neurons per pair of days (posterior >= {level_for_matrix})")
    # B: tracked neurons vs level
    cols = [c for c in summary.columns if c.startswith(">=")]
    w = 0.8 / len(cols)
    for k, c in enumerate(cols):
        ax[0, 1].bar(np.arange(len(summary)) + k * w, summary[c], width=w, label=c, color=plt.cm.viridis(k / max(len(cols) - 1, 1)))
        for x, v in zip(np.arange(len(summary)) + k * w, summary[c]):
            ax[0, 1].text(x, v, str(int(v)), ha="center", va="bottom", fontsize=7)
    ax[0, 1].set_xticks(np.arange(len(summary)) + 0.4 - w / 2); ax[0, 1].set_xticklabels([f"{l:.2f}" for l in summary.level])
    ax[0, 1].set_xlabel("confidence level (minimum posterior of every link)"); ax[0, 1].set_ylabel("neurons tracked")
    ax[0, 1].set_title("B  neurons tracked across >= k days"); ax[0, 1].legend(title="days")
    # C: calibration
    for (a, b), f in fdr_tables.items():
        ax[1, 0].plot(f.model_fdr, f.est_fdr, "o-", alpha=0.7, label=f"{a} vs {b}")
    lim = 0.3
    ax[1, 0].plot([0, lim], [0, lim], "k--", lw=1); ax[1, 0].set_xlim(0, lim); ax[1, 0].set_ylim(0, lim)
    ax[1, 0].set_xlabel("false-match rate claimed by the model (1 - mean posterior)")
    ax[1, 0].set_ylabel("false-match rate estimated with decoys")
    ax[1, 0].set_title("C  calibration check: on the diagonal = honest probabilities"); ax[1, 0].legend(fontsize=6, ncol=2)
    # D: drift
    D = np.full((len(days), len(days)), np.nan)
    for (a, b), r in results.items():
        D[days.index(a), days.index(b)] = r.drift_um; D[days.index(b), days.index(a)] = -r.drift_um
    ax[1, 1].imshow(D, cmap="coolwarm", vmin=-60, vmax=60)
    for i, j in itertools.product(range(len(days)), repeat=2):
        if i != j and D[i, j] == D[i, j]:
            ax[1, 1].text(j, i, f"{D[i, j]:+.0f}", ha="center", va="center", fontsize=10)
    ax[1, 1].set_xticks(range(len(days))); ax[1, 1].set_xticklabels(days); ax[1, 1].set_yticks(range(len(days))); ax[1, 1].set_yticklabels(days)
    ax[1, 1].set_title("D  estimated depth drift (um): column day relative to row day")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig
