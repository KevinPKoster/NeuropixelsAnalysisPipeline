"""
Stage 3: units, depth and regions.

Reads the curated Kilosort/Phy folder (the copy with the most recent curation, see
SessionConfig.ks_dir) FRESH every run and never writes to it.

For every cluster:
  * label from Phy ('good' / 'noise' / 'mua' / 'unsorted'),
  * best channel and position from the UNWHITENED template (reproduces Phy's own best
    channel for 100% of clusters) combined with the RECORDED probe layout, so depth is
    right even if Kilosort was given a wrong channel map,
  * DV (um below the brain surface) = tip_dv_um - electrode y, and a region label.

Writes (all canonical seconds = since NP stream start):
  units/units.csv                  one row per cluster
  units/spike_times.npy, spike_clusters.npy              good units
  units/noise_spike_times.npy, noise_spike_clusters.npy  matched random noise-unit control
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from .config import SessionConfig, read_json, update_status, write_json
from .layout import load_recorded_layout


# ----------------------------------------------------------------- Phy labels
def read_phy_labels(ks_dir):
    """
    Labels from the more recently saved of cluster_group.tsv / cluster_info.tsv.
    Returns (Series id -> label, info dict with both files' counts and disagreements).
    """
    g_path, i_path = ks_dir / "cluster_group.tsv", ks_dir / "cluster_info.tsv"
    g = i = None
    if g_path.exists():
        df = pd.read_csv(g_path, sep="\t")
        g = df.set_index("cluster_id")["group"]
    if i_path.exists():
        df = pd.read_csv(i_path, sep="\t")
        i = df.set_index("id")["group"]
    if g is None and i is None:
        raise FileNotFoundError(f"No cluster_group.tsv or cluster_info.tsv in {ks_dir}")

    info = {}
    for name, s in (("cluster_group.tsv", g), ("cluster_info.tsv", i)):
        info[name] = None if s is None else s.fillna("unsorted").value_counts().to_dict()
    if g is not None and i is not None:
        both = pd.concat([g.rename("g"), i.rename("i")], axis=1).fillna("unsorted")
        dis = both[both.g != both.i]
        info["disagree_ids"] = {int(k): f"{r.g} (group.tsv) vs {r.i} (info.tsv)" for k, r in dis.iterrows()}
    newer_is_info = i is not None and (g is None or i_path.stat().st_mtime >= g_path.stat().st_mtime)
    info["used"] = "cluster_info.tsv" if newer_is_info else "cluster_group.tsv"
    return (i if newer_is_info else g).fillna("unsorted"), info


# ------------------------------------------------------------------ templates
def template_positions(ks_dir, spike_clusters, spike_templates, layout):
    """Per-cluster best channel, x, y (best channel) and amplitude-weighted y (local COM)."""
    T = np.load(ks_dir / "templates.npy").astype(np.float64)
    wi = np.load(ks_dir / "whitening_mat_inv.npy").astype(np.float64)
    T = T @ wi                                             # unwhiten
    cmap_path = ks_dir / "channel_map.npy"
    cmap = np.load(cmap_path).astype(int).ravel() if cmap_path.exists() else np.arange(T.shape[2])
    nt = T.shape[0]

    clusters = np.unique(spike_clusters)
    ci = np.searchsorted(clusters, spike_clusters).astype(np.int64)
    W = np.bincount(ci * nt + spike_templates.astype(np.int64),
                    minlength=len(clusters) * nt).reshape(len(clusters), nt).astype(np.float64)
    W /= W.sum(1, keepdims=True)
    wf = (W @ T.reshape(nt, -1)).reshape(len(clusters), T.shape[1], T.shape[2])
    p2p_ks = np.ptp(wf, axis=1)                            # (clusters, kilosort channel rows)
    p2p = np.zeros_like(p2p_ks)
    p2p[:, cmap] = p2p_ks                                  # -> recorded channel index
    best = p2p.argmax(1)

    y, x = layout["y"], layout["x"]
    y_com = np.empty(len(clusters))
    for k in range(len(clusters)):
        near = np.abs(y - y[best[k]]) <= 60.0
        w = p2p[k] ** 2 * near
        y_com[k] = (w * y).sum() / w.sum()
    return clusters, best, x[best], y[best], y_com


# ---------------------------------------------------------------------- stage
def assign_region(dv, bounds):
    for name, (lo, hi) in bounds.items():
        if lo <= dv < hi:
            return name
    return "other"


def run_units(cfg: SessionConfig) -> pd.DataFrame:
    a = cfg.analysis_dir
    layout = load_recorded_layout(cfg)
    tb = read_json(a / "metadata" / "timebase.json")
    duration = tb["recording_duration_s"]

    # ---- which Kilosort/Phy copy ----------------------------------------
    cands = cfg.ks_candidates()
    ks = cfg.ks_dir
    print("Kilosort/Phy copies (newest curation first):")
    for c in cands:
        mark = "  <-- USED" if c["path"] == ks else ""
        print(f"   {c['root']:<40s} curated {datetime.datetime.fromtimestamp(c['curated_mtime']):%Y-%m-%d %H:%M}{mark}")

    st = np.load(ks / "spike_times.npy").ravel()
    sc = np.load(ks / "spike_clusters.npy").ravel()
    stm = np.load(ks / "spike_templates.npy").ravel()
    t_can = st / cfg.np_sample_rate                         # canonical seconds (NP start = 0)
    if t_can.min() < -0.5 or t_can.max() > duration + 1.0:
        raise RuntimeError(
            f"Spike times span {t_can.min():.2f}..{t_can.max():.2f} s but the NP recording is "
            f"0..{duration:.2f} s: spike_times do not belong to this recording.")

    labels, label_info = read_phy_labels(ks)
    clusters, best, x_um, y_um, y_com = template_positions(ks, sc, stm, layout)
    n_spk = np.bincount(np.searchsorted(clusters, sc), minlength=len(clusters))

    df = pd.DataFrame(dict(id=clusters, label=labels.reindex(clusters).fillna("unsorted").values,
                           n_spikes=n_spk, rate_hz=n_spk / duration, best_ch=best,
                           x_um=x_um, depth_um=y_um, depth_com_um=y_com))
    df["dv_um"] = cfg.tip_dv_um - df.depth_um
    df["region"] = [assign_region(v, cfg.region_bounds_dv_um) for v in df.dv_um]

    # units whose best channel Kilosort had at the wrong depth (wrong channel map)
    lay_info = read_json(a / "metadata" / "channel_layout.json")
    bad_ch = set(lay_info["kilosort_position_mismatch_channels"])
    df["ks_position_artifact"] = df.best_ch.isin(bad_ch)
    df["excluded"] = df.id.isin(cfg.exclude_units)
    df["in_analysis"] = (df.label == "good") & ~df.excluded

    # cross-check best channel against Phy's own, when cluster_info.tsv is available
    ci_path = ks / "cluster_info.tsv"
    if ci_path.exists():
        ci = pd.read_csv(ci_path, sep="\t").set_index("id")
        m = ci["ch"].reindex(df.id).values
        agree = float(np.mean(m == df.best_ch.values))
        print(f"Best channel agrees with Phy cluster_info 'ch' for {agree:.1%} of clusters")
        if agree < 0.97:
            print("  WARNING: low agreement; check that templates/whitening belong to this sort.")

    # ---- noise-unit control: matched random sample (same N as good units) -----
    rng = np.random.default_rng(0)
    pool = df[(df.label == "noise") & (df.rate_hz >= cfg.min_noise_rate_hz)].id.values
    n_good = int(df.in_analysis.sum())
    pick = rng.choice(pool, size=min(n_good, len(pool)), replace=False) if len(pool) else np.array([], int)
    df["noise_control"] = df.id.isin(pick)
    if len(pick) < n_good:
        print(f"  NOTE only {len(pick)} eligible noise units for a control of {n_good} good units")

    # ---- spike trains --------------------------------------------------------
    good_ids = df[df.label == "good"].id.values
    for name, ids in (("", good_ids), ("noise_", pick)):
        m = np.isin(sc, ids)
        order = np.argsort(t_can[m], kind="stable")
        np.save(a / "units" / f"{name}spike_times.npy", t_can[m][order])
        np.save(a / "units" / f"{name}spike_clusters.npy", sc[m][order].astype(np.int32))
    df.to_csv(a / "units" / "units.csv", index=False)

    src = dict(ks_dir=str(ks), candidates=[dict(root=c["root"], path=str(c["path"]),
                                                  curated=datetime.datetime.fromtimestamp(c["curated_mtime"]).isoformat(timespec="minutes"))
                                           for c in cands],
               label_files=label_info, n_good=n_good, n_noise_control=int(len(pick)))
    write_json(a / "metadata" / "units_source.json", src)

    # ---- report --------------------------------------------------------------
    print(f"Labels used from {label_info['used']}: "
          + ", ".join(f"{k}={v}" for k, v in df.label.value_counts().items()))
    if label_info.get("disagree_ids"):
        d = label_info["disagree_ids"]
        print(f"  NOTE cluster_group.tsv and cluster_info.tsv disagree on {len(d)} cluster(s): "
              + "; ".join(f"{k}: {v}" for k, v in list(d.items())[:6]))
    g = df[df.in_analysis]
    print(f"{len(g)} good units in analysis ({int(df.excluded.sum())} excluded by config)")
    print(g.groupby("region").size().rename("n").to_frame().T.to_string())
    art = df[df.ks_position_artifact & (df.label == "good")]
    if len(art):
        print("  Units whose Kilosort depth was wrong (recomputed): "
              + ", ".join(f"{r.id} (ch{r.best_ch}, DV {r.dv_um:.0f} um, {r.region})" for r in art.itertuples()))
    update_status(cfg, "units", True, n_good=int(len(g)), regions=g.region.value_counts().to_dict(),
                  ks_dir=str(ks), ks_position_artifact_units=[int(i) for i in art.id])
    return df


def load_units(cfg: SessionConfig):
    """-> (units table, good spike_times, good spike_clusters), canonical seconds."""
    u = cfg.analysis_dir / "units"
    return (pd.read_csv(u / "units.csv"), np.load(u / "spike_times.npy"), np.load(u / "spike_clusters.npy"))


def load_noise_spikes(cfg: SessionConfig):
    u = cfg.analysis_dir / "units"
    return np.load(u / "noise_spike_times.npy"), np.load(u / "noise_spike_clusters.npy")


def plot_probe_units(cfg: SessionConfig, units: pd.DataFrame | None = None, save=True):
    """Units along the probe at their DV depth, coloured by region, sized by firing rate."""
    import matplotlib.pyplot as plt
    u = units if units is not None else pd.read_csv(cfg.analysis_dir / "units" / "units.csv")
    g = u[u.in_analysis]
    colors = dict(CFA="tab:blue", WM="0.55", THAL="tab:cyan", VAL="tab:green", other="tab:red")
    fig, ax = plt.subplots(1, 2, figsize=(8, 8), gridspec_kw=dict(width_ratios=[2, 1]), sharey=True)
    for name, (lo, hi) in cfg.region_bounds_dv_um.items():
        for a_ in ax:
            a_.axhspan(lo, hi, color=colors.get(name, "0.8"), alpha=0.10)
        ax[0].text(0.02, (lo + hi) / 2, name, transform=ax[0].get_yaxis_transform(),
                   fontsize=9, va="center", ha="left")
    for reg, d in g.groupby("region"):
        ax[0].scatter(d.x_um + np.random.default_rng(0).uniform(-6, 6, len(d)), d.dv_um,
                      s=8 + 6 * np.sqrt(d.rate_hz), color=colors.get(reg, "k"), alpha=0.8, label=f"{reg} ({len(d)})")
    art = g[g.ks_position_artifact]
    if len(art):
        ax[0].scatter(art.x_um, art.dv_um, s=120, facecolors="none", edgecolors="k", label="Kilosort depth was wrong")
    lay = load_recorded_layout(cfg)
    ax[0].scatter(lay["x"], cfg.tip_dv_um - lay["y"], s=1, color="k", alpha=0.15)   # recorded sites
    ax[0].set_ylim(cfg.tip_dv_um - lay["y"].min() + 100, -100)
    ax[0].set_xlabel("x on probe (um)"); ax[0].set_ylabel("DV below surface (um)")
    ax[0].legend(fontsize=8, loc="center right"); ax[0].set_title(f"{cfg.session[:10]}: good units on the probe")
    ax[1].hist(g.dv_um, bins=np.arange(0, cfg.tip_dv_um, 100), orientation="horizontal", color="k")
    ax[1].set_xlabel("units / 100 um")
    plt.tight_layout()
    if save:
        fig.savefig(cfg.analysis_dir / "qc" / "probe_units.png", dpi=130)
    return fig
