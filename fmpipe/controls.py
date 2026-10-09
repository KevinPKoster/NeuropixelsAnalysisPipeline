"""
Controls on the reach-PETH results.

Noise-unit control
------------------
Units labelled 'noise' in Phy are NOT a clean null: they contain genuine noise, but also multi-unit activity and decent units
that were put there because of an unclear ACG or waveform. So the question asked here is not "is the false-positive rate
right", but "how much of the reach modulation do the noise-labelled units carry, and is it concentrated in the ones that
look least like noise?" For the matched random sample of noise-labelled units (>= 0.5 Hz) it computes the same PETH, the same
shuffle null and the same Z threshold as for the good units, and relates modulation to simple quality measures:
  * isi_viol   fraction of inter-spike intervals < 1.5 ms (refractory-period violations; ~0 for a well isolated unit)
  * contam_pct Kilosort's contamination estimate
  * ks_label   Kilosort's own label ('good' / 'mua') before manual curation
and lists noise-labelled units that look clean AND are reach-modulated, as candidates to re-inspect in Phy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SessionConfig, read_json, update_status
from .peth import (PETH_DEFAULTS, SUBSET_PARAMS, _Psth, _events, _modulated, _null_stats, load_reach_peth)
from .units import load_noise_spikes, load_units

ISI_REFRACTORY_S = 0.0015


def _isi_violation(spikes):
    d = np.diff(spikes)
    return float((d < ISI_REFRACTORY_S).mean()) if len(d) else np.nan


def run_noise_control(cfg: SessionConfig, subset: str = "all", seed: int = 11, **overrides) -> pd.DataFrame:
    P = dict(PETH_DEFAULTS); P.update(SUBSET_PARAMS.get(subset, {})); P.update(overrides)
    a = cfg.analysis_dir
    duration = read_json(a / "metadata" / "timebase.json")["recording_duration_s"]
    events, _ = _events(cfg, subset)
    units, st_g, sc_g = load_units(cfg)
    good = pd.read_csv(a / "peth" / f"reach_peth_{subset}_units.csv")
    z_good = float(read_json(a / "peth" / f"reach_peth_{subset}.json")["z_threshold"])

    noise = units[units.noise_control].copy()
    st_n, sc_n = load_noise_spikes(cfg)
    trains = [st_n[sc_n == i] for i in noise.id.values]
    ps = _Psth(trains, duration, P)
    c = ps.centers
    resp = (c >= P["response_win"][0]) & (c < P["response_win"][1])
    min_bins = int(np.ceil(P["min_consecutive_ms"] / 1000.0 / P["bin_size"]))
    rng = np.random.default_rng(seed)
    null_mean, null_std, z_noise, fpr_noise = _null_stats(ps, len(events), duration, P, rng, resp, min_bins)
    z = (ps.mean_psth(events) - null_mean) / null_std
    zr = z[:, resp]; pk = np.argmax(np.abs(zr), axis=1)
    noise["peak_z"] = zr[np.arange(len(noise)), pk]
    noise["modulated"] = _modulated(z, resp, min_bins, z_good)              # same threshold as the good units
    noise["isi_viol"] = [_isi_violation(t) for t in trains]

    ci = pd.read_csv(cfg.ks_dir / "cluster_info.tsv", sep="\t").set_index("id")
    for col, new in (("ContamPct", "contam_pct"), ("KSLabel", "ks_label"), ("Amplitude", "amplitude")):
        if col in ci.columns:
            noise[new] = ci.loc[noise.id.values, col].values
            good[new] = ci.reindex(good.id.values)[col].values
    good["isi_viol"] = [_isi_violation(st_g[sc_g == i]) for i in good.id.values]
    noise["direction"] = np.where(~noise.modulated, "none", np.where(noise.peak_z > 0, "excited", "suppressed"))
    noise["day"] = cfg.session[5:10]
    noise["clean_looking"] = (noise.isi_viol < 0.01) & (noise.contam_pct < 10 if "contam_pct" in noise else True)
    (a / "peth").mkdir(exist_ok=True)
    noise.to_csv(a / "peth" / f"noise_control_{subset}.csv", index=False)

    frac = lambda m: (float(m.mean()), int(m.sum()), int(len(m)))
    summary = dict(
        z_good=z_good, z_noise_calibrated=z_noise, shuffled_fpr_noise=fpr_noise,
        good_modulated=frac(good.modulated), noise_modulated=frac(noise.modulated),
        noise_clean_looking_modulated=frac(noise[noise.clean_looking].modulated) if noise.clean_looking.any() else None,
        noise_ks_good_modulated=frac(noise[noise.ks_label == "good"].modulated) if "ks_label" in noise and (noise.ks_label == "good").any() else None,
        noise_ks_mua_modulated=frac(noise[noise.ks_label == "mua"].modulated) if "ks_label" in noise and (noise.ks_label == "mua").any() else None)
    update_status(cfg, f"noise_control_{subset}", True, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in summary.items()})
    return noise, good, summary
