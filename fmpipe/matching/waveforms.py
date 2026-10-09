"""
Cross-day matching, step 1: raw mean waveforms of every good unit, from two time halves.

Every good unit's spikes are split into the first and the second half of the recording and
up to `n_per_half` spikes per half are read from the raw AP data (continuous.dat). The
two halves are what lets the matcher learn what 'the same neuron' looks like within a
recording (the positive reference) without any behavioural information.

Snippets: -40..+80 samples around the spike (1.33 ms before, 2.67 ms after), baseline
(first 20 samples) subtracted per channel, common-median reference across all 384 channels,
converted to microvolts. Snippets with |V| > 800 uV after referencing are dropped (artifacts).

This module (and the whole `matching` package) never imports anything from reaches/PETH:
behaviour must stay independent of how units are matched.
"""

from __future__ import annotations

import time

import numpy as np

from ..config import SessionConfig, read_json, update_status, write_json
from ..units import load_units

PRE, POST = 40, 81            # samples before / after (window length 121)
BASELINE = 20
ARTIFACT_UV = 800.0


def _open_dat(cfg: SessionConfig):
    dat = cfg.np_cont_dir / "continuous.dat"
    n = dat.stat().st_size // (384 * 2)
    return np.memmap(dat, dtype="<i2", mode="r", shape=(n, 384)), n


def _mean_waveform(mm, samples, bit_volts):
    """Mean referenced waveform (T, 384) in uV and the number of snippets used."""
    acc = np.zeros((PRE + POST, 384), np.float64)
    used = 0
    for lo in range(0, len(samples), 100):                       # chunks bound memory
        idx = samples[lo:lo + 100]
        snip = np.stack([mm[s - PRE:s + POST] for s in idx]).astype(np.float32) * bit_volts   # (n, T, 384)
        snip -= snip[:, :BASELINE].mean(axis=1, keepdims=True)
        snip -= np.median(snip, axis=2, keepdims=True)           # common median reference
        ok = np.abs(snip).max(axis=(1, 2)) < ARTIFACT_UV
        acc += snip[ok].sum(axis=0)
        used += int(ok.sum())
    return (acc / max(used, 1)).astype(np.float32), used


def extract_waveforms(cfg: SessionConfig, n_per_half: int = 400, force: bool = False) -> dict:
    out_path = cfg.analysis_dir / "units" / "waveforms.npz"
    if out_path.exists() and not force:
        print(f"waveforms exist: {out_path.name} (use force=True to recompute)")
        return dict(np.load(out_path))

    units, st, sc = load_units(cfg)
    ids = units[units.in_analysis].id.values
    mm, n_samp = _open_dat(cfg)
    oebin = read_json(cfg.rec_dir / "structure.oebin")
    stream = next(s for s in oebin["continuous"] if s["folder_name"].startswith(cfg.np_cont_dir.name))
    bit_volts = float(stream["channels"][0]["bit_volts"])

    wf = np.zeros((len(ids), 2, PRE + POST, 384), np.float32)
    n_used = np.zeros((len(ids), 2), int)
    n_spk = np.zeros((len(ids), 2), int)
    t0 = time.time()
    for k, uid in enumerate(ids):
        s = np.round(st[sc == uid] * cfg.np_sample_rate).astype(np.int64)
        s = s[(s > PRE) & (s < n_samp - POST)]
        halves = (s[: len(s) // 2], s[len(s) // 2:])
        for h in (0, 1):
            n_spk[k, h] = len(halves[h])
            if len(halves[h]) == 0:
                continue
            pick = halves[h][np.unique(np.linspace(0, len(halves[h]) - 1, n_per_half).round().astype(int))]
            wf[k, h], n_used[k, h] = _mean_waveform(mm, pick, bit_volts)
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(ids)} units  ({time.time() - t0:.0f} s)")

    np.savez(out_path, unit_ids=ids, wf=wf, n_used=n_used, n_spikes_half=n_spk,
             bit_volts=bit_volts, pre=PRE, post=POST)
    print(f"{cfg.session[:10]}: waveforms for {len(ids)} units in {time.time() - t0:.0f} s -> {out_path.name}")
    update_status(cfg, "waveforms", True, n_units=int(len(ids)), seconds=round(time.time() - t0))
    return dict(unit_ids=ids, wf=wf, n_used=n_used, n_spikes_half=n_spk)


def load_waveforms(cfg: SessionConfig) -> dict:
    d = np.load(cfg.analysis_dir / "units" / "waveforms.npz")
    return {k: d[k] for k in d.files}
