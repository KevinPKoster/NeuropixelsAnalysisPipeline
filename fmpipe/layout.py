"""
Channel layout: what was actually recorded, read from Open Ephys settings.xml.

settings.xml (experiment 1) / settings_N.xml (experiment N) lists, for every one
of the 384 recorded channels, its bank and its electrode x/y position. That is
the ground truth for the recording. Kilosort's channel_positions.npy only
reflects whatever channel map the sorter was handed, so the two are compared
and any disagreement is reported.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from .config import SessionConfig, update_status, write_json

N_CH = 384


def _attr_array(elem, n=N_CH):
    return np.array([float(elem.attrib[f"CH{i}"]) for i in range(n)])


def read_recorded_layout(cfg: SessionConfig) -> dict:
    """Per-channel bank, x, y (um from probe tip) and NP1.0 electrode id, as recorded."""
    xml = cfg.settings_xml
    if not xml.exists():
        raise FileNotFoundError(f"Open Ephys settings not found: {xml}")
    root = ET.parse(xml).getroot()
    probe = root.find(".//NP_PROBE")
    if probe is None:
        raise RuntimeError(f"No NP_PROBE section in {xml}")
    bank = _attr_array(probe.find("CHANNELS")).astype(int)
    x = _attr_array(probe.find("ELECTRODE_XPOS"))
    y = _attr_array(probe.find("ELECTRODE_YPOS"))
    imro = [f.attrib.get("PATH") for f in probe.findall("./IMRO_FILES/FILE")]
    return dict(
        bank=bank, x=x, y=y,
        electrode=np.arange(N_CH) + N_CH * bank,   # NP1.0: electrode = channel + 384 * bank
        imro=imro, settings_xml=str(xml),
        serial=probe.attrib.get("probe_serial_number"),
    )


def bank_runs(bank):
    """[(first_ch, last_ch, bank), ...] -- compact description of the layout."""
    runs, s = [], 0
    for i in range(1, len(bank) + 1):
        if i == len(bank) or bank[i] != bank[s]:
            runs.append((s, i - 1, int(bank[s])))
            s = i
    return runs


def run_layout(cfg: SessionConfig) -> dict:
    """Stage 0: save the recorded layout and compare with Kilosort's channel positions."""
    lay = read_recorded_layout(cfg)
    a = cfg.analysis_dir
    np.savez(a / "metadata" / "channel_layout.npz", **{k: v for k, v in lay.items()
                                                       if isinstance(v, np.ndarray)})

    ks_pos_path = cfg.ks_dir / "channel_positions.npy"
    ks_chan_map = cfg.ks_dir / "channel_map.npy"
    mismatch = []
    if ks_pos_path.exists():
        ks = np.load(ks_pos_path)
        cmap = np.load(ks_chan_map).astype(int) if ks_chan_map.exists() else np.arange(N_CH)
        # Kilosort row k is the channel cmap[k]
        y_ks = np.empty(N_CH); y_ks[cmap] = ks[:, 1]
        # Only y (depth) is compared: Kilosort channel maps use their own x origin
        # (e.g. 0/16/32/48 vs Open Ephys 11/27/43/59), which does not affect depth.
        bad = np.where(np.abs(y_ks - lay["y"]) > 1e-6)[0]
        mismatch = [int(c) for c in bad]

    info = dict(
        bank_runs=bank_runs(lay["bank"]),
        electrode_range=[int(lay["electrode"].min()), int(lay["electrode"].max())],
        y_range_um=[float(lay["y"].min()), float(lay["y"].max())],
        imro_files=lay["imro"], probe_serial=lay["serial"],
        kilosort_position_mismatch_channels=mismatch,
    )
    write_json(a / "metadata" / "channel_layout.json", info)

    print(f"Layout ({cfg.settings_xml.name}): bank runs {info['bank_runs']}, "
          f"y {info['y_range_um'][0]:.0f}-{info['y_range_um'][1]:.0f} um")
    if mismatch:
        print(f"WARNING: Kilosort channel_positions disagree with the recording on "
              f"{len(mismatch)} channel(s): {mismatch}\n"
              f"  Unit depths will be recomputed from the recorded layout.")
    update_status(cfg, "layout", True, bank_runs=info["bank_runs"],
                  n_ks_mismatch=len(mismatch))
    return info


def load_recorded_layout(cfg: SessionConfig) -> dict:
    d = np.load(cfg.analysis_dir / "metadata" / "channel_layout.npz")
    return {k: d[k] for k in d.files}
