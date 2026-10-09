"""
Session configuration for the freely-moving pipeline.

One SessionConfig describes one Open Ephys experiment/recording. Every path is
derived from `data_root` + `session` + `experiment`, so moving the data to a
different drive letter or machine only means changing `data_root`.

Everything the pipeline writes goes into `cfg.analysis_dir`; raw data is only
ever read.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# NI DAQ digital line -> meaning, for the pellet rig as used Sept 14-18 2026.
# (P0.5 / ch5 is the third camera when connected; ch7 is unused.)
DEFAULT_PIN_LABELS = {
    0: "paw_cam_trigger",
    1: "second_cam_trigger",
    2: "opto_led",
    3: "sync_1hz",
    4: "back_beam",
    5: "third_cam_trigger",
    6: "pellet_beam",
    7: "unused",
}

# Region boundaries in DV (um below the brain surface): DV = tip_dv_um - electrode_y.
#   CFA   motor cortex
#   WM    white matter / ventricle (low, irregular firing)
#   THAL  thalamus proper starts at ~3000 um (fibrous anterior thalamus first), not yet VA/VL
#   VAL   VA/VL motor thalamus
# "Thalamus" in summaries = THAL + VAL (everything >= 3000 um); VAL is the further subdivision >= 3500 um.
DEFAULT_REGION_BOUNDS_DV_UM = {
    "CFA": (0, 1300),
    "WM": (1300, 3000),
    "THAL": (3000, 3500),
    "VAL": (3500, 4500),
}

FMPIPE_VERSION = "0.1"


def _one(paths, what, where):
    paths = sorted(paths)
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {what} under\n  {where}\nfound {len(paths)}: "
            f"{[p.name for p in paths]}"
        )
    return paths[0]


@dataclass
class SessionConfig:
    # --- where the data is -------------------------------------------------
    data_root: str                  # e.g. "K:\\"  (folder holding the session folders)
    session: str                    # "2026-09-16_13-21-12"
    experiment: int = 1             # Open Ephys experiment number (1-based)
    recording: int = 1
    record_node: str | None = None  # autodetected ("Record Node 101") if None

    # --- cameras -----------------------------------------------------------
    video_dir: str = r"S:\MP4_BACKUP"   # counters CSVs / MP4s / DLC .h5 matching below
    # view name as it appears in the counters-CSV filename -> NI pin + trigger rate
    cameras: dict = field(default_factory=lambda: {
        "paw_angled": dict(pin=0, trigger_hz=200.0),
        "bottom":     dict(pin=1, trigger_hz=120.19),
    })
    primary_view: str = "paw_angled"    # must align, or the stage fails
    segment: str | int = "longest"      # "longest" or a video number (e.g. 3 = 16092026-3)

    # --- DLC ---------------------------------------------------------------
    dlc_h5: dict = field(default_factory=dict)   # view -> path of DLC .h5
    # view -> number of trailing ledger frames allowed to be missing from the video/DLC
    # (e.g. acquisition stopped by a full disk: counters logged frames that never reached disk)
    dlc_max_tail_missing: dict = field(default_factory=dict)

    # --- sync / timebase ---------------------------------------------------
    ni_sync_bit: int = 3
    np_sync_bit: int = 0
    np_sample_rate: float = 30000.0

    # --- reaches -----------------------------------------------------------
    # Reach zone in pixels of the primary camera view: a reach is paw_center/digit2 with
    # x > x_min AND y > y_min. The camera/slit moves between days, so calibrate PER DAY.
    reach_zone: dict | None = None            # {"x_min": ..., "y_min": ...}
    reach_params: dict = field(default_factory=dict)   # overrides of fmpipe.reaches.REACH_DEFAULTS
    trial_pin: int = 4                        # back beam: pellet arm comes up, trial starts
    beam_pin: int = 6                         # pellet beam (annotation only; unreliable in Sept 2026)

    # --- spike sorting output ----------------------------------------------
    ks_search_roots: tuple = ()     # extra roots to look for kilosort4 copies (missing ones are skipped)
    ks_dir_override: str | None = None
    min_noise_rate_hz: float = 0.5  # noise-unit control pool: only noise clusters at least this fast

    # --- probe / regions ---------------------------------------------------
    tip_dv_um: float = 5429.0      # chronic implant: tip depth below surface, fixed
    region_bounds_dv_um: dict = field(default_factory=lambda: dict(DEFAULT_REGION_BOUNDS_DV_UM))
    exclude_units: tuple = ()       # cluster ids to leave out of analyses

    pin_labels: dict = field(default_factory=lambda: dict(DEFAULT_PIN_LABELS))

    # ------------------------------------------------------------------ paths
    @property
    def session_dir(self) -> Path:
        return Path(self.data_root) / self.session

    @property
    def rn_dir(self) -> Path:
        if self.record_node:
            return self.session_dir / self.record_node
        nodes = [p for p in self.session_dir.iterdir()
                 if p.is_dir() and p.name.lower().startswith("record node")]
        return _one(nodes, "'Record Node' folder", self.session_dir)

    @property
    def rec_dir(self) -> Path:
        return self.rn_dir / f"experiment{self.experiment}" / f"recording{self.recording}"

    @property
    def ni_ttl_dir(self) -> Path:
        d = _one((self.rec_dir / "events").glob("NI-DAQmx-*"), "NI-DAQmx events folder",
                 self.rec_dir / "events")
        return d / "TTL"

    @property
    def np_cont_dir(self) -> Path:
        return _one((self.rec_dir / "continuous").glob("Neuropix-PXI-*AP"),
                    "Neuropixels AP stream", self.rec_dir / "continuous")

    @property
    def ni_cont_dir(self) -> Path:
        return _one((self.rec_dir / "continuous").glob("NI-DAQmx-*"),
                    "NI-DAQmx stream", self.rec_dir / "continuous")

    @property
    def np_ttl_dir(self) -> Path:
        return self.rec_dir / "events" / self.np_cont_dir.name / "TTL"

    # ---- Kilosort / Phy folder -------------------------------------------------
    # Several copies of a session's kilosort4 folder can exist (local OE disk, transfer SSD,
    # archive) with DIFFERENT curation. Labels, spike_clusters.npy etc. must all come from
    # ONE copy (Phy merges/splits rewrite spike_clusters), so the copy with the most recent
    # curation file wins unless `ks_dir_override` is set.
    def ks_candidates(self) -> list:
        rel = self.np_cont_dir.relative_to(self.session_dir) / "kilosort4"
        out = []
        for root in dict.fromkeys([self.data_root, *self.ks_search_roots]):
            d = Path(root) / self.session / rel
            if not (d / "spike_times.npy").exists():
                continue
            files = [d / f for f in ("cluster_group.tsv", "cluster_info.tsv") if (d / f).exists()]
            mtime = max((f.stat().st_mtime for f in files), default=0.0)
            out.append(dict(root=str(root), path=d, curated_mtime=mtime))
        return sorted(out, key=lambda c: -c["curated_mtime"])

    @property
    def ks_dir(self) -> Path:
        if self.ks_dir_override:
            return Path(self.ks_dir_override)
        c = self.ks_candidates()
        if not c:
            raise FileNotFoundError(
                f"No kilosort4 folder with spike_times.npy found for {self.session} experiment "
                f"{self.experiment} under {[self.data_root, *self.ks_search_roots]}")
        return c[0]["path"]

    @property
    def settings_xml(self) -> Path:
        # Open Ephys writes settings.xml for experiment 1, settings_N.xml for N >= 2
        name = "settings.xml" if self.experiment == 1 else f"settings_{self.experiment}.xml"
        return self.rn_dir / name

    @property
    def analysis_dir(self) -> Path:
        return self.rn_dir / f"{self.session}_experiment{self.experiment}_recording{self.recording}_analysis"

    @property
    def video_date_prefix(self) -> str:
        """Camera files are named DDMMYYYY-N_<view>_...; derived from the session date."""
        return datetime.strptime(self.session[:10], "%Y-%m-%d").strftime("%d%m%Y")

    # -------------------------------------------------------------- analysis
    def init_analysis(self) -> Path:
        """
        Create the analysis folder. A folder left by the old notebooks (no
        fmpipe marker) is renamed to '<name>_legacy', never deleted.
        """
        a = self.analysis_dir
        marker = a / "metadata" / "fmpipe_version.json"
        if a.exists() and not marker.exists():
            legacy = a.with_name(a.name + "_legacy")
            if legacy.exists():
                raise FileExistsError(f"Both {a.name} and {legacy.name} exist; resolve by hand.")
            a.rename(legacy)
            print(f"Old-style analysis folder renamed -> {legacy.name}")
        for sub in ("metadata", "ttls", "camera_alignment", "dlc_alignment", "qc", "units"):
            (a / sub).mkdir(parents=True, exist_ok=True)
        write_json(marker, {"fmpipe_version": FMPIPE_VERSION, "git_commit": _git_commit(),
                            "created": datetime.now().isoformat(timespec="seconds")})
        return a

    def save(self):
        """Write the resolved config next to the results so the folder is self-describing."""
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}
        write_json(self.analysis_dir / "metadata" / "session_config.json", d)


# ---------------------------------------------------------------------- utils
def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _git_commit():
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def update_status(cfg: SessionConfig, stage: str, ok: bool, **info):
    """Merge one stage's result into metadata/status.json (read by the dashboard)."""
    p = cfg.analysis_dir / "metadata" / "status.json"
    st = read_json(p) if p.exists() else {}
    st[stage] = {"ok": bool(ok), "time": datetime.now().isoformat(timespec="seconds"), **info}
    write_json(p, st)
