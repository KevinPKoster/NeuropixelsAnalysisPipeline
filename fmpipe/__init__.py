"""fmpipe - freely-moving Neuropixels + DeepLabCut analysis pipeline (Sherman lab)."""

from .config import SessionConfig, read_json, write_json, update_status  # noqa: F401
from .layout import run_layout, load_recorded_layout                       # noqa: F401
from .sync import run_sync, load_ttl                                       # noqa: F401
from .frames import run_frames, load_frame_times                           # noqa: F401
from .units import run_units, load_units, load_noise_spikes, plot_probe_units  # noqa: F401
from .dlc import run_dlc, load_dlc                                         # noqa: F401
from .reaches import detect_reaches, plot_reach_zone, propose_zones, trial_endpoints  # noqa: F401
from .peth import (run_reach_peth, load_reach_peth, plot_reach_peth, plot_peth_depth,  # noqa: F401
                   plot_stacked_regions, compare_peth, plot_probe_modulation)
from .crossday import plot_tracked_probe                                       # noqa: F401
