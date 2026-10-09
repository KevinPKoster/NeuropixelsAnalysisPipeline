"""
Cross-day matching, step 2: per-unit profiles and pairwise similarity features.

A *profile* describes one unit in one half of one recording: its mean waveform on the
physical electrodes around its peak, its position, and spike-train statistics (ACG, ISI,
rate). Pairs of profiles give the similarity features the matcher works with.

Everything is expressed on PHYSICAL electrode positions (row = electrode // 2 -> y = 20 um * row,
slot = electrode % 2 -> left/right site in the row), never on channel numbers, so recordings
with slightly different channel maps are comparable.

Nothing here knows about behaviour: only waveforms, positions and spike trains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import SessionConfig, read_json
from ..layout import load_recorded_layout
from ..units import load_units
from .waveforms import load_waveforms

ROW_UM = 20.0
HALF_ROWS = 8                      # stored window: peak row +- 8 rows
CMP_ROWS = 4                       # rows (each side) used when comparing two waveforms
LAGS = (-2, -1, 0, 1, 2)           # waveform time lags tried when comparing (samples)
N_ISI_BINS = 40


# --------------------------------------------------------------- spike trains
def acg(spikes, max_lag, bin_size):
    """Autocorrelogram counts for lags in (0, max_lag] (one-sided), normalised to Hz per spike."""
    n_bins = int(round(max_lag / bin_size))
    counts = np.zeros(n_bins)
    if len(spikes) < 2:
        return counts
    k = 1
    while k < len(spikes):
        d = spikes[k:] - spikes[:-k]
        d = d[d <= max_lag]
        if len(d) == 0:
            break
        counts += np.bincount(np.minimum((d / bin_size).astype(int), n_bins - 1), minlength=n_bins)
        k += 1
    return counts / (len(spikes) * bin_size)


def isi_hist(spikes):
    isi = np.diff(spikes)
    isi = isi[isi > 0]
    h, _ = np.histogram(np.log10(isi), bins=np.linspace(-3, 1, N_ISI_BINS + 1))
    return h / max(h.sum(), 1)


def _corr(a, b):
    """Pearson correlation; 0 (no evidence) when either trace is constant."""
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


# --------------------------------------------------------------------- profiles
@dataclass
class DayProfiles:
    name: str
    unit_ids: np.ndarray
    region: np.ndarray
    dv_um: np.ndarray
    n_spikes: np.ndarray                 # (U, 2) per half
    # waveform windows: (U, 2 halves, 2*HALF_ROWS+1 rows, 2 slots, T)
    win: np.ndarray
    peak_row: np.ndarray                 # (U,) electrode row of the peak channel
    peak_ch_trace: np.ndarray            # (U, 2, T) waveform on the peak channel
    amp: np.ndarray                      # (U, 2) peak-to-peak on the peak channel (uV)
    cent_y: np.ndarray                   # (U, 2) amplitude-weighted y per half (um, probe coordinates)
    cent_x: np.ndarray
    acg_mid: np.ndarray                  # (U, 2, 100)   1 ms bins to 100 ms
    acg_long: np.ndarray                 # (U, 2, 100)   10 ms bins to 1 s
    isi: np.ndarray                      # (U, 2, N_ISI_BINS)
    rate: np.ndarray                     # (U, 2) Hz
    duration: float

    @property
    def n(self):
        return len(self.unit_ids)


def build_profiles(cfg: SessionConfig, name: str | None = None) -> DayProfiles:
    w = load_waveforms(cfg)
    units, st, sc = load_units(cfg)
    ids = w["unit_ids"]
    u = units.set_index("id").loc[ids]
    lay = load_recorded_layout(cfg)
    electrode = lay["electrode"].astype(int)
    row, slot = electrode // 2, electrode % 2
    ch_y = lay["y"]; ch_x = lay["x"]
    duration = read_json(cfg.analysis_dir / "metadata" / "timebase.json")["recording_duration_s"]

    wf = w["wf"]                                          # (U, 2, T, 384)
    U, _, T, _ = wf.shape
    mean_wf = wf.mean(1)
    p2p = np.ptp(mean_wf, axis=1)                         # (U, 384)
    peak_ch = p2p.argmax(1)
    peak_row = row[peak_ch]

    win = np.zeros((U, 2, 2 * HALF_ROWS + 1, 2, T), np.float32)
    for k in range(U):
        rr = row - peak_row[k] + HALF_ROWS
        ok = (rr >= 0) & (rr < 2 * HALF_ROWS + 1)
        for h in (0, 1):
            win[k, h, rr[ok], slot[ok], :] = wf[k, h][:, ok].T

    # Amplitude-weighted position within +-3 rows (60 um) of the peak row, per half. Using only the
    # neighbourhood of the peak keeps far-away noisy channels from dragging the centroid.
    X_BY_PARITY_SLOT = np.array([[27.0, 59.0], [11.0, 43.0]])       # NP1.0: x of (row parity, slot)
    cent_y = np.zeros((U, 2)); cent_x = np.zeros((U, 2))
    rows_abs = peak_row[:, None] + np.arange(-HALF_ROWS, HALF_ROWS + 1)[None, :]           # (U, 17)
    near = np.abs(np.arange(-HALF_ROWS, HALF_ROWS + 1)) <= 3
    for h in (0, 1):
        p2p_w = np.ptp(win[:, h], axis=-1)                                # (U, 17, 2)
        wt = (p2p_w ** 2) * near[None, :, None]
        y_site = rows_abs[:, :, None] * ROW_UM + np.zeros((1, 1, 2))
        x_site = X_BY_PARITY_SLOT[rows_abs % 2, :]                          # (U, 17, 2)
        cent_y[:, h] = (wt * y_site).sum((1, 2)) / wt.sum((1, 2))
        cent_x[:, h] = (wt * x_site).sum((1, 2)) / wt.sum((1, 2))
    trace = np.stack([wf[k][:, :, peak_ch[k]] for k in range(U)])      # (U, 2, T)
    amp = np.ptp(trace, axis=2)

    acg_mid = np.zeros((U, 2, 100)); acg_long = np.zeros((U, 2, 100))
    isi = np.zeros((U, 2, N_ISI_BINS)); rate = np.zeros((U, 2)); nsp = np.zeros((U, 2), int)
    for k, uid in enumerate(ids):
        s = st[sc == uid]
        mid = len(s) // 2
        t_split = s[mid] if len(s) else duration / 2
        spans = (t_split, duration - t_split)
        for h, part in enumerate((s[:mid], s[mid:])):
            nsp[k, h] = len(part)
            rate[k, h] = len(part) / max(spans[h], 1e-9)
            acg_mid[k, h] = acg(part, 0.1, 0.001)
            acg_long[k, h] = acg(part, 1.0, 0.01)
            isi[k, h] = isi_hist(part)
    return DayProfiles(name or cfg.session[:10], ids, u.region.values.astype(str), u.dv_um.values, nsp, win,
                       peak_row, trace, amp, cent_y, cent_x, acg_mid, acg_long, isi, rate, duration)


# ------------------------------------------------------------------- pair features
FEATURES = ["wf_cos", "amp_abs_logratio", "dy_abs", "dx_abs", "acg_mid_r", "isi_r", "rate_abs_logratio"]
EXTRA = ["wf_peak_r", "acg_long_r"]        # computed for display / held-out checks, not used by the model


def _window_pair(A: DayProfiles, ia, ha, B: DayProfiles, jb, hb, drift_rows):
    """Cosine of the spatiotemporal windows (best time lag), aligned on PHYSICAL rows."""
    off = int(A.peak_row[ia] + drift_rows - B.peak_row[jb])          # row of A's peak, as seen by B, relative to B's peak
    if abs(off) > HALF_ROWS - CMP_ROWS:
        return np.nan, np.nan
    a = A.win[ia, ha, HALF_ROWS - CMP_ROWS: HALF_ROWS + CMP_ROWS + 1]                    # (9,2,T)
    lo = HALF_ROWS + off - CMP_ROWS
    b = B.win[jb, hb, lo: lo + 2 * CMP_ROWS + 1]
    best = -2.0
    for lag in LAGS:
        if lag >= 0:
            aa, bb = a[..., lag:], b[..., : b.shape[-1] - lag]
        else:
            aa, bb = a[..., : a.shape[-1] + lag], b[..., -lag:]
        na, nb = np.linalg.norm(aa), np.linalg.norm(bb)
        if na > 0 and nb > 0:
            best = max(best, float((aa * bb).sum() / (na * nb)))
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    ratio = np.log(max(nb, 1e-9) / max(na, 1e-9))
    return best, ratio


def pair_features(A: DayProfiles, ia, B: DayProfiles, jb, drift_um: float, halves=((0, 1), (1, 0))):
    """
    Similarity features of unit ia (day A) and jb (day B) given the day-B vs day-A probe drift
    (um, B relative to A). Averaged over the cross-half combinations in `halves`, so that
    every comparison (within or across days) has the same noise level.
    """
    d_rows = int(round(drift_um / ROW_UM))
    rows, dys, dxs = [], [], []
    for ha, hb in halves:
        cos, ratio = _window_pair(A, ia, ha, B, jb, hb, d_rows)
        if cos != cos:
            return None
        pk = _corr(A.peak_ch_trace[ia, ha], B.peak_ch_trace[jb, hb])
        dys.append((B.cent_y[jb, hb] - drift_um) - A.cent_y[ia, ha])
        dxs.append(B.cent_x[jb, hb] - A.cent_x[ia, ha])
        rows.append([cos, abs(ratio), pk,
                     _corr(A.acg_mid[ia, ha], B.acg_mid[jb, hb]),
                     _corr(A.acg_long[ia, ha], B.acg_long[jb, hb]),
                     _corr(A.isi[ia, ha], B.isi[jb, hb]),
                     abs(np.log((B.rate[jb, hb] + 1e-3) / (A.rate[ia, ha] + 1e-3)))])
    r = np.nanmean(np.array(rows), axis=0)
    # signed mean offset (for drift estimation) and mean ABSOLUTE offset per half-combination
    # (the feature: averaging signed offsets would cancel for the same unit)
    dy = float(np.mean(dys))
    dy_abs, dx_abs = float(np.mean(np.abs(dys))), float(np.mean(np.abs(dxs)))
    return dict(wf_cos=r[0], amp_abs_logratio=r[1], wf_peak_r=r[2], acg_mid_r=r[3], acg_long_r=r[4], isi_r=r[5],
                rate_abs_logratio=r[6], dy_abs=dy_abs, dx_abs=dx_abs, dy=dy)


def candidate_pairs(A: DayProfiles, B: DayProfiles, drift_um: float, same_day=False):
    """All (i, j) whose peak rows are within CMP_ROWS of each other after drift (i != j on the same day)."""
    d_rows = int(round(drift_um / ROW_UM))
    ii, jj = np.meshgrid(np.arange(A.n), np.arange(B.n), indexing="ij")
    near = np.abs(A.peak_row[ii] + d_rows - B.peak_row[jj]) <= HALF_ROWS - CMP_ROWS
    if same_day:
        near &= ii != jj
    return np.where(near)


def pair_table(A: DayProfiles, B: DayProfiles, drift_um: float, same_day=False, halves=((0, 1), (1, 0))) -> pd.DataFrame:
    ii, jj = candidate_pairs(A, B, drift_um, same_day)
    rec = []
    for i, j in zip(ii, jj):
        f = pair_features(A, i, B, j, drift_um, halves)
        if f is not None:
            f.update(i=int(i), j=int(j))
            rec.append(f)
    return pd.DataFrame(rec)


def same_unit_reference(A: DayProfiles) -> pd.DataFrame:
    """Same unit, first half vs second half of the same recording."""
    rec = []
    for i in range(A.n):
        f = pair_features(A, i, A, i, 0.0)
        if f is not None:
            f.update(i=i, j=i)
            rec.append(f)
    return pd.DataFrame(rec)
