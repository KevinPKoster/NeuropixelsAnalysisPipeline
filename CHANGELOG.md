# CHANGELOG - freely-moving pipeline (`fmpipe`)

## 2026-10-06 - v0.1: stages 0-2 (layout, sync/TTLs, frame alignment)

New module `fmpipe/` plus per-session notebooks in
`Anchored analysis/Freely moving analysis/Mouse 4103/`. The old notebooks
(`09152026.ipynb`, `09162026.ipynb`) are untouched.

### What changed relative to the old Cell A / Cell B
- **No blocks.** The camera pulse stream is split into trains; a train is a real
  recording only if its rate is the trigger rate (200.0 Hz paw cam, 120.19 Hz old
  second cam) **and** a counters CSV with a matching counter span exists.
  Free-run bursts (Live Preview / Test Snap: ~316 Hz) and single stray pulses are
  logged and ignored. The longest matched train is analysed (config `segment=`
  can pick another video). One `<view>_frame_times.npy` per camera, no `_block{N}`.
- **Hard errors instead of silent trimming**: NP<->NI sync pulses are paired by time
  (not by index), and the stage raises if <99% pair or the fit is off
  (|slope-1| > 1e-6, |offset| > 1 ms, residual std > 0.5 ms).
  The fit is verified, not applied: both streams share one OE clock.
- Counter span may not exceed the pulse count; trailing pulses are reported (warn > 50).
- Out-of-order NI timestamps (<=5 per train, each <1/4 frame interval) are repaired
  by sorting and logged; anything else raises. (Sept 14 video 2 has one: 0.69 ms.)
- No `save_incrementing`: re-runs overwrite, so no stale `TTL_Pin_*_1.npy`.
- NP continuous timestamps are memory-mapped (only first/last read) instead of
  loading ~1 GB.
- Everything is derived from `data_root + session + experiment`; the resolved config
  and git commit are saved in `metadata/session_config.json` and `fmpipe_version.json`.
- **Recorded probe layout** is read from Open Ephys `settings.xml` / `settings_N.xml`
  (bank/x/y per channel) and compared to Kilosort's `channel_positions.npy`.
- Old-style analysis folders are renamed `<name>_legacy`, never deleted.

### Validation
Sept 16 exp 2: frame times (780,495) and all 8 TTL pins identical to the old
notebook output (max |diff| = 0.0 s).
All of Sept 14-18 pass stages 0-2 (see below).

### Findings about the data (Sept 14-18)
- Layouts: Sept 14 and 15 = ch0-151 bank 1 / ch152-383 bank 0 ("9.14 onward" imro).
  Sept 16 exp 2, 17, 18 = ch0-153 bank 1 / ch154-383 bank 0 ("826 minus 1 row").
  Sept 16 exp 1 = all bank 0 (short wrong-map run, not analysed).
- Sept 16 exp 2 and Sept 17 were sorted with a Kilosort channel map that has channels
  154-155 at the wrong end of the probe (y 5360/5380 instead of 1540/1560). Only
  one good unit is affected (Sept 16 unit 373, best channel 154); depth is recomputed
  from the recorded layout. No re-sort needed.
- Free-run bursts come from the cameras' own ExposureActive line (Live Preview /
  Test Snap), not the Arduino.
- NI ch7 (pellet present) never fired Sept 14-18; ch5 = third camera (not connected).
- Sept 18: second camera ran at 200 Hz (new sensor) instead of 120 Hz.
- Sept 17 video 1 (48,141 frames) ended before the OE recording started: no pulses.

## 2026-10-06 - stage 3: units, depth, regions (`fmpipe/units.py`)
- **Kilosort/Phy copy selection.** Copies of `kilosort4` live on C:, K:, Z: with different
  curation (C:/Z: hold old Kilosort-default `mua/good` labels; K: holds the curated
  `good/noise`; Sept 14 exists only on C:). Labels and `spike_clusters.npy` must come
  from the SAME copy (Phy merges/splits rewrite spike_clusters), so the copy with the newest
  curation file is used and the choice is printed and saved to `metadata/units_source.json`.
- Labels come from the newer of `cluster_group.tsv` / `cluster_info.tsv`; disagreements
  are listed. Sept 16: the two files disagree on units 130 and 343 (good in group.tsv,
  noise in info.tsv). These are the two units the old notebook excluded by hand, so
  `exclude_units` is no longer needed.
- Best channel from the **unwhitened** template (matches Phy's `ch` for 100% of clusters);
  depth = recorded-layout y of that channel, DV = tip_dv_um (5429, chronic) - y.
  Sept 16 unit 373 (ch154) moves from a wrong DV 49 um (CFA) to DV 3889 um (VAL).
- Outputs: `units/units.csv`, `units/spike_times.npy` + `spike_clusters.npy` (good units),
  matched random noise-unit control (`units/noise_spike_*.npy`), `qc/probe_units.png`.
- Validated on Sept 16 vs the old notebook: spike trains identical, good-unit regions
  CFA 87->86 / VAL 37->38 (unit 373), WM 66.
- Good units: 09-14 112, 09-15 210, 09-16 190, 09-17 199, 09-18 206.
- `pip install tables` added to `spikeinterface_env` (needed to read DLC .h5).

## 2026-10-09 - stage 4: DLC -> canonical time (`fmpipe/dlc.py`)
- One `dlc_alignment/<view>_dlc.npz` per view (canonical_time + x/y/likelihood of every
  bodypart). No per-block files, no merge step.
- Checks: .h5 filename must be the video that stage 2 aligned (DDMMYYYY-N); DLC rows must
  equal saved frames, otherwise error.
- Sept 18: the acquisition stopped with "No space left on device" (camera PC E: full);
  the ledger counts 600,427 saved frames but only 600,350 reached disk (MP4 and DLC both
  600,350). The lost frames are the LAST 77, so the first 600,350 align one-to-one.
  Declared per session with `dlc_max_tail_missing={"paw_angled": 100}`; without the
  declaration the stage refuses.
- DLC models differ by day (09-14/17/18 shuffle3-130, 09-15 shuffle2-100,
  09-16 shuffle1-150): tracking is not strictly comparable across days.
- **Reach zone does not transfer between days.** The camera view/slit/pellet position
  differs between sessions (pellet median x/y: 09-14 539/348, 09-16 528/391, 09-17 425/365,
  09-18 419/335), and the Sept-16 absolute zone (y>380, x>400) is entered 58-81 times on
  Sept 16 but only 1-10 times on the other days. See
  `Mouse 4103/zone_check_sept14-18.png`. The zone must be calibrated per day.

## 2026-10-09 - stage 5: reach detection, stage 6: reach-locked PETH
- `fmpipe/reaches.py`: the Sept 16 absolute-zone detector, ported. **Regression: Sept 16 with
  zone (400, 380) gives 163 reaches / 66 bouts / 1 beam-tagged bout, onset times identical
  to the old notebook.**
- Reach zone is a per-session config value (`cfg.reach_zone`) because the camera/slit moves
  between days. Proposals: Sept 16 corner shifted by the difference in where the paw ends its
  reaches after pellet presentation (back-beam trials): 09-14 (411,321), 09-15 (311,328),
  09-16 (400,380), 09-17 (287,318), 09-18 (281,321). Not tuned on the behaviour they are
  used to analyse. Check with `plot_reach_zone` (frame + paw tracks + zone + trial-locking).
- Trial-locking enrichment (bout onsets in 12 s after back beam vs random windows) is the
  quick check of a zone: 09-14 40x, 09-15 9.6x, 09-16 12x, 09-17 3.5x, 09-18 39x.
  Sept 17 is low (animal reaches a lot between trials; weakest DLC model), but 43/75 of its
  bouts coincide with a pellet-beam break (beam was re-tuned by then).
- `fmpipe/peth.py`: shuffle-calibrated reach-locked PETH (200 shuffles, 25 ms bins, Z
  calibrated to <=10% shuffled FPR, >=80 ms in -0.5..2 s). Spikes are pre-binned at 5 ms
  (events quantised to 5 ms), ~4 s per session. **Regression Sept 16: Z=2.75, FPR 6.5% (old 6.6%),
  CFA 58 modulated, VAL 23 modulated (as old); denominators differ by one unit (373 moved CFA->VAL).**
- New depth-resolved figure (`plot_peth_depth`): each unit's z-scored PETH drawn at its true
  DV depth, probe map coloured by response, modulated fraction vs depth.
- Caveat inherited from the old method: the null is random times over the whole session, so
  units that are simply more active during reaching bouts than on average are flagged
  (many units are already elevated before t=0).
- `tools/sessions.py` + `tools/regenerate_notebooks.py`: session registry; the generator will
  not overwrite a notebook unless `--force` (which discards edits made in Jupyter).

## 2026-10-09 - dashboard, every-reach PETH, stacked region plot
- `session_status.py` (Documents): new columns Mouse, Counters, Aligned, Units, DLC->time,
  Reaches, Reach PETH, Report (link to per-day HTML in `Mouse 4103/reports/`); the hard-coded
  "R4078" header is gone; Kilosort/Phy copy and experiment come from the pipeline's own record;
  the Phy column also considers `cluster_info.tsv` when it is newer than `cluster_group.tsv`
  (Sept 16: units 130 and 343 are noise). Original saved as `session_status.py.bak_20261009`.
- `run_reach_peth(cfg, subset="all_reaches")`: every reach, not only the first of a bout;
  window -1.5..+1.5 s, response window -1..+1 s. Because reaches inside a bout are clustered,
  the null keeps each bout's internal reach timing and randomises only where bouts fall.
  Robustness vs bout-onset analysis (`compare_peth`): 91-96% of jointly modulated units keep
  their direction, peak-z correlation r = 0.91-0.96 on all five days.
- `plot_stacked_regions`: one heatmap, CFA on top and VAL below a divider; within each region
  excited (by latency), suppressed (by latency), then 10 representative non-modulated units.

## 2026-10-09 - cross-day unit matching (`fmpipe/matching/`)
Probabilistic matcher, intrinsic features only (nothing behavioural is imported by the package).
- `waveforms.py`: raw mean waveforms (CMR, 400 spikes per half) for every good unit from two time
  halves of the recording (~4 min per day, `units/waveforms.npz`).
- `features.py`: waveform footprint on PHYSICAL electrodes (row = electrode//2), so the 2-channel layout
  offset between days does not matter; centroid within +-3 rows of the peak (an earlier version used the
  10 strongest channels anywhere and was unstable by up to 170 um); ACG, ISI, rate.
- `model.py`: negative reference = different units of the same recording that sit close together;
  EM mixture per pair of days (naive-Bayes densities, one-to-one scaling); depth drift from the scan
  at which obvious matches agree in position; **decoy test** (other day displaced by 160-320 um) estimates the
  false-match rate at each level. `chains.py`: tracked neurons per confidence level; every pair of days in a chain
  must itself pass the level (chains are pruned to a mutually consistent set; at 0.8, 47 of 143 chains had to be pruned).
- Results (Sept 14-18, all 10 day pairs, matches at posterior >= 0.8): 14-15 13, 14-16 9, 14-17 8, 14-18 8,
  15-16 38, 15-17 54, 15-18 51, 16-17 61, 16-18 67, 17-18 78. Drift 1-18 um. Decoy-estimated false-match
  rate at >= 0.8: 2-8% per pair; at >= 0.9: 0-4%.
- Neurons tracked on >= k days with EVERY pair passing the level (0.6 / 0.8 / 0.95): >=2 days 178 / 143 / 102,
  >=3 days 89 / 59 / 31, >=4 days 36 / 14 / 3, all 5 days 3 / 2 / 1.
- Outputs in `D:\Neuropixels data-motor project\unit_matching\4103_sept14-18\`; notebook
  `Mouse 4103/cross_day_matching.ipynb` (+ HTML in `reports/`). `pip install opencv-python-headless` added to
  `spikeinterface_env` (reference video frames).

- Sequential tracking (`build_sequential_chains`, `sequential_summary`): link consecutive days (optionally one day
  gap), non-adjacent pairs reported but not required to pass. Neurons on all 5 days, strict -> sequential -> sequential
  with a 1-day gap: 0.6: 3 -> 13 -> 15; 0.8: 2 -> 3 -> 8; 0.95: 1 -> 2 -> 4. Some sequential 5-day neurons have a very low
  direct first-vs-last-day posterior (0.0-0.1): slow change or an accumulated wrong link.
- Sept 14 diagnostics: NOT noise-limited (same-unit split-half waveform cos 0.975 vs 0.979-0.982, ACG r 0.85, ~10k spikes per half) and
  yield is flat across spike-count terciles. A waveform+position-only model (no ACG/ISI/rate: state-independent, `STATE_FREE_COLS`)
  doubles 14-vs-15 matches at >=0.8 (13 -> 30) but not 14-vs-16/17/18. Dropping Sept 14 does not change the other pairs;
  4-day tracking on 15-18 (sequential, 0.8): 17 neurons.
- Stacked PETH plots for every day copied to `D:\4103_plots`.

## 2026-10-09 - Sept 15-18 is the main matching window; datasheets and probe figures
- Decision (Kevin): matching/analysis window = Sept 15-18 (training period); Sept 14 dropped from matching (short static-shaping
  session, 112 good units); full model kept. Results in `D:\Neuropixels data-motor project\unit_matching\4103_sept15-18\`.
- Matches at >= 0.8: 15-16 34, 15-17 50, 15-18 49, 16-17 61, 16-18 57, 17-18 74. Decoy false-match rate at >= 0.8: 3.8-7.1% per pair, at >= 0.9: 1.4-5.6%.
- Neurons on all 4 days at 0.6 / 0.7 / 0.8 / 0.9 / 0.95: strict 27 / 20 / 10 / 4 / 3; sequential 40 / 26 / 14 / 9 / 6;
  sequential with a one-day gap 55 / 44 / 31 / 20 / 11.
- `fmpipe/matching/datasheets.py`: Excel workbook (summary, matches per pair, decoy false-match rates, drift, one sheet per rule
  and level, and `_resp` sheets with each day's reach response; those columns were never used for matching).
- `plot_probe_modulation` (peth.py): reach modulation along the probe, dark and light themes, no marker outlines; faint
  bands mark CFA/WM/VAL; colour = signed peak z, size = |peak z|; blank where no unit; colour limits +-10 stated on the plot.
  `plot_tracked_probe` (crossday.py): tracked neurons on every day with lines joining the same neuron + peak z across days.
- Figures copied to `D:\4103_plots` (stacked PETHs, probe_modulation, cross_day_matching). `pip install openpyxl` added to `spikeinterface_env`.

- `plot_tracked_probe` population summary now split by region (CFA / WM / VAL; a tracked neuron's region = its mean drift-corrected
  depth) with mean +- SEM of signed peak z and of |peak z|. Only MEASURED values are used: days on which a neuron was not tracked
  are absent from that day's average, and a tracked but non-significant neuron contributes its measured peak z (earlier version
  substituted 0). `tracked_trajectories` writes the long table `tracked_trajectories_<rule>_p0.80.csv`.
- Sequential rule, level 0.8, >= 3 days (74 neurons: 38 CFA, 18 WM, 18 VAL): signed mean peak z is small and positive in CFA (0.9-2.5)
  and near zero in VAL (-0.1 to 1.5), clearly positive and stable in WM (~3.7-4.1); no opposite-sign population trend between CFA and VAL.
  |peak z|: CFA 4.2 / 6.0 / 4.9 / 3.4 and VAL 4.1 / 4.7 / 4.1 / 3.2 on 09-15..09-18. Magnitudes depend on the number of events per day
  (09-18 has only 23 bouts), so cross-day magnitude comparisons are confounded.

- **Region scheme changed (Kevin):** CFA 0-1300 um DV, WM 1300-3000, THAL 3000-3500, VAL >= 3500. "Thalamus" = THAL + VAL (everything
  >= 3000 um: firing pattern changes from white matter / ventricle to fibrous anterior thalamus to thalamus proper around 3000 um);
  VAL is the further subdivision. `DEFAULT_REGION_BOUNDS_DV_UM` in config.py; all days, PETHs, matching tables and figures were
  recomputed. Plots now show CFA / thalamus (>= 3000) / VAL (>= 3500) (stacked plot: CFA, THAL, VAL blocks).
- Reach-bout PETH modulated, thalamus (>= 3000 um): 09-15 55/87, 09-16 57/85, 09-17 47/72, 09-18 40/86 (VAL subset 26/40, 23/38, 22/32, 17/37).
- Tracked (sequential, 0.8, >= 3 days; 74 neurons: 38 CFA, 28 thalamus [18 VAL + 10 THAL], 8 WM): signed mean peak z, CFA 0.9-2.5 vs
  thalamus 0.8-2.4 vs VAL -0.1-1.5: same direction, no opposite-sign trend. |peak z| CFA 4.2/6.0/4.9/3.4, thalamus 4.4/4.9/4.1/3.6.

## 2026-10-09 - anatomy check of the region labels against the Allen atlas (`D:\4103_plots\anatomy\`)
Isolated venv in %TEMP% (brainglobe-atlasapi, Allen CCF 25 um); nothing installed into spikeinterface_env.
- Insertion (entry ML +1.3 / AP +0.6, 20 deg posterior pitch, 5 deg medial yaw; the sketch's "target AP -2.8" looks like a typo of -0.8:
  at 3.8 mm DV the track is at AP ~-0.8) walked through the atlas: MOp layers 1-6b (to ~1245 um vertical), cingulum/corpus callosum
  (~1245-1520), subependymal zone/lateral ventricle ("VL" in Allen = ventricle)/choroid plexus/fimbria (~1520-2770), AV (2770-3295),
  AMv (3295-3791), VM (3791-4315), ZI (4315+). Entry point lands in MOp (sanity check passed).
- **DV convention issue:** the units' "DV" (5429 - y) mixes vertical depth (tip) with along-shaft distance (y). True vertical depth is
  0.936 x along-shaft distance. One-parameter fit using two landmarks (cortex bottom, thalamus start): brain surface ~5380 um from the
  tip along the shaft (landmarks agree within 135 um). In the old coordinate the atlas boundaries are: cortex bottom ~1380,
  callosum 1380-1670, ventricle/fimbria 1670-3000 (13 units pooled), thalamus from ~3000 (AV to ~3570), then AMv ~3570-4100, VM from ~4100
  (deepest recording site 3889, so VM is not reached). Old labels in true vertical depth: 1300 -> 1170, 3000 -> 2762, 3500 -> 3230.
- Perturbed tracks (entry +-0.2 mm, angles +-2 deg, scale +-3%): the >= 3570 stretch is AM ~40%, VAL ~23-25%, RT ~9%, AV ~8%: the "VAL"
  label for >= 3500 is supported by roughly a quarter of plausible tracks; "thalamus >= 3000" is well supported.
- Firing rate by depth matches the atlas sequence (cortex ~2-8 Hz, dip at the L6b/callosum edge, ~1.5 Hz in callosum, silent ventricle,
  thalamus ~6-10 Hz). Files: probe_vs_atlas.png, structure_probability_by_depth.csv, atlas_segments_in_unit_coordinates.csv.

- Insertion planning (atlas sweep, yaw fixed at 0): pitch 22-25 deg is best (mean P(>= 500 um VAL in the recorded stretch) 0.27-0.28; 0.06 at 15 deg,
  0.11 at 30 deg); a 0.1 mm more posterior entry matters little; entry ML 1.2-1.4. The recorded window is the larger limit: VAL starts ~3480 um below the
  surface but the current layout stops at ~3860 um. Shifting the window 0.26-0.46 mm deeper (e.g. an Aug-25-style layout, or inserting ~0.45 mm deeper)
  raises P(>= 500 um VAL) to 0.55-0.62 for AP 0.5 / ML 1.3 / pitch 25, and 0.80-0.83 for AP 0.6 / ML 1.2 / pitch 25. Sweep tables: `insertion_sweep_*.csv`.
- `tools/draw_probe_in_brain.py`: Allen-atlas sagittal / dorsal / coronal drawing of the recommended insertion with CFA and VAL highlighted
  (now light theme only, simplified: no internal structure outlines, no previous-insertion line, fewer labels; output
  `D:\4103_plots\anatomy\probe_in_brain_light.png`). CFA is drawn as the MOp/MOs part within AP -0.25..+1.25, ML 0.8..2.0 mm (approximation).

## 2026-10-09 - controls before committing (no analyses were rerun; region scheme unchanged until histology)
- Backup of all plots, both matching output folders, the HTML reports and the zone-check image to `K:\4103_figures_backup_2026-10-09\` (verified file counts and sizes).
- `fmpipe/controls.py` `run_noise_control`: the matched random sample of noise-labelled units (>= 0.5 Hz) gets the same PETH / shuffle null / Z as the good units.
  Noise-labelled units are NOT a clean null (MUA, decent units labelled noise): reach-modulated 47% pooled (585 units; 35 / 72 / 34 / 58% on 09-15..18) vs shuffled
  chance 7-10%; 15% for units with > 10% refractory violations vs 46-56% for the better isolated; 40 noise-labelled units are clean-looking (ISI viol < 1%, contamination < 10%)
  AND modulated -> list for re-inspection in Phy (`D:\4103_plots\downstream\noise_labelled_units_to_recheck_in_phy.csv`). Caveat: movement artefacts coupled to reaches would also
  look modulated; check waveforms before relabelling.
- `fmpipe/peth.py` `run_matched_peth` / `load_matched_peth`: PETH at a common number of events (09-18 has only 23 bouts / 69 reaches) so z magnitudes are comparable
  across days: null built for that N, Z calibrated at that N, `replace=True` bootstrap draws (gives sampling uncertainty) or `replace=False` (distinct events). Bootstrapping
  with replacement inflates peak-based measures on the day whose pool equals N (09-18 CFA |peak z| 4.04 vs 3.47 without replacement), so |peak z| comparisons use
  `replace=False`; net z (mean z in the response window) is unaffected. At equal N=23 bouts (without replacement), mean |peak z| CFA 3.11 / 3.95 / 3.58 / 3.47 and
  thalamus 2.93 / 3.28 / 3.02 / 3.67 on 09-15..18: the apparent drop on 09-18 at full N (3.40 vs 4.3-5.8) disappears; no monotonic trend.
- `fmpipe/downstream.py` (reach data used only AFTER matching; never as a matching criterion):
  * validity test: 1596 matched pairs over 5 levels. Response (z-trace) correlation of matched pairs median 0.30-0.39 vs 0.005 for the clearly rejected (posterior < 0.05)
    candidates of the same unit (paired difference +0.28 to +0.37, 95% CI excludes 0 at every level, ~77-80% of pairs positive); sign of response (both modulated) agrees in 91-94% of matched pairs vs 53-55%
    for rejected candidates. Similarity falls slowly with days apart (median r 0.44 / 0.34 / 0.30 at 1 / 2 / 3 days, level 0.6; Spearman -0.14, p = 0.007; at 0.8 -0.11, p = 0.08).
  * per-neuron training trajectories (sequential rule, level 0.8, matched N=23 bouts, net z): CFA 31 modulated tracked cells: 10% increase / 61% stable / 29% decrease (bootstrap 95% CI), mean slope
    -0.13 z/day (permutation p = 0.12); thalamus 23 cells: 17% / 74% / 9%, mean slope +0.02 (p = 0.93). Spread of slopes across cells 0.54-0.59 z/day vs ~0.29 expected from sampling noise alone
    (individual cells change more than noise, in both directions); CFA mean decrease is not consistent across rules/levels (strict 0.8: -0.21, p = 0.03; sequential 0.95: -0.15, p = 0.15).
    Caveat: day-to-day change is confounded with behaviour/state/recording stability; four days only.
  * outputs in `D:\4103_plots\downstream\` (validity_of_matching.png, training_trajectories.png, noise_control.png, CSV tables, `noise_labelled_units_to_recheck_in_phy.csv`).
- Nothing was recomputed with a different region scheme; labels remain the target-area hypotheses until histology.

### Not done yet
Behaviour-state-matched null; confirm the track with histology / Pinpoint and then decide the final region scheme (and the DV geometry fix); recheck the flagged noise-labelled units in Phy.
