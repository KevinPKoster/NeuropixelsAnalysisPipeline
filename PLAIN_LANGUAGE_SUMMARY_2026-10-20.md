# What was done on 19-20 Oct 2026, in plain language

Written for Kevin. Mouse 4103, freely moving reach-to-grasp, sessions Sept 14-18 2026.
The technical version with all the numbers is in `CHANGELOG.md`; this file explains the same
work without jargon, and lists every quantity that goes into the new figures.

Nothing in stages 0-6 was re-run and no region labels were changed. Everything new reads the
results that were already saved.

---

## 1. Setting up the office PC

- The external drive ("TransferSSD") shows up here as `D:` but is the same disk that is `K:`
  on the lab PC. It can be renamed to `K:` permanently with one admin command (in section 7).
- The conda environment `spikeinterface_env` was already here. Three packages were missing and
  were installed: **tables** (needed to read DLC `.h5` files), **openpyxl** (needed for the
  unit-matching spreadsheet) and **seaborn**.
- The repo is at `C:\Users\ShermanLab\NeuropixelsAnalysisPipeline\NeuropixelsAnalysisPipeline`
  and was already up to date.
- All five sessions have complete analysis folders on the drive, so the pipeline runs here
  without needing the video files or the DLC files.

---

## 2. Three things were added

### a) A kinematics stage (`fmpipe/kinematics.py`)

Asks: **how does the firing relate to how the paw actually moved?** Stage 6 only asks whether
a unit changes its firing around a reach. This stage adds the paw's path and speed, and checks
whether firing follows the movement.

Two problems in the tracking had to be handled first, because both changed the numbers:

1. **DLC sometimes jumps.** The tracker occasionally puts the paw marker confidently on the
   wrong thing, so the paw appears to move up to 151 pixels in 5 milliseconds (about 21,000
   pixels/second, which is impossible). About 0.3% of frames do this. They are now thrown out.
   Before the fix, the average reach speed peaked at ~1200 px/s; after, ~500 px/s.
2. **Some units barely fire.** One unit looked like the strongest result in the whole session,
   but the number came from 57 spikes spread over all 66 reaches. Units with fewer than
   2 spikes per reach are now left out of the correlation test only (they stay in everything else).

### b) Grant figures (`fmpipe/grantfig.py`)

Drawing only - no new analysis happens in this file. Three figures, described in section 3.

### c) An alignment audit (no code changed)

The freely-moving responses looked much blurrier than the old head-fixed ones, so the timing
was re-derived from the raw Open Ephys files and compared to what the pipeline had saved.
**The timing is correct** - see section 5.

---

## 3. The figures, and how each one was made

All in `D:\4103_plots\jason_grant_figures\` (PNG at 300 dpi, plus PDF), except the kinematics
figure which is saved next to each session's data in `.../qc/kinematics_all.png`.

| Figure | What it shows, in one sentence | How it was made |
|---|---|---|
| `stacked_peth_by_day` | All five days side by side, motor cortex above thalamus, so you can see the reach response on each day. | Takes the already-computed PETHs, cuts them to -1 to +1 second, groups units into excited / suppressed / not modulated, and sorts each group by when its response peaks. |
| `exemplar_rasters_<date>` | The clearest individual units: every spike on every reach, with the average firing rate underneath. | Picks the 10 strongest excited and 8 strongest suppressed units, alternating cortex and thalamus so neither dominates, and draws each unit's own baseline rate as a dashed line. |
| `change_from_baseline` | How much each unit's firing changed, in Hz and as a percentage, on each day. | For each unit, finds the moment of biggest change inside the response window and reports how far it is from that unit's own baseline. |
| `kinematics_all` (per session) | Where the paw went on each reach, how fast, and whether firing follows it. | Paw position and speed from DLC, plus two tests of whether firing tracks paw speed. |

---

## 4. Every variable used in the figures, explained simply

### Already produced by stage 6 (not new, but used throughout)

| Name | Plain meaning |
|---|---|
| `psth` | The unit's average firing rate (in Hz) at each moment around a reach. |
| `null_mean` | **The baseline.** The same average, but for randomly chosen moments instead of reaches - i.e. what this unit normally does. |
| `null_std` | How much that random version bounces around. Used as the yardstick for "is this change unusual?" |
| `z` | `(psth - null_mean) / null_std`. How many yardsticks away from normal the firing is. Positive = firing up, negative = firing down. |
| `modulated` | True if `z` stayed beyond 2.75 for at least 80 milliseconds in a row. 2.75 was not chosen by hand: it is the value at which the same test applied to shuffled data wrongly flags only 10% of units. |
| `peak_z` | The biggest `z` (positive or negative) in the response window. |
| `peak_latency` | *When* that biggest change happened, relative to reach onset. |
| `region` | Which part of the brain the unit sits in, from its depth. **These are target guesses, not histology.** |

### New: change from baseline (`fmpipe/peth.py`, `peak_change`)

| Name | Plain meaning |
|---|---|
| `delta_hz` | How many Hz the firing changed from that unit's own baseline, at the moment of biggest change. Positive = sped up. |
| `baseline_hz` | What the unit was doing normally at that same moment, in Hz. |
| `pct_change` | The same change as a percentage of baseline. A unit going 5 Hz -> 10 Hz is +100%. |
| `latency_s` | When the biggest change happened. |

### New: paw kinematics (`fmpipe/kinematics.py`)

| Name | Plain meaning |
|---|---|
| `x`, `y` | Where the paw is, in camera pixels. Blank whenever DLC is not confident. |
| `speed` | How fast the paw is moving, in pixels per second. Worked out only within stretches where tracking is continuous - never across a gap. |
| `seg_id` | Which unbroken stretch of tracking a frame belongs to. Used so plots don't draw a straight line across a gap the paw never travelled. |
| `frac_tracked` | What fraction of all video frames have a confidently tracked paw. On these sessions: 27-48%. The paw is simply out of view most of the time. |
| `n_jump_frames` | How many frames were thrown out as impossible jumps (see 2a). |
| `coverage` | For each moment around a reach, the fraction of reaches that actually have a tracked paw then. **Important:** this runs from about 10% to 70%, so the average paw position long before a reach is built from only a handful of reaches. Shaded grey in the figure. |
| `peak_r`, `best_lag_s`, `coupling_p` | Whole-session test: does firing rise and fall with paw speed, and at what delay? **This test is confounded** - the paw is only tracked while reaching, so a unit that simply fires more during reaches will pass it. Do not quote it on its own. |
| `tw_peak_r`, `tw_best_lag_s`, `tw_p` | The honest version. The average reach is subtracted from both the speed and the firing, leaving only how each individual reach differed from the average. Then it asks whether those leftovers match. |

---

## 5. What we learned

**The timing is not the problem.** Re-derived from raw files on all five days: the reference
time matches to 0 microseconds, the TTL pulse times match to 0 nanoseconds, the Neuropixels and
NI clocks agree to within about 20 microseconds, and 100% of video frame times are exactly
camera pulses. The convention is identical to the old `030926` notebook.
(One caveat: an attempt to confirm this against a hardware artefact was *inconclusive*, not
passing - there simply isn't a sharp artefact to find.)

**The blurriness is jitter in when we say a reach starts.** A clock error would shift the
response but keep it sharp; these are broad instead. Re-aligning to the moment of furthest paw
extension, instead of the moment the paw crosses the threshold box, sharpens the population
response on 4 of 5 days. Working backwards from the widths, the reach-onset estimate is
scattered by roughly **half a second** - which fits the idea that mice make tentative
proto-reaches before the one the detector catches.

**Event count contaminates across-day comparisons, in both directions.** `z` gets bigger when
there are more reaches (it is a detection score, not a size). `delta Hz` gets *smaller* when
there are more reaches, because it is an absolute value and noisy estimates inflate it. So
**match the number of reaches before comparing days, whichever number you report.**

**Firing does track trial-to-trial differences in the movement**, modestly: 24 of 151 units on
Sept 16 versus 7.6 expected by chance, and it replicated on Sept 14, 15 and 17.

**No evidence that thalamus responds later than cortex** (medians identical, p = 0.29). Given
the half-second jitter, this comparison cannot really be made with this data either way.

**In Hz the two regions change by similar amounts; as a percentage, cortex changes more**,
because cortical baseline rates are lower. Pick one and say which.

---

## 6. Things to be careful about

- The reach zone is a hand-placed box, and it **moves every day** because the camera moves.
  That alone makes day-to-day comparison partly a camera comparison.
- The paw is tracked on only about a third of frames, and the tracked frames are mostly the
  reaches themselves. Any whole-session correlation inherits that.
- Never sort the reaches inside a single unit's raster by that unit's own firing. It produces
  a convincing-looking pattern out of pure noise. (The old `PETHforTopUnits` notebook did this.)
  Sorting *units* by response latency across a population is fine and standard.
- Region names are still target guesses pending histology.

---

## 7. Loose ends

- **Rename the drive** (needs an admin PowerShell; sticks across reboots):
  `Set-Partition -DiskNumber 1 -PartitionNumber 2 -NewDriveLetter K`
- The cross-day matching notebook points at `D:\Neuropixels data-motor project\...`, which does
  not exist on this PC. Those results are in `D:\4103_figures_backup_2026-10-09\unit_matching_sept15-18\`.
- Because of the jitter, a **window-average** change would be a more sensitive measure than the
  peak used in the current figures (jitter flattens peaks but preserves the total). Not done.
- Second mouse with better tracking and denser recordings: not looked at yet.
- The complicated controls in the kinematics stage exist because the tracking is rough. On clean
  data most of them should be deleted, not kept.
