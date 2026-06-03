# Peak Tracking User Guide

## Overview

The peak tracking workflow fits a configurable `lmfit` model to each peak in every frame of a GIWAXS time series, then integrates the fitted peak area and exports results to CSV. It is designed to stay stable when peaks drift, appear, or disappear over time.

**Core files:**
- `peak_tracking_workflow.py` — runner, fitting, integration, export, and plotting logic.
- `peak_tracking_config.py` — your experiment definitions (edit this to set up new runs).

---

## Quick Start

Install dependencies:

```powershell
uv sync
```

List available presets in the config:

```powershell
uv run python peak_tracking_workflow.py --list-runs
```

Run one preset:

```powershell
uv run python peak_tracking_workflow.py --run insitu_0.5M_MeOMBAI
```

Run all presets in parallel:

```powershell
uv run python peak_tracking_workflow.py --run-all --jobs 4
```

Use a different config file (for a different beamtime or experiment):

```powershell
uv run python peak_tracking_workflow.py --config my_config.py --run my_run_name
```

Override the output directory:

```powershell
uv run python peak_tracking_workflow.py --run insitu_0.5M_MeOMBAI --output-root "./local_results"
```

---

## How the Workflow Works

For each peak in each selected frame, the workflow:

1. **Loads** the `.nc` time series and applies an optional global 2θ shift.
2. **Searches** a window around the last known peak center.
3. **Pre-fit SNR check**: estimates signal-to-noise and skips fitting if no peak is likely present (`peak_absent`).
4. **Finds a candidate** peak using `scipy.signal.find_peaks` (smoothed, scored by proximity and prominence).
5. **Fits** the candidate with an `lmfit` model (Voigt + linear background by default). Adaptive bounds can constrain σ/γ/amplitude based on the previous frame.
6. **Post-fit validation**: rejects fits where the optimizer hit parameter bounds or the peak is too weak (`fit_invalid`).
7. **Integrates** raw area and background area over the fitted width; computes net area.
8. **Writes** per-frame results to CSV and plots key-frame fits as PNGs.

The center search anchors on the previous fitted center, so small frame-to-frame drift is handled automatically. Key frames (user-specified or auto-detected) reset the search to the expected center and always save a diagnostic PNG.

---

## Writing a Config File

A config file is a plain Python module that defines `RUN_CONFIGS` (and optionally `DEFAULT_OUTPUT_ROOT`).

```python
from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path(r"G:\...\peak_tracking_outputs")

RUN_CONFIGS = {
    "my_experiment": {
        "da_file": Path(r"G:\...\my_data.nc"),
        "global_shift_deg": 0.0,          # 2θ offset correction (see below)
        "keyframe_plot_margin_deg": 2.0,   # Plot window ± around integration region
        "peaks": [
            (30.6, 1.0, (0, -1), "ITO", {"drift_tolerance_deg": 0.05}),
            (13.3, 2.0, (0, -1), "PbI2", {"drift_tolerance_deg": 0.05}),
            (6.4,  1.0, (-1, 25), "2D (002)", {
                "drift_tolerance_deg": 0.05,
                "fitter": {"kind": "voigt_exp", "fit_half_width_deg": 0.4},
            }),
        ],
    },
}
```

Peaks can be specified as tuples `(center_deg, window_deg, (start_idx, stop_idx), name, extras_dict)` or as `PeakSpec` dataclass instances.

---

## PeakSpec Parameter Reference

### Core Localization Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `center_deg` | float | Expected 2θ position of the peak in degrees. Used as the initial search anchor at the start of tracking or after a key-frame reset. |
| `window_deg` | float | Half-width of the search window in degrees. Controls how far from the anchor to search for a candidate peak. |
| `drift_tolerance_deg` | float | Maximum allowed shift of the fitted center from the candidate center (default: 0.15°). Also bounds how far the center parameter can move during fitting. Set tighter for stable peaks (0.05°), wider for drifting peaks. |
| `reacquire_window_deg` | float\|None | If set, expands the search window after a lost frame during reacquisition (defaults to `drift_tolerance_deg × 3`). |
| `start_idx` | int | First frame index to process (negative = from end, e.g. `-1` is last frame). |
| `stop_idx` | int | Last frame index to process (inclusive). |
| `frame_step` | int | Step size between frames (default: 1, every frame). Set to e.g. 5 to skip frames. |

### How to Choose `center_deg` and `window_deg`

- **`center_deg`**: Inspect the diffractogram visually. Use the 2θ value of the peak maximum. It is OK to be off by ±0.1° — the algorithm will locate the exact peak.
- **`window_deg`**: Should be wide enough to contain the peak plus a little margin, but not so wide that a neighboring peak falls inside. A window of 1–2× the peak FWHM is typical. For a sharp peak (σ ≈ 0.05°), `window_deg = 0.5` is usually fine. For a broad peak, use 1.5–2.0°.
- **`drift_tolerance_deg`**: For substrate peaks (ITO) that barely move, use 0.05°. For perovskite peaks that may shift during crystallization or thermal cycling, use 0.1–0.15°.

### Global 2θ Shift

`global_shift_deg` shifts every frame's 2θ axis by a fixed offset before any processing. Use this to correct for a miscalibrated poni file or a systematic offset between calibrant and sample geometry. To find it: compare the known 2θ of a reference peak (e.g. ITO at 30.6°) with where the peak actually sits in the data.

Note: some `.nc` files from the 260214 session used a different calibration (`manual_calib.poni`) that shifts all peaks by approximately −0.94×. Those configs use `_260214_peaks()` helper functions which have the corrected center values.

### Frame Range

`(start_idx, stop_idx)` is an inclusive range. Use negative indices to index from the end of the time series (`-1` = last frame). Common patterns:

- `(0, -1)` — all frames (frame 0 through last)
- `(-1, 0)` — reverse direction (last frame first, useful for tracking backwards)
- `(-1, 25)` — from the end back to frame 25

### Key Frames

```python
"key_frames": [0, 100, 200]   # Force-reset tracking at these frame indices
"key_frames_mode": "additive"  # "additive" = process all frames plus reset at key frames
                               # "restrict" = only process key frames (no others)
"auto_keyframe_every": 15      # Also auto-reset every N consecutive tracked frames (default: 15)
```

Key frames cause the tracker to restart from `center_deg`. A fit plot PNG is always saved for key frames for human inspection.

### Integration Parameters

| Parameter | Description |
|-----------|-------------|
| `integration_height` | Fractional peak height at which to measure the width for integration bounds (default: 0.25 = 25% of peak height, i.e. roughly the half-maximum). |
| `integration_multiplier` | Multiplies the measured width to set the integration window. Default 2.0× to ensure full peak area is captured. |
| `max_integration_span_deg` | Hard cap on integration width in degrees (default: 3.0°). Prevents absurdly wide integration when fitting fails gracefully. |
| `peak_only_window_deg` | If set, restricts the fitting window to this half-width around the candidate. Useful when neighboring peaks are close. |

### Fitter Selection

```python
"fitter": {"kind": "voigt_linear"}   # Default: Voigt peak + linear background
"fitter": {"kind": "voigt_exp", "fit_half_width_deg": 0.4}   # Voigt + exponential background, fit within ±0.4°
"fitter": {"kind": "gaussian_linear"}
"fitter": {"kind": "voigt_constant"}
"fitter": {"kind": "pseudo_voigt_linear"}
```

**When to use which:**
- `voigt_linear`: Default for most GIWAXS peaks. Handles a smooth linear background.
- `voigt_exp`: Use when the background has an exponential tail (e.g. a rising background from a nearby amorphous hump). Also set `fit_half_width_deg` to a narrow window around the peak to avoid the exponential dominating.
- `gaussian_linear`: Slightly faster than Voigt; use if peaks are clearly Gaussian and Voigt parameters are unstable.

`fit_half_width_deg`: When set, restricts fitting to this half-width around the candidate center. Use for sharp peaks in a wide integration window to prevent the fit from being influenced by neighboring features.

### Baseline Correction

```python
"baseline_method": "snip"     # or "aspls", "modpoly", etc. (pybaselines methods)
"baseline_kwargs": {"max_half_window": 40}
```

Most GIWAXS 1D patterns do not need explicit baseline correction if a linear background component is in the model. Baseline correction (via `pybaselines`) is applied per-frame before fitting and is most useful for patterns with a strong curved background.

---

## Validation Parameters

These control the two-stage system that prevents false positives (fitting noise when no peak is present).

### Stage 1: Pre-fit SNR Check

```python
"min_snr": 3.0   # Minimum signal-to-noise ratio (default: 3.0)
```

Before fitting, the algorithm estimates SNR = (peak_max − baseline) / noise, where baseline is the 10th percentile and noise is the std of the bottom 25th percentile. If SNR is below `min_snr`, the frame is skipped with status `peak_absent`.

**How to tune `min_snr`:**
- Strong, always-present peaks (ITO, PbI2): use 4.0–5.0.
- Weak or intermittent peaks (MeOMBAI, ClMBAI): use 2.0–3.0.
- If `peak_absent` is firing on frames where you can visually see the peak, lower the threshold.
- If spurious fits are passing through, raise the threshold.

### Stage 2: Post-fit Quality Check

```python
"validate_fit_quality": True   # Default: True — highly recommended to keep on
```

After fitting, the algorithm rejects results where:
1. σ or γ hit their parameter bounds (within 5%) — optimizer struggled.
2. Amplitude < 1e-4 — essentially fitting nothing.
3. Peak height < 5% of data maximum — too weak.
4. Reduced chi-square > 20 — poor fit quality.

Failed fits are marked `fit_invalid`. The `validation_reason` CSV column explains why.

---

## Adaptive Parameter Bounds

By default the fitter uses wide parameter bounds every frame. Adaptive bounds tighten σ, γ, and amplitude constraints based on the previous frame's fitted values, which speeds up convergence for stable peaks.

```python
"use_adaptive_bounds": True,            # Enable (default: False)
"adaptive_min_consecutive": 5,          # Warm-up: wait for N consecutive good fits
"adaptive_sigma_tolerance": 0.5,        # Allow ±50% change in σ/γ
"adaptive_amplitude_tolerance": 0.5,    # Allow ±50% change in amplitude
"fallback_on_validation_failure": True, # Retry with wide bounds if validation fails
```

**When to use adaptive bounds:**

| Peak type | Recommendation | Reason |
|-----------|----------------|--------|
| Strong, always-present (ITO, PbI2) | Enable (`True`) | Smooth evolution; tight bounds → faster fitting |
| Weak / intermittent (MeOMBAI, ClMBAI) | Disable (`False`) | Peak appears/disappears; needs full flexibility |
| Perovskite peaks with transitions | Enable with wider tolerance (0.7) + key frames | Allows gradual change but resets at transitions |

---

## Output Files

Each run creates a timestamped folder:

```
<output_root>/<run_name>_YYYYMMDD_HHMMSS/
  tracking_results_long.csv    — one row per peak per frame, all columns
  tracking_results_compact.csv — reduced table with the most useful columns
  summary_peak_traces.png      — net area and normalized net area per peak
  summary_normalized_overlay.png — all peaks normalized on one plot
  run_config.json              — the configuration used for this run
  peak_tracking.log            — detailed log for debugging
  key_frames_<peak>.json       — which frames were treated as key frames
  key_frames_used.csv          — run-level summary of key frame usage
  fit_plots/                   — per-key-frame fit PNGs for inspection
    fit_<peak>_idx<N>.png
    failed_fit_<peak>_idx<N>.png
```

### CSV Columns

**Frame info:** `time`, `frame_index`, `is_key_frame`, `is_config_key_frame`, `is_auto_key_frame`, `peak_name`, `run_name`

**Fit geometry:** `expected_center_deg`, `candidate_center_deg`, `fit_center_deg`, `drift_tolerance_deg`, `global_shift_deg`

**Areas:** `raw_area`, `background_area`, `net_area`, `integration_left_deg`, `integration_right_deg`, `integration_width_deg`

**Normalized:** `normalized_raw_area`, `normalized_net_area` (each normalized to its own peak's maximum)

**Fit quality:** `fit_r2`, `fit_redchi`, `fit_rmse`, `fit_rel_rmse`, `fit_quality_score`

**Fit parameters:** `fit_amplitude`, `fit_sigma`, `fit_gamma`, `fit_slope`, `fit_offset`

**Status:** `fit_status`, `fit_success`, `fit_message`, `fit_error`

**Validation & adaptive:** `peak_snr`, `peak_presence_check`, `validation_reason`, `used_adaptive_bounds`, `consecutive_good_fits`

**Baseline:** `baseline_method`, `baseline_applied`, `raw_y_max`

### `fit_status` Values

| Status | Meaning | Action |
|--------|---------|--------|
| `track` | Successfully tracked and fitted | ✅ Normal |
| `search` | First successful fit (before tracking active) | ✅ Normal at start |
| `peak_absent` | SNR below threshold — no peak detected | ✅ Expected for intermittent peaks |
| `fit_invalid` | Fit succeeded but failed quality checks | ⚠️ Check `validation_reason` and `fit_plots/` |
| `no_candidate` | `find_peaks` found no candidate in window | ⚠️ Check `min_prominence` or `window_deg` |
| `lost` | Generic fit failure (exception during fitting) | ⚠️ Check `fit_error` column |

---

## Troubleshooting

**Peak not found in early frames:**
- If `start_idx = -1` (backwards tracking), the first frame processed is the last one in the series, where the peak may be better formed.
- Check `min_prominence` — lower it if peaks are being missed.

**Too many `fit_invalid` rejections:**
- Check `fit_plots/` for key frames to understand what's happening visually.
- If sigma hits bounds, `window_deg` may be too narrow (sigma_ceiling = max(window_deg, drift_tolerance_deg × 4)).
- Try lowering `min_snr` or adjusting `validate_fit_quality=False` temporarily to diagnose.

**Tracking lost after a phase transition:**
- Add a `key_frame` at the transition point to force re-anchor.
- Widen `reacquire_window_deg` to help reacquisition.

**Exponential background causing poor fits:**
- Switch to `"fitter": {"kind": "voigt_exp", "fit_half_width_deg": 0.4}` and set `fit_half_width_deg` to just wide enough to contain the peak.

**Calibration offset (peaks at wrong 2θ):**
- Set `global_shift_deg` to shift the axis. Positive shifts the axis up, negative shifts down.
- Or use `_260214_peaks()` style helpers with corrected center values if the shift varies by peak.
