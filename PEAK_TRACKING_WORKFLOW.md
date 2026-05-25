# Peak Tracking Workflow Guide

This project now includes a config-driven peak tracking workflow for GIWAXS time series.

It replaces the older noisy integration path with a repeatable pipeline that:

- tracks each peak frame by frame,
- fits a pluggable `lmfit` model,
- integrates the fitted peak region,
- exports CSV tables and summary plots,
- saves debug artifacts under `agent_temp`.

## Files to know

- `peak_tracking_workflow.py`: main runner, fitting, integration, export, and plotting.
- `peak_tracking_config.py`: preset run definitions and peak settings.
- `vogit_width.py`: helper used to estimate the integration width.

## Quick start

Install dependencies first:

```powershell
uv sync
```

List the available presets:

```powershell
uv run python peak_tracking_workflow.py --list-runs
```

Run one preset:

```powershell
uv run python peak_tracking_workflow.py --run "MeOMBAI_120c_aging_1 004868 Images"
```

Or run the MBAI test dataset:

```powershell
uv run python peak_tracking_workflow.py --run "MBAI_0M5_3 004909 Images"
```

You can override the output root if needed:

```powershell
uv run python peak_tracking_workflow.py --run "MeOMBAI_120c_aging_1 004868 Images" --output-root "G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\agent_temp"
```

## What the workflow does

For each peak and each selected frame, the workflow:

1. loads the NetCDF time series,
2. optionally applies a baseline correction,
3. searches near the last known peak center first,
4. fits the selected `lmfit` model,
5. rejects fits that drift too far,
6. integrates the raw peak area and background area,
7. writes raw and normalized results to CSV.

The result is designed to stay stable even when peaks drift slightly over time.

## Output files

Each run creates a timestamped folder under `agent_temp`, for example:

```text
G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\agent_temp\MeOMBAI_120c_aging_1 004868 Images_YYYYMMDD_HHMMSS
```

Typical contents:

- `tracking_results_long.csv`: one row per peak per frame with all fit details.
- `tracking_results_compact.csv`: reduced table with the most useful columns.
- `summary_peak_traces.png`: net area and normalized net area per peak.
- `summary_normalized_overlay.png`: all normalized traces on one plot.
- `run_config.json`: saved configuration used for the run.
- `peak_tracking.log`: detailed log for debugging.
 - `fit_plots/`: folder containing per-key-frame fit PNGs for human inspection (one file per key frame).

## CSV columns

The long CSV includes:

- frame and time information,
- expected center, candidate center, and fitted center,
- fit success/status/message,
- raw area, background area, and net area,
- integration bounds,
- normalization columns,
- baseline and model metadata.
- `is_key_frame`: boolean, True if this row corresponds to a configured `key_frames` entry for the peak.

The compact CSV includes the same run in a smaller, easier-to-scan format.

## Editing presets

Open `peak_tracking_config.py` to change a preset.

Each peak is defined as a tuple like this:

```python
(center_deg, window_deg, (start_idx, stop_idx), name, extras)
```

Example:

```python
(6.0, 0.5, (-1, 0), "2D (002)", {"drift_tolerance_deg": 0.15})
```

Meaning:

- `center_deg`: starting peak position in degrees.
- `window_deg`: search window size.
- `(start_idx, stop_idx)`: frame range; negative values work from the end.
- `name`: label written to CSV and plots.
- `extras`: optional settings such as `drift_tolerance_deg`.

Notes about `key_frames` and run outputs:

- If you set `key_frames` on a peak (a list/tuple of frame indices), the tracker will only process those frames for that peak. The configured `key_frames` are saved in `run_config.json` (under each peak entry) and individual rows in `tracking_results_long.csv` include an `is_key_frame` column indicating which rows correspond to those configured key frames.

Auto-detected key frames

- The tracker now also auto-detects key frames for human inspection. These are marked in the outputs as `is_auto_key_frame` (and the combined `is_key_frame` will be True if either user-configured or auto-detected). The auto-detection rules are:
    - Initial detection: the first frame where the peak is successfully located and fitted.
    - Reacquire: the frame where a previously-lost peak is found again after one or more failed frames.
    - Every-N frames: when a peak has been tracked successfully for `N` consecutive frames without dropping, the frame at the Nth step is marked. The default `N` is 15 and can be set per-peak with the `auto_keyframe_every` extra.

- The tracker saves a per-peak summary JSON in the run folder named `key_frames_<peak_name>.json` containing the configured and auto-detected key frame lists.

User-supplied key frames are now additive (default)

- Previously, specifying `key_frames` caused the tracker to only process those frames for a peak. That behavior was restrictive. Now `key_frames` are additive by default: the tracker will process the full configured frame range (respecting `frame_step`) and will *also* mark and force inspection at any user-specified `key_frames`.

- Manual force reset behavior at configured key frames:
    - When a configured key frame is encountered the tracker forcibly drops the current tracked peak (if active), resets the search anchor to the peak's configured `center_deg`, and attempts to redetect and refit at that frame. This lets a human-defined key frame act as a manual reinitialization point.

Keyframe plot margin

- The x-axis margin for key-frame fit plots is configurable at the run level with `keyframe_plot_margin_deg` (default 5.0°). The plot x-limits use the integration interval ± this margin for focused inspection.
You can also define peaks as dictionaries if you prefer named fields.

## Useful per-peak settings

- `drift_tolerance_deg`: maximum allowed shift from the **current frame's** candidate center; it is checked per frame, not accumulated across the whole run.

Example:

- frame 1 candidate is `5.4°` and the fit center is `5.42°` → accept.
- frame 2 candidate is `5.7°` and the fit center is `5.69°` → accept if the difference is within `drift_tolerance_deg`.
- frame 2 candidate is `5.4°` but the fit tries to jump to `5.7°` → reject if that exceeds `drift_tolerance_deg`.

So the check is always local to the current frame's candidate, not a running total across frames.
- `reacquire_window_deg`: wider search window when a peak is temporarily lost.
- `frame_step`: track every X frames inside the configured frame range.
- `key_frames`: track only the listed frame indices.
- `integration_height`: height used to estimate the integration span.
- `integration_multiplier`: scales the integration width.
- `max_integration_span_deg`: hard cap on the integration width.
- `min_prominence`: minimum prominence for candidate peak detection.
- `baseline_method`: optional `pybaselines` method name.
- `fitter`: model selection or custom factory hook.

## Built-in fitter kinds

The workflow includes a small registry of built-in model bundles:

- `voigt_linear`
- `voigt_constant`
- `gaussian_linear`
- `pseudo_voigt_linear`
- `voigt`

## Custom model hook

If you want a different `lmfit` model, set `fitter` to a custom factory.

Use either:

- a callable, or
- a string in `module:function` form.

The factory must return a `FitBundle`.

Example concept:

```python
{
    "name": "My Peak",
    "center_deg": 7.2,
    "window_deg": 0.4,
    "start_idx": -1,
    "stop_idx": 0,
    "fitter": {
        "kind": "voigt_linear",
        "factory": "my_models:build_peak_bundle",
    },
}
```

Your factory should build and return the model bundle used by the tracker.

## Tips for good results

- Keep `drift_tolerance_deg` tight enough to prevent jumping to a neighboring peak.
- Increase `reacquire_window_deg` if a peak occasionally disappears.
- Reduce `integration_multiplier` if the integrated region is too broad.
- Use `peak_tracking.log` first when debugging a failed fit.
- Check `tracking_results_long.csv` to see exactly where a peak was lost.

## Troubleshooting

- If no rows are produced, confirm the preset name with `--list-runs`.
- If the run fails immediately, confirm the NetCDF file exists in the path from `peak_tracking_config.py`.
- If a fit keeps jumping, lower `drift_tolerance_deg` and narrow the search window.
- If the background looks wrong, try a different built-in fitter or add a custom factory.

## Recommended workflow

1. Start from one of the presets in `peak_tracking_config.py`.
2. Run the workflow once and inspect the CSV and plots.
3. Adjust peak windows or drift tolerances.
4. Add a custom fitter only when the built-in models are not enough.
