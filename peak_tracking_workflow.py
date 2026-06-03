"""Peak tracking and integration workflow for GIWAXS time series.

This module provides a config-driven pipeline that, for each peak in a
NetCDF time series:

1. Searches a 2θ window around the last known peak center.
2. Performs a pre-fit SNR check to detect peak absence without fitting.
3. Fits an lmfit model (Voigt + background by default).
4. Validates fit quality to reject optimizer-converged-but-unphysical results.
5. Integrates raw area, background area, and net area over the fitted peak width.
6. Exports CSV tables and summary plots, saving fit PNGs for key frames.

Entry point::

    uv run python peak_tracking_workflow.py --run <preset_name>
    uv run python peak_tracking_workflow.py --run-all --jobs 4
    uv run python peak_tracking_workflow.py --list-runs

See PEAK_TRACKING_GUIDE.md for full user documentation.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from lmfit import Model
from lmfit.models import ConstantModel, GaussianModel, LinearModel, PseudoVoigtModel, VoigtModel
from scipy.signal import find_peaks, peak_prominences
from tqdm import tqdm

from vogit_width import voigt_width_at_height


DEFAULT_WAVELENGTH_A = 1.5418
logger = logging.getLogger(__name__)


def load_config_module(config_path: Path) -> tuple[Path, dict[str, Any]]:
    """
    Dynamically load a peak tracking configuration module.
    
    Args:
        config_path: Path to the config .py file
    
    Returns:
        (DEFAULT_OUTPUT_ROOT, RUN_CONFIGS) tuple
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    spec = importlib.util.spec_from_file_location("peak_tracking_config_dynamic", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["peak_tracking_config_dynamic"] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, "RUN_CONFIGS"):
        raise AttributeError(f"Config file {config_path} must define RUN_CONFIGS")
    
    default_output_root = getattr(module, "DEFAULT_OUTPUT_ROOT", Path("output"))
    run_configs = getattr(module, "RUN_CONFIGS")
    
    return Path(default_output_root), dict(run_configs)


@dataclass(frozen=True)
class FitterSpec:
    """Specification for the lmfit model used to fit a single peak.

    Attributes:
        kind: Predefined model name. One of: ``voigt_linear``, ``voigt_constant``,
            ``voigt_exp``, ``gaussian_linear``, ``pseudo_voigt_linear``,
            ``double_voigt_linear``, ``voigt``. Default ``voigt_linear``.
        factory: Optional callable or ``"module:function"`` string for a custom
            model factory. The factory must accept ``peak_spec`` as a keyword
            argument and return a :class:`FitBundle`.
        kwargs: Extra keyword arguments forwarded to the factory.
        fit_half_width_deg: If set, restricts the fitting window to this
            half-width around the candidate center. Useful when a sharp peak
            sits inside a wider integration window so that the fit is not
            influenced by distant background features.
    """
    kind: str = "voigt_linear"
    factory: str | Callable[..., Any] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    fit_half_width_deg: float | None = None


@dataclass(frozen=True)
class PeakSpec:
    """Complete specification for tracking and fitting one peak in a time series.

    All angular values are in degrees (2θ).

    Core localization:
        name: Human-readable peak label used in CSV output and plot titles.
        center_deg: Expected 2θ position; used as the initial search anchor and
            for key-frame resets.
        window_deg: Half-width of the candidate search window. Should be wide
            enough to contain the peak plus a small margin, but not so wide that
            a neighboring peak falls inside.
        start_idx / stop_idx: Inclusive frame range (negative = from end).
        drift_tolerance_deg: Maximum allowed shift of the fitted center from the
            candidate center. Also sets the bounds on the ``peak_center``
            parameter during fitting. Tight for substrate peaks (0.05°), wider
            for drifting perovskite peaks (0.10–0.15°).
        reacquire_window_deg: Expanded search half-width during reacquisition
            after a lost frame (defaults to ``drift_tolerance_deg × 3``).
        frame_step: Process every N-th frame (default 1 = every frame).

    Key frames:
        key_frames: Tuple of frame indices at which to force-reset tracking to
            ``center_deg``. A fit PNG is always saved at these frames.
        key_frames_mode: ``"additive"`` (default) processes all frames and also
            resets at key frames. ``"restrict"`` processes only key frames.
        auto_keyframe_every: Auto-reset tracking and save a fit PNG every N
            consecutive successfully tracked frames (default 15).

    Integration:
        integration_height: Fractional peak height at which to measure the Voigt
            width for setting integration bounds (default 0.25, i.e. roughly the
            quarter-maximum width).
        integration_multiplier: Multiplier applied to the measured width
            (default 2.0) to ensure the full peak area is captured.
        max_integration_span_deg: Hard cap on integration width in degrees
            (default 3.0°).
        min_prominence: Minimum prominence required for ``scipy.signal.find_peaks``
            to accept a candidate (default 0.0 = no filter).
        peak_only_window_deg: If set, restricts the fitting window to this
            half-width regardless of ``fitter.fit_half_width_deg``.

    Model:
        fitter: :class:`FitterSpec` controlling the lmfit model.
        baseline_method: Optional ``pybaselines`` method name applied per-frame
            before fitting (e.g. ``"snip"``, ``"aspls"``).
        baseline_kwargs: Extra keyword arguments forwarded to the baseline method.

    Validation:
        min_snr: Minimum signal-to-noise ratio for the pre-fit peak presence
            check. Frames below this threshold are marked ``peak_absent`` and
            skipped. Lower for weak peaks (2.0–2.5), higher for strong (4.0–5.0).
        validate_fit_quality: When True (default), apply post-fit quality checks
            that reject fits where the optimizer hit bounds, amplitude is near
            zero, the peak is too weak, or reduced chi-square > 20.

    Adaptive bounds (optional, disabled by default):
        use_adaptive_bounds: Constrain σ/γ/amplitude to ± tolerance of the
            previous frame's values after a warm-up period. Enable for strong,
            always-present peaks; disable for weak/intermittent peaks.
        adaptive_sigma_tolerance: Allowed fractional change in σ and γ (default
            0.5 = ±50%).
        adaptive_amplitude_tolerance: Allowed fractional change in amplitude
            (default 0.5 = ±50%).
        adaptive_min_consecutive: Number of consecutive successful fits required
            before adaptive bounds activate (default 5).
        fallback_on_validation_failure: If True (default), retry with wide bounds
            when validation fails under adaptive constraints.
    """
    name: str
    center_deg: float
    window_deg: float
    start_idx: int
    stop_idx: int
    drift_tolerance_deg: float = 0.15
    reacquire_window_deg: float | None = None
    frame_step: int = 1
    key_frames: tuple[int, ...] | None = None
    key_frames_mode: str = "additive"
    integration_height: float = 0.25
    integration_multiplier: float = 2.0
    max_integration_span_deg: float = 3.0
    min_prominence: float = 0.0
    fitter: FitterSpec = field(default_factory=FitterSpec)
    baseline_method: str | None = None
    baseline_kwargs: dict[str, Any] = field(default_factory=dict)
    peak_only_window_deg: float | None = None
    auto_keyframe_every: int | None = 15
    # Peak presence validation
    min_snr: float = 3.0  # Minimum signal-to-noise ratio to consider peak present
    validate_fit_quality: bool = True  # Enable post-fit validation
    # Adaptive parameter bounds
    use_adaptive_bounds: bool = False  # Enable adaptive parameter constraints
    adaptive_sigma_tolerance: float = 0.5  # Allowed fractional change in sigma/gamma (50%)
    adaptive_amplitude_tolerance: float = 0.5  # Allowed fractional change in amplitude (50%)
    adaptive_min_consecutive: int = 5  # Consecutive successful fits before enabling adaptive bounds
    fallback_on_validation_failure: bool = True  # Retry with wide bounds if validation fails


@dataclass(frozen=True)
class RunConfig:
    """Complete configuration for one tracking run.

    Attributes:
        name: Run identifier used in output folder names and CSV ``run_name`` column.
        da_file: Path to the NetCDF ``.nc`` file. Must contain a ``time`` dimension
            and either a ``twoTheta_deg`` or ``q_A^-1`` coordinate.
        peaks: List of :class:`PeakSpec` objects; one entry per peak to track.
        global_shift_deg: Fixed 2θ offset applied to every frame before any
            processing. Use to correct a miscalibrated poni file.
        output_root: Root directory for timestamped output folders.
        baseline_method: Run-level baseline method applied to all peaks unless
            overridden by an individual :class:`PeakSpec`.
        baseline_kwargs: Run-level baseline kwargs.
        wavelength_a: X-ray wavelength in Ångströms used for q→2θ conversion
            (default Cu Kα = 1.5418 Å).
        keyframe_plot_margin_deg: Plot window half-width (in 2θ) around the
            integration region in key-frame fit PNGs (default 5.0°).
    """
    name: str
    da_file: Path
    peaks: list[PeakSpec]
    global_shift_deg: float = 0.0
    output_root: Path = field(default_factory=lambda: Path("output"))
    baseline_method: str | None = None
    baseline_kwargs: dict[str, Any] = field(default_factory=dict)
    wavelength_a: float = DEFAULT_WAVELENGTH_A
    keyframe_plot_margin_deg: float = 5.0


@dataclass
class FitBundle:
    """Container pairing an lmfit composite model with metadata.

    Attributes:
        model: The composite lmfit Model (e.g. VoigtModel + LinearModel).
        fit_kind: The ``FitterSpec.kind`` string used to build this model.
        component_names: Tuple of model component prefixes (e.g.
            ``("peak_", "bkg_")``). Used to separate peak from background when
            computing net area.
    """
    model: Model
    fit_kind: str
    component_names: tuple[str, ...] = ()


@dataclass
class TrackingState:
    """Mutable state carried frame-to-frame during peak tracking.

    Attributes:
        last_center: Fitted 2θ center from the previous successful frame.
            Used as the search anchor for the next frame.
        last_sigma: Fitted σ from the previous successful frame; used for
            adaptive bounds.
        last_gamma: Fitted γ from the previous successful frame; used for
            adaptive bounds.
        last_amplitude: Fitted amplitude from the previous successful frame;
            used for adaptive bounds.
        tracking_active: True once the first successful fit has been obtained.
        lost_count: Number of consecutive frames without a successful fit.
        consecutive_good_fits: Consecutive successful fits since last loss or
            key-frame reset. Counts toward the adaptive bounds warm-up.
        peak_present: Whether the peak was physically present in the last
            processed frame (as opposed to a fit failure on a present peak).
    """
    last_center: float | None = None
    last_sigma: float | None = None
    last_gamma: float | None = None
    last_amplitude: float | None = None
    tracking_active: bool = False
    lost_count: int = 0
    consecutive_good_fits: int = 0  # Count for enabling adaptive bounds
    peak_present: bool = True  # Track if peak is physically present vs fit failure


def configure_logging(run_dir: Path) -> None:
    """Set up root logger to write to both console and a log file in *run_dir*."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "peak_tracking.log"
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logger.info("Logging to %s", log_path)


def _as_dataarray(data: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """Extract an intensity DataArray from a Dataset, trying common variable names."""
    if isinstance(data, xr.DataArray):
        return data

    for preferred_name in ("sample_norm_int", "intensity"):
        if preferred_name in data.data_vars:
            return data[preferred_name]

    if len(data.data_vars) == 1:
        return next(iter(data.data_vars.values()))

    raise ValueError("Could not identify an intensity data variable in the dataset.")


def load_time_series(da_file: Path, wavelength_a: float = DEFAULT_WAVELENGTH_A) -> xr.DataArray:
    """Load a NetCDF time series and ensure it has ``twoTheta_deg`` and ``time`` dimensions.

    If the file has a ``q_A^-1`` coordinate instead of ``twoTheta_deg``, it is
    converted using Bragg's law: 2θ = 2 arcsin(q λ / 4π).

    Args:
        da_file: Path to a NetCDF file (.nc).
        wavelength_a: X-ray wavelength in Ångströms for q→2θ conversion.

    Returns:
        DataArray sorted by time and 2θ.
    """
    if not da_file.is_file():
        raise FileNotFoundError(f"NetCDF file not found: {da_file}")

    try:
        data = xr.open_dataarray(da_file)
    except Exception:
        data = _as_dataarray(xr.open_dataset(da_file))

    if "twoTheta_deg" not in data.coords:
        if "q_A^-1" not in data.coords:
            raise ValueError("Input data must contain either 'twoTheta_deg' or 'q_A^-1'.")
        two_theta = np.rad2deg(2 * np.arcsin(data["q_A^-1"] * wavelength_a / (4 * np.pi)))
        data = data.assign_coords({"twoTheta_deg": ("q_A^-1", np.asarray(two_theta))}).swap_dims({"q_A^-1": "twoTheta_deg"})

    if "time" not in data.dims:
        raise ValueError("Input data must have a 'time' dimension.")

    if "twoTheta_deg" not in data.dims:
        raise ValueError("Input data must have a 'twoTheta_deg' dimension after loading.")

    return data.sortby("time").sortby("twoTheta_deg")


def apply_global_shift(data: xr.DataArray, shift_deg: float) -> xr.DataArray:
    """Shift the ``twoTheta_deg`` coordinate of *data* by *shift_deg* degrees.

    Used to correct a systematic 2θ offset from a miscalibrated poni file.
    A no-op when *shift_deg* is 0.
    """
    if shift_deg == 0:
        return data
    return data.assign_coords(twoTheta_deg=data.twoTheta_deg + shift_deg)


def baseline_correct_frame(frame: xr.DataArray, method: str | None, **kwargs: Any) -> tuple[xr.DataArray, xr.DataArray | None]:
    """Apply a ``pybaselines`` baseline correction to a single frame.

    Args:
        frame: 1-D intensity DataArray for one time step.
        method: Name of a ``pybaselines.Baseline`` method (e.g. ``"snip"``).
            ``None`` skips correction.
        **kwargs: Forwarded to the baseline method.

    Returns:
        Tuple of (baseline-corrected frame, baseline DataArray). The baseline
        DataArray is ``None`` when no correction is applied.
    """
    if not method:
        return frame, None

    from pybaselines import Baseline

    baseline_finder = Baseline()
    baseline_method = getattr(baseline_finder, method)
    baseline, _ = baseline_method(frame.values, **kwargs)
    corrected = xr.DataArray(frame.values - baseline, coords=frame.coords, dims=frame.dims)
    baseline_da = xr.DataArray(baseline, coords=frame.coords, dims=frame.dims)
    return corrected, baseline_da


def normalize_peak_entries(entries: Iterable[Any]) -> list[PeakSpec]:
    """Convert a mixed list of peak definitions into a uniform list of :class:`PeakSpec`.

    Accepts three input formats per entry:

    * A :class:`PeakSpec` instance (passed through unchanged).
    * A ``dict`` with keys matching :class:`PeakSpec` field names.
    * A tuple/list ``(center_deg, window_deg, (start_idx, stop_idx), name)``
      or ``(center_deg, window_deg, (start_idx, stop_idx), name, extras_dict)``
      where *extras_dict* holds optional PeakSpec fields.

    All missing optional fields receive their :class:`PeakSpec` defaults.
    """
    normalized: list[PeakSpec] = []
    for entry in entries:
        if isinstance(entry, PeakSpec):
            normalized.append(entry)
            continue

        if isinstance(entry, dict):
            fitter = entry.get("fitter", "voigt_linear")
            fitter_spec = fitter if isinstance(fitter, FitterSpec) else FitterSpec(**fitter) if isinstance(fitter, dict) else FitterSpec(kind=str(fitter))
            normalized.append(
                PeakSpec(
                    name=str(entry["name"]),
                    center_deg=float(entry["center_deg"]),
                    window_deg=float(entry["window_deg"]),
                    start_idx=int(entry["start_idx"]),
                    stop_idx=int(entry["stop_idx"]),
                    drift_tolerance_deg=float(entry.get("drift_tolerance_deg", 0.15)),
                    reacquire_window_deg=entry.get("reacquire_window_deg"),
                    frame_step=max(int(entry.get("frame_step", 1)), 1),
                    key_frames=tuple(int(idx) for idx in entry.get("key_frames", [])) or None,
                    key_frames_mode=str(entry.get("key_frames_mode", "additive")),
                    integration_height=float(entry.get("integration_height", 0.25)),
                    integration_multiplier=float(entry.get("integration_multiplier", 2.0)),
                    max_integration_span_deg=float(entry.get("max_integration_span_deg", 3.0)),
                    min_prominence=float(entry.get("min_prominence", 0.0)),
                    fitter=fitter_spec,
                    baseline_method=entry.get("baseline_method"),
                    baseline_kwargs=dict(entry.get("baseline_kwargs", {})),
                    peak_only_window_deg=entry.get("peak_only_window_deg"),
                    auto_keyframe_every=entry.get("auto_keyframe_every", 15),
                    # Validation & adaptive bounds parameters
                    min_snr=float(entry.get("min_snr", 3.0)),
                    validate_fit_quality=bool(entry.get("validate_fit_quality", True)),
                    use_adaptive_bounds=bool(entry.get("use_adaptive_bounds", False)),
                    adaptive_sigma_tolerance=float(entry.get("adaptive_sigma_tolerance", 0.5)),
                    adaptive_amplitude_tolerance=float(entry.get("adaptive_amplitude_tolerance", 0.5)),
                    adaptive_min_consecutive=int(entry.get("adaptive_min_consecutive", 5)),
                    fallback_on_validation_failure=bool(entry.get("fallback_on_validation_failure", True)),
                )
            )
            continue

        if isinstance(entry, (tuple, list)):
            if len(entry) not in {4, 5}:
                raise ValueError("Peak tuple entries must be (center, window, (start_idx, stop_idx), name[, extras]).")
            center, window, frame_range, name = entry[:4]
            extras = dict(entry[4]) if len(entry) == 5 and isinstance(entry[4], dict) else {}
            fitter = extras.get("fitter", "voigt_linear")
            fitter_spec = fitter if isinstance(fitter, FitterSpec) else FitterSpec(**fitter) if isinstance(fitter, dict) else FitterSpec(kind=str(fitter))
            normalized.append(
                PeakSpec(
                    name=str(name),
                    center_deg=float(center),
                    window_deg=float(window),
                    start_idx=int(frame_range[0]),
                    stop_idx=int(frame_range[1]),
                    drift_tolerance_deg=float(extras.get("drift_tolerance_deg", 0.15)),
                    reacquire_window_deg=extras.get("reacquire_window_deg"),
                    frame_step=max(int(extras.get("frame_step", 1)), 1),
                    key_frames=tuple(int(idx) for idx in extras.get("key_frames", [])) or None,
                    key_frames_mode=str(extras.get("key_frames_mode", "additive")),
                    integration_height=float(extras.get("integration_height", 0.25)),
                    integration_multiplier=float(extras.get("integration_multiplier", 2.0)),
                    max_integration_span_deg=float(extras.get("max_integration_span_deg", 3.0)),
                    min_prominence=float(extras.get("min_prominence", 0.0)),
                    fitter=fitter_spec,
                    baseline_method=extras.get("baseline_method"),
                    baseline_kwargs=dict(extras.get("baseline_kwargs", {})),
                    peak_only_window_deg=extras.get("peak_only_window_deg"),
                    auto_keyframe_every=extras.get("auto_keyframe_every", 15),
                    # Validation & adaptive bounds parameters
                    min_snr=float(extras.get("min_snr", 3.0)),
                    validate_fit_quality=bool(extras.get("validate_fit_quality", True)),
                    use_adaptive_bounds=bool(extras.get("use_adaptive_bounds", False)),
                    adaptive_sigma_tolerance=float(extras.get("adaptive_sigma_tolerance", 0.5)),
                    adaptive_amplitude_tolerance=float(extras.get("adaptive_amplitude_tolerance", 0.5)),
                    adaptive_min_consecutive=int(extras.get("adaptive_min_consecutive", 5)),
                    fallback_on_validation_failure=bool(extras.get("fallback_on_validation_failure", True)),
                )
            )
            continue

        raise TypeError(f"Unsupported peak entry: {type(entry)!r}")

    return normalized


def normalize_run_config(name: str, config: dict[str, Any], default_output_root: Path | None = None) -> RunConfig:
    """
    Normalize a run config dictionary into a RunConfig object.
    
    Args:
        name: Name of the run
        config: Config dictionary
        default_output_root: Default output root if not specified in config
    
    Returns:
        RunConfig object
    """
    if default_output_root is None:
        default_output_root = Path("output")
    
    return RunConfig(
        name=name,
        da_file=Path(config["da_file"]),
        peaks=normalize_peak_entries(config["peaks"]),
        global_shift_deg=float(config.get("global_shift_deg", 0.0)),
        output_root=Path(config.get("output_root", default_output_root)),
        baseline_method=config.get("baseline_method"),
        baseline_kwargs=dict(config.get("baseline_kwargs", {})),
        wavelength_a=float(config.get("wavelength_a", DEFAULT_WAVELENGTH_A)),
        keyframe_plot_margin_deg=float(config.get("keyframe_plot_margin_deg", 5.0)),
    )


def inclusive_frame_indices(start_idx: int, stop_idx: int, size: int) -> list[int]:
    """Return an inclusive list of frame indices between *start_idx* and *stop_idx*.

    Supports negative indices (Python-style, from end of series). The direction
    of the returned list follows the sign of ``stop_idx - start_idx``.
    """
    start = start_idx if start_idx >= 0 else size + start_idx
    stop = stop_idx if stop_idx >= 0 else size + stop_idx
    if start < 0 or start >= size or stop < 0 or stop >= size:
        raise IndexError(f"Frame range ({start_idx}, {stop_idx}) is outside series length {size}.")
    step = 1 if stop >= start else -1
    return list(range(start, stop + step, step))


def local_window(data: xr.DataArray, center: float, half_width: float) -> xr.DataArray:
    """Slice *data* to the 2θ interval [center − half_width, center + half_width]."""
    return data.sel(twoTheta_deg=slice(center - half_width, center + half_width))


def choose_candidate(
    window_data: xr.DataArray,
    center_guess: float,
    min_prominence: float = 0.0,
) -> dict[str, float] | None:
    """Find the most plausible peak candidate within *window_data*.

    Applies a rolling-mean smooth then uses ``scipy.signal.find_peaks`` to
    locate peaks. Ranks candidates by a score that favours proximity to
    *center_guess* and high prominence. Returns the best candidate or ``None``
    if no peaks are found.

    Args:
        window_data: 1-D DataArray slice to search.
        center_guess: Expected 2θ position used for proximity scoring.
        min_prominence: Minimum peak prominence passed to ``find_peaks``.

    Returns:
        Dict with ``center``, ``intensity``, and ``prominence`` keys, or
        ``None`` if no candidate was found.
    """
    x_vals = window_data.twoTheta_deg.values
    y_vals = window_data.values
    finite_mask = np.isfinite(y_vals)
    x_vals = x_vals[finite_mask]
    y_vals = y_vals[finite_mask]

    if len(x_vals) < 3:
        return None

    smoothed = pd.Series(y_vals).rolling(window=min(7, len(y_vals)), center=True, min_periods=1).mean().to_numpy()
    peak_indices, properties = find_peaks(smoothed, prominence=min_prominence)
    if len(peak_indices) == 0:
        return None

    prominences = np.asarray(properties.get("prominences", np.zeros_like(peak_indices, dtype=float)), dtype=float)
    candidate_positions = x_vals[peak_indices]
    # Prefer the peak closest to the previous center so reacquisition stays local.
    distance = np.abs(candidate_positions - center_guess)
    if len(prominences) and np.isfinite(prominences).any():
        prominence_scale = max(float(np.nanmax(prominences)), 1e-6)
    else:
        prominence_scale = 1.0
    score = distance / max(prominence_scale, 1e-6) - (prominences / max(prominence_scale, 1e-6)) * 0.05
    best_idx = int(peak_indices[int(np.nanargmin(score))])
    try:
        prominence, _, _ = peak_prominences(smoothed, np.array([best_idx]))
        prom_value = float(prominence[0])
    except Exception:
        prom_value = float(prominences[int(np.argmax(prominences))]) if len(prominences) else 0.0

    return {
        "center": float(x_vals[best_idx]),
        "intensity": float(y_vals[best_idx]),
        "prominence": prom_value,
    }


def check_peak_presence(window_data: xr.DataArray, min_snr: float = 3.0) -> tuple[bool, str, float]:
    """
    Stage 1 validation: Check BEFORE fitting if a real peak exists above noise.
    
    Args:
        window_data: Data window to check for peak presence
        min_snr: Minimum signal-to-noise ratio required
    
    Returns:
        (has_peak: bool, reason: str, snr: float)
    """
    y_vals = window_data.values
    finite_mask = np.isfinite(y_vals)
    y_clean = y_vals[finite_mask]
    
    if len(y_clean) < 3:
        return False, "Insufficient data points", 0.0
    
    # Estimate baseline and noise from lower percentiles
    baseline = float(np.percentile(y_clean, 10))  # Bottom 10% as baseline
    noise = float(np.std(y_clean[y_clean < np.percentile(y_clean, 25)]))
    
    # Find maximum signal
    peak_max = float(np.max(y_clean))
    
    # Calculate SNR
    snr = (peak_max - baseline) / noise if noise > 1e-6 else 0.0
    
    if snr < min_snr:
        return False, f"SNR={snr:.2f} below threshold {min_snr}", snr
    
    return True, "Peak detected", snr


def validate_fit_quality(
    fit_res: Any,
    peak_spec: PeakSpec,
    window_data: xr.DataArray,
) -> tuple[bool, str]:
    """
    Stage 2 validation: Check AFTER fitting if result represents a real peak.
    
    Rejects fits where:
    - Sigma/gamma hit parameter bounds (optimizer struggling)
    - Amplitude is essentially zero (fitting noise)
    - Peak height is much smaller than data range (weak/absent peak)
    - Fit quality is poor (high reduced chi-square)
    
    Args:
        fit_res: lmfit fit result object
        peak_spec: Peak specification with constraints
        window_data: Original data window used for fitting
    
    Returns:
        (is_valid: bool, reason: str)
    """
    best = fit_res.best_values
    
    # 1. Check if sigma hit bounds → optimizer struggling
    sigma = float(best.get("peak_sigma", 0))
    _fit_hw = peak_spec.fitter.fit_half_width_deg
    effective_window = min(peak_spec.window_deg, 2 * _fit_hw) if _fit_hw else peak_spec.window_deg
    sigma_floor = max(0.01, effective_window / 50)
    sigma_ceiling = max(effective_window, peak_spec.drift_tolerance_deg * 4)
    
    if sigma <= sigma_floor * 1.05:
        return False, f"Sigma at lower bound: {sigma:.4f} ≤ {sigma_floor*1.05:.4f}"
    if sigma >= sigma_ceiling * 0.95:
        return False, f"Sigma at upper bound: {sigma:.4f} ≥ {sigma_ceiling*0.95:.4f}"
    
    # 2. Check if amplitude is essentially zero
    amplitude = float(best.get("peak_amplitude", 0))
    if amplitude < 1e-4:
        return False, f"Amplitude near zero: {amplitude:.6f}"
    
    # 3. Check peak height relative to data scale
    gamma = float(best.get("peak_gamma", 1))
    # For Voigt: approximate height = amplitude / (π * gamma)
    peak_height = amplitude / (np.pi * gamma) if gamma > 1e-6 else 0
    data_vals = window_data.values[np.isfinite(window_data.values)]
    if len(data_vals) > 0:
        data_max = float(np.max(data_vals))
        if peak_height < data_max * 0.05:  # Peak < 5% of max signal
            return False, f"Peak too weak: height={peak_height:.3f} vs data_max={data_max:.3f}"
    
    # 4. Check fit quality (if available)
    if hasattr(fit_res, "redchi") and np.isfinite(fit_res.redchi):
        if fit_res.redchi > 20:
            return False, f"Poor fit quality: redchi={fit_res.redchi:.2f}"
    
    return True, "Valid fit"


def _linear_background_guess(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) of a linear fit to (x_vals, y_vals)."""
    if len(x_vals) < 2:
        return 0.0, float(np.nanmedian(y_vals) if len(y_vals) else 0.0)
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    return float(slope), float(intercept)


def _exp_background(x: np.ndarray, amplitude: float, decay: float) -> np.ndarray:
    """Exponential background model: amplitude * exp(decay * (x - x0))."""
    x0 = float(np.nanmin(x)) if np.size(x) else 0.0
    return amplitude * np.exp(decay * (x - x0))


def build_model_bundle(peak_spec: PeakSpec) -> FitBundle:
    """Build the lmfit composite model described by *peak_spec.fitter*.

    For built-in kinds the model is composed from lmfit primitives. For custom
    kinds, ``fitter.factory`` is called with ``peak_spec`` as a keyword argument
    and must return a :class:`FitBundle`.

    Returns:
        :class:`FitBundle` with the model, kind string, and component prefixes.

    Raises:
        ValueError: If ``fitter.kind`` is not recognised.
        TypeError: If a custom factory returns the wrong type.
    """
    fitter = peak_spec.fitter

    if fitter.factory is not None:
        factory = fitter.factory
        if isinstance(factory, str):
            module_name, sep, attr = factory.partition(":")
            if not sep:
                raise ValueError(f"Custom factory must be 'module:function', got {factory!r}")
            module = importlib.import_module(module_name)
            factory_fn = getattr(module, attr)
        else:
            factory_fn = factory
        bundle = factory_fn(peak_spec=peak_spec, **fitter.kwargs)
        if not isinstance(bundle, FitBundle):
            raise TypeError("Custom factory must return a FitBundle.")
        return bundle

    kind = fitter.kind
    if kind == "voigt_linear":
        return FitBundle(VoigtModel(prefix="peak_") + LinearModel(prefix="bkg_"), kind, ("peak_", "bkg_"))
    if kind == "voigt_constant":
        return FitBundle(VoigtModel(prefix="peak_") + ConstantModel(prefix="bkg_"), kind, ("peak_", "bkg_"))
    if kind == "voigt_exp":
        return FitBundle(VoigtModel(prefix="peak_") + Model(_exp_background, prefix="bkg_"), kind, ("peak_", "bkg_"))
    if kind == "gaussian_linear":
        return FitBundle(GaussianModel(prefix="peak_") + LinearModel(prefix="bkg_"), kind, ("peak_", "bkg_"))
    if kind == "pseudo_voigt_linear":
        return FitBundle(PseudoVoigtModel(prefix="peak_") + LinearModel(prefix="bkg_"), kind, ("peak_", "bkg_"))
    if kind == "double_voigt_linear":
        return FitBundle(VoigtModel(prefix="peak_") + VoigtModel(prefix="bkg_peak_") + LinearModel(prefix="bkg_lin_"), kind, ("peak_", "bkg_peak_", "bkg_lin_"))
    if kind == "voigt":
        return FitBundle(VoigtModel(prefix="peak_"), kind, ("peak_",))

    raise ValueError(f"Unsupported fitter kind: {kind}")


def initialize_params(
    bundle: FitBundle,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    peak_spec: PeakSpec,
    candidate_center: float,
    previous_params: dict[str, float | None] | None = None,
    use_adaptive: bool = False,
) -> Any:
    """
    Initialize fit parameters with optional adaptive bounds based on previous fit.
    
    Args:
        bundle: FitBundle with model
        x_vals: x-coordinates for fitting
        y_vals: y-values for fitting
        peak_spec: Peak specification
        candidate_center: Estimated peak center
        previous_params: Previous fit parameters (sigma, gamma, amplitude) for adaptive bounds
        use_adaptive: Whether to apply adaptive bounds (requires previous_params)
    
    Returns:
        lmfit Parameters object
    """
    params = bundle.model.make_params()
    finite_mask = np.isfinite(y_vals)
    x_vals = np.asarray(x_vals[finite_mask], dtype=float)
    y_vals = np.asarray(y_vals[finite_mask], dtype=float)

    if len(x_vals) < 3:
        return params

    slope, intercept = _linear_background_guess(x_vals, y_vals)
    peak_height = max(float(np.nanmax(y_vals) - np.nanmedian(y_vals)), 1e-6)
    # When fit_half_width_deg is set, use the actual fit window for sigma bounds
    # rather than the (usually larger) integration window_deg.  This allows the
    # optimizer to find sharper peaks that would otherwise hit the sigma floor
    # computed from the wider integration window.
    _fit_hw = peak_spec.fitter.fit_half_width_deg
    effective_window = min(peak_spec.window_deg, 2 * _fit_hw) if _fit_hw else peak_spec.window_deg
    sigma_floor = max(0.01, effective_window / 50)
    sigma_ceiling = max(effective_window, peak_spec.drift_tolerance_deg * 4)
    # Initialize gamma narrower (window/16 instead of window/8) to start closer to
    # a physically realistic peak width for perovskite GIWAXS peaks (~0.05-0.1 deg).
    gamma_init = max(effective_window / 16, sigma_floor)
    # VoigtModel amplitude is the profile *area* (integral), not the peak height.
    # For the Lorentzian limit: height = amplitude / (pi * gamma), so
    # amplitude = height * pi * gamma.  This initialization keeps the starting
    # model height close to the data peak and avoids the optimizer starting from
    # an amplitude value that is ~8x too large.
    amplitude_init = peak_height * np.pi * gamma_init
    
    # ADAPTIVE BOUNDS: Use previous fit parameters if available and enabled
    if use_adaptive and previous_params and "sigma" in previous_params and "gamma" in previous_params:
        prev_sigma_val = previous_params["sigma"]
        prev_gamma_val = previous_params["gamma"]
        prev_amplitude_val = previous_params.get("amplitude")
        
        # Only proceed if we have valid float values (not None)
        if prev_sigma_val is not None and prev_gamma_val is not None:
            prev_sigma = float(prev_sigma_val)
            prev_gamma = float(prev_gamma_val)
            prev_amplitude = float(prev_amplitude_val) if prev_amplitude_val is not None else amplitude_init
            
            # Sigma bounds: previous ± tolerance, but respect floor/ceiling
            sigma_tolerance = peak_spec.adaptive_sigma_tolerance
            sigma_min_adaptive = max(sigma_floor, prev_sigma * (1 - sigma_tolerance))
            sigma_max_adaptive = min(sigma_ceiling, prev_sigma * (1 + sigma_tolerance))
            
            # Gamma bounds: similar logic
            gamma_min_adaptive = max(sigma_floor, prev_gamma * (1 - sigma_tolerance))
            gamma_max_adaptive = min(sigma_ceiling, prev_gamma * (1 + sigma_tolerance))
            
            # Amplitude bounds: previous ± tolerance
            amp_tolerance = peak_spec.adaptive_amplitude_tolerance
            amp_min_adaptive = max(1e-6, prev_amplitude * (1 - amp_tolerance))
            amp_max_adaptive = prev_amplitude * (1 + amp_tolerance)
            
            params["peak_amplitude"].set(value=prev_amplitude, min=amp_min_adaptive, max=amp_max_adaptive)
            params["peak_sigma"].set(value=prev_sigma, min=sigma_min_adaptive, max=sigma_max_adaptive)
            params["peak_gamma"].set(value=prev_gamma, min=gamma_min_adaptive, max=gamma_max_adaptive)
        else:
            # Fallback to wide bounds if values are None
            params["peak_amplitude"].set(value=amplitude_init, min=1e-6)
            params["peak_sigma"].set(value=gamma_init, min=sigma_floor, max=sigma_ceiling)
            params["peak_gamma"].set(value=gamma_init, min=sigma_floor, max=sigma_ceiling)
    else:
        # Wide bounds (original behavior)
        params["peak_amplitude"].set(value=amplitude_init, min=1e-6)
        params["peak_sigma"].set(value=gamma_init, min=sigma_floor, max=sigma_ceiling)
        params["peak_gamma"].set(value=gamma_init, min=sigma_floor, max=sigma_ceiling)
    
    params["peak_center"].set(
        value=candidate_center,
        min=candidate_center - peak_spec.drift_tolerance_deg,
        max=candidate_center + peak_spec.drift_tolerance_deg,
    )

    if "peak_fraction" in params:
        params["peak_fraction"].set(value=0.5, min=0.0, max=1.0)
    if "bkg_slope" in params:
        params["bkg_slope"].set(value=slope)
    if "bkg_intercept" in params:
        params["bkg_intercept"].set(value=intercept)
    if "bkg_c" in params:
        params["bkg_c"].set(value=intercept)
    if "bkg_amplitude" in params:
        positive_y = np.clip(y_vals, 1e-6, None)
        try:
            exp_slope, exp_intercept = np.polyfit(x_vals, np.log(positive_y), 1)
            amp_guess = float(np.exp(exp_intercept))
            decay_guess = float(exp_slope)
        except Exception:
            amp_guess = float(max(np.nanmedian(y_vals), 1e-6))
            decay_guess = -0.5
        params["bkg_amplitude"].set(value=amp_guess, min=1e-6)
        if "bkg_decay" in params:
            params["bkg_decay"].set(value=min(decay_guess, -1e-6), min=-10.0, max=0.0)
    if "bkg_peak_amplitude" in params:
        background_center = peak_spec.fitter.kwargs.get("background_peak_center_deg", candidate_center - max(peak_spec.window_deg * 2.0, 1.0))
        background_sigma = float(peak_spec.fitter.kwargs.get("background_peak_sigma", max(peak_spec.window_deg, 0.35)))
        background_gamma = float(peak_spec.fitter.kwargs.get("background_peak_gamma", background_sigma))
        background_amp = float(peak_spec.fitter.kwargs.get("background_peak_amplitude", max(0.5 * peak_height, 1e-6)))
        params["bkg_peak_amplitude"].set(value=background_amp, min=1e-6)
        params["bkg_peak_center"].set(
            value=background_center,
            min=background_center - max(0.35, peak_spec.window_deg * 0.25),
            max=background_center + max(0.35, peak_spec.window_deg * 0.25),
        )
        params["bkg_peak_sigma"].set(value=background_sigma, min=1e-6, max=max(0.8, peak_spec.window_deg))
        params["bkg_peak_gamma"].set(value=background_gamma, min=1e-6, max=max(0.8, peak_spec.window_deg))
    if "bkg_lin_slope" in params:
        params["bkg_lin_slope"].set(value=slope)
    if "bkg_lin_intercept" in params:
        params["bkg_lin_intercept"].set(value=intercept)

    return params


def fit_frame(
    frame: xr.DataArray,
    peak_spec: PeakSpec,
    candidate_center: float,
    previous_params: dict[str, float | None] | None = None,
    use_adaptive: bool = False,
) -> dict[str, Any]:
    """
    Fit a peak in a single frame with optional adaptive parameter bounds.
    
    Args:
        frame: Data array for this frame
        peak_spec: Peak specification
        candidate_center: Estimated peak center
        previous_params: Previous fit parameters for adaptive bounds
        use_adaptive: Whether to use adaptive bounds
    
    Returns:
        Dictionary with fit results and integration bounds
    """
    fit_half_width = (
        peak_spec.fitter.fit_half_width_deg
        or peak_spec.peak_only_window_deg
        or max(peak_spec.window_deg / 2, peak_spec.drift_tolerance_deg * 2)
    )
    fit_data = local_window(frame, candidate_center, fit_half_width)
    if fit_data.sizes.get("twoTheta_deg", 0) < 4:
        raise RuntimeError("Fit window too small.")

    x_vals = fit_data.twoTheta_deg.values
    y_vals = fit_data.values
    bundle = build_model_bundle(peak_spec)
    params = initialize_params(
        bundle, x_vals, y_vals, peak_spec, candidate_center,
        previous_params=previous_params, use_adaptive=use_adaptive
    )
    fit_method = "least_squares" if bundle.fit_kind in {"voigt_exp", "double_voigt_linear"} else "leastsq"
    fit_res = bundle.model.fit(y_vals, params, x=x_vals, method=fit_method, calc_covar=False, max_nfev=10000)

    if not fit_res.success:
        raise RuntimeError(f"Fit failed: {fit_res.message}")

    best = fit_res.best_values
    center = float(best.get("peak_center", candidate_center))
    if abs(center - candidate_center) > peak_spec.drift_tolerance_deg:
        raise RuntimeError(
            f"Fitted center drifted too far from candidate: {center:.4f} vs {candidate_center:.4f}"
        )

    sigma = float(best.get("peak_sigma", max(peak_spec.window_deg / 8, 1e-3)))
    gamma = float(best.get("peak_gamma", max(peak_spec.window_deg / 8, 1e-3)))
    width = float(voigt_width_at_height(sigma, gamma, peak_spec.integration_height) * peak_spec.integration_multiplier)
    width = max(min(width, peak_spec.max_integration_span_deg), max(peak_spec.drift_tolerance_deg * 2, peak_spec.window_deg / 10))

    return {
        "fit_res": fit_res,
        "fit_kind": bundle.fit_kind,
        "candidate_center": candidate_center,
        "fit_center": center,
        "sigma": sigma,
        "gamma": gamma,
        "integration_left": center - width / 2,
        "integration_right": center + width / 2,
        "fit_half_width": fit_half_width,
        "fit_left": candidate_center - fit_half_width,
        "fit_right": candidate_center + fit_half_width,
    }


def summarize_fit_quality(fit_res: Any) -> dict[str, float]:
    """Compute fit quality metrics from an lmfit ModelResult.

    Returns a dict with keys: ``fit_chisqr``, ``fit_redchi``, ``fit_rmse``,
    ``fit_rel_rmse``, ``fit_r2``, ``fit_quality_score``.

    ``fit_quality_score`` = 1 / (1 + rel_rmse), ranging from 0 (bad) to 1 (perfect).
    """
    data = np.asarray(fit_res.data, dtype=float)
    residual = np.asarray(fit_res.residual, dtype=float)
    ss_res = float(np.nansum(residual**2))
    data_centered = data - float(np.nanmean(data)) if np.isfinite(np.nanmean(data)) else data
    ss_tot = float(np.nansum(data_centered**2))
    rmse = float(np.sqrt(np.nanmean(residual**2))) if residual.size else np.nan
    data_scale = float(max(np.nanmax(np.abs(data)), np.nanstd(data), 1e-6)) if data.size else 1e-6
    rel_rmse = float(rmse / data_scale) if np.isfinite(rmse) else np.nan
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan
    quality_score = float(1.0 / (1.0 + rel_rmse)) if np.isfinite(rel_rmse) else np.nan
    return {
        "fit_chisqr": float(getattr(fit_res, "chisqr", np.nan)),
        "fit_redchi": float(getattr(fit_res, "redchi", np.nan)),
        "fit_rmse": rmse,
        "fit_rel_rmse": rel_rmse,
        "fit_r2": r2,
        "fit_quality_score": quality_score,
    }


def integrate_interval(frame: xr.DataArray, left: float, right: float, num_points: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate *frame* onto a uniform grid over [left, right] for numerical integration."""
    grid = np.linspace(left, right, num_points)
    values = frame.interp(twoTheta_deg=grid).values
    return grid, values


def integrate_peak(frame: xr.DataArray, fit_info: dict[str, Any]) -> dict[str, Any]:
    """Compute raw area, background area, and net area for a fitted peak.

    Integration bounds are taken from *fit_info* (set by :func:`fit_frame`) and
    clamped to the data range. Background area is the trapezoid integral of all
    non-``peak_`` model components. Net area = raw area − background area.

    Args:
        frame: Baseline-corrected 1-D frame DataArray.
        fit_info: Output dict from :func:`fit_frame`.

    Returns:
        Dict with ``raw_area``, ``background_area``, ``net_area``,
        ``integration_left``, ``integration_right``.
    """
    left = float(max(fit_info["integration_left"], float(frame.twoTheta_deg.min())))
    right = float(min(fit_info["integration_right"], float(frame.twoTheta_deg.max())))
    if not math.isfinite(left) or not math.isfinite(right) or left >= right:
        raise RuntimeError("Invalid integration bounds.")

    grid, raw_values = integrate_interval(frame, left, right)
    raw_area = float(np.trapezoid(raw_values, grid))

    fit_res = fit_info["fit_res"]
    comps = fit_res.eval_components(x=grid)
    background_area = float(np.trapezoid(np.sum([values for name, values in comps.items() if not name.startswith("peak_")], axis=0), grid)) if any(not name.startswith("peak_") for name in comps) else 0.0
    net_area = raw_area - background_area

    return {
        "raw_area": raw_area,
        "background_area": background_area,
        "net_area": net_area,
        "integration_left": left,
        "integration_right": right,
    }


def frame_indices_for_peak(peak_spec: PeakSpec, size: int) -> list[int]:
    """Compute the ordered list of frame indices to process for *peak_spec*.

    Handles negative indices, ``frame_step``, and ``key_frames_mode``.
    In ``"restrict"`` mode only the configured key frames are processed.
    In ``"additive"`` mode (default) the full range is returned; key frames
    only control tracking resets, not which frames are visited.
    """
    start = peak_spec.start_idx if peak_spec.start_idx >= 0 else size + peak_spec.start_idx
    stop = peak_spec.stop_idx if peak_spec.stop_idx >= 0 else size + peak_spec.stop_idx
    if start < 0 or stop < 0 or start >= size or stop >= size:
        raise IndexError(f"Frame range ({peak_spec.start_idx}, {peak_spec.stop_idx}) is outside series length {size}.")
    step = peak_spec.frame_step if peak_spec.frame_step > 0 else 1
    if peak_spec.key_frames is not None and peak_spec.key_frames_mode == "restrict":
        lo, hi = sorted((start, stop))
        selected: list[int] = []
        for raw_idx in peak_spec.key_frames:
            idx = raw_idx if raw_idx >= 0 else size + raw_idx
            if 0 <= idx < size and lo <= idx <= hi:
                selected.append(idx)
        if not selected:
            raise IndexError(f"No key frames from {peak_spec.key_frames} fall within the series range {start}->{stop}.")
        return selected

    # additive/default behavior: return full range
    if stop >= start:
        return list(range(start, stop + 1, step))
    return list(range(start, stop - 1, -step))


def _save_fit_plot(run_dir: Path, frame: xr.DataArray, fit_info: dict[str, Any], peak_spec: PeakSpec, frame_index: int, time_value: float, margin_deg: float = 5.0) -> None:
    run_dir = Path(run_dir)
    plot_dir = run_dir / "fit_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fit_res = fit_info["fit_res"]
    # Limit plotting range to integration ± margin for focused inspection
    left = float(fit_info.get("integration_left", float(frame.twoTheta_deg.min())))
    right = float(fit_info.get("integration_right", float(frame.twoTheta_deg.max())))
    plot_left = left - margin_deg
    plot_right = right + margin_deg
    fit_left = float(fit_info.get("fit_left", left))
    fit_right = float(fit_info.get("fit_right", right))
    fit_grid = np.linspace(fit_left, fit_right, 512)
    comps = fit_res.eval_components(x=fit_grid)
    total = fit_res.eval(x=fit_grid)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    # plot only the data within the focused window
    try:
        data_slice = frame.sel(twoTheta_deg=slice(plot_left, plot_right))
        x_data = data_slice.twoTheta_deg.values
        y_data = data_slice.values
    except Exception:
        x_data = frame.twoTheta_deg.values
        y_data = frame.values
    ax.plot(x_data, y_data, label="Data", color="#1f77b4")
    ax.plot(fit_grid, total, "--", color="#d62728", label="Fit")
    if "peak_" in comps:
        ax.plot(fit_grid, comps["peak_"], ":", color="#ff7f0e", label="Peak")
        ax.fill_between(fit_grid, comps["peak_"], 0, where=(fit_grid >= fit_info["integration_left"]) & (fit_grid <= fit_info["integration_right"]), color="#ffdd99", alpha=0.6, label="Integrated Area")
    background_index = 0
    for name, values in comps.items():
        if name.startswith("peak_"):
            continue
        label = "Background" if background_index == 0 else f"Background {background_index + 1}"
        ax.plot(fit_grid, values, "-.", label=label)
        background_index += 1

    ax.axvline(fit_info["integration_left"], linestyle="--", color="0.5", linewidth=1)
    ax.axvline(fit_info["integration_right"], linestyle="--", color="0.5", linewidth=1)
    ax.axvline(fit_left, linestyle=":", color="0.4", linewidth=1)
    ax.axvline(fit_right, linestyle=":", color="0.4", linewidth=1)

    best = fit_res.best_values
    amp = float(best.get("peak_amplitude", np.nan))
    cen = float(best.get("peak_center", fit_info.get("candidate_center", np.nan)))
    sigma = float(best.get("peak_sigma", np.nan)) if "peak_sigma" in best else float(best.get("peak_sigma", np.nan))
    gamma = float(best.get("peak_gamma", np.nan)) if "peak_gamma" in best else float(best.get("peak_gamma", np.nan))

    bbox_text = f"Amp: {amp:.4f}\nCen: {cen:.4f}\nSigma: {sigma:.3f}\nGamma: {gamma:.3f}"
    ax.text(0.02, 0.95, bbox_text, transform=ax.transAxes, fontsize=8, va="top", bbox=dict(facecolor="wheat", alpha=0.8, boxstyle="round"))

    ax.set_xlabel("twoTheta_deg")
    ax.set_ylabel("sample_norm_int")
    ax.set_title(f"{peak_spec.name} Tracked at t={time_value:.1f}s (idx {frame_index})")
    ax.legend(fontsize="small")
    fig.tight_layout()
    # set x-limits to focused window
    try:
        ax.set_xlim(plot_left, plot_right)
    except Exception:
        pass
    fname = f"fit_{peak_spec.name.replace(' ', '_')}_idx{frame_index}.png"
    fig.savefig(plot_dir / fname)
    plt.close(fig)


def _save_failed_fit_plot(run_dir: Path, frame: xr.DataArray, candidate_center: float | None, peak_spec: PeakSpec, frame_index: int, time_value: float, margin_deg: float = 5.0) -> None:
    run_dir = Path(run_dir)
    plot_dir = run_dir / "fit_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # show the data and mark candidate and center
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(frame.twoTheta_deg.values, frame.values, label="Data", color="#1f77b4")
    if candidate_center is not None:
        ax.axvline(candidate_center, color="#d62728", linestyle="--", label="Candidate")
    ax.set_xlabel("twoTheta_deg")
    ax.set_ylabel("sample_norm_int")
    ax.set_title(f"FAILED FIT {peak_spec.name} at t={time_value:.1f}s (idx {frame_index})")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fname = f"failed_fit_{peak_spec.name.replace(' ', '_')}_idx{frame_index}.png"
    fig.savefig(plot_dir / fname)
    plt.close(fig)


def track_peak_series(series: xr.DataArray, peak_spec: PeakSpec, run_dir: Path, global_shift_deg: float = 0.0, keyframe_plot_margin_deg: float = 5.0) -> list[dict[str, Any]]:
    """Track and fit one peak across all frames in *series*.

    Implements the full frame-by-frame loop:
    baseline correction → SNR check → candidate search → fitting
    (with optional adaptive bounds) → post-fit validation → integration
    → row assembly.

    Key-frame resets (user-configured and auto-detected) cause the search
    anchor to revert to ``peak_spec.center_deg`` and reset ``TrackingState``.
    Fit PNGs are saved to ``run_dir/fit_plots/`` for every key frame.

    Args:
        series: Full 2-D (time × twoTheta_deg) DataArray for one run.
        peak_spec: Specification for this peak.
        run_dir: Output directory for log and fit plots.
        global_shift_deg: 2θ offset already applied to *series* (recorded in
            CSV for traceability).
        keyframe_plot_margin_deg: Plot window half-width around the integration
            region in key-frame PNGs.

    Returns:
        List of row dicts, one per processed frame. Columns match the long CSV
        schema described in PEAK_TRACKING_GUIDE.md.
    """
    times = series.time.values
    frames = frame_indices_for_peak(peak_spec, len(times))
    state = TrackingState(
        last_center=peak_spec.center_deg,
        tracking_active=False,
        lost_count=0,
        consecutive_good_fits=0,
        peak_present=True,
    )
    rows: list[dict[str, Any]] = []
    auto_key_frames: list[int] = []
    first_detected = False
    consecutive_tracked = 0
    prev_was_lost = False

    for frame_index in tqdm(frames, desc=f"{peak_spec.name}", unit="frame", leave=False):
        time_value = float(times[frame_index])
        frame = series.isel(time=frame_index).sortby("twoTheta_deg")
        frame_corr, baseline_da = baseline_correct_frame(frame, peak_spec.baseline_method, **peak_spec.baseline_kwargs)

        # Check if this frame is a user-configured key frame (compute early so we can force-reset)
        is_config_key = peak_spec.key_frames is not None and frame_index in peak_spec.key_frames if peak_spec.key_frames is not None else False
        if is_config_key:
            # manual force reset: drop current tracking and anchor search to configured center
            state.tracking_active = False
            state.last_center = peak_spec.center_deg
            state.last_sigma = None
            state.last_gamma = None
            state.last_amplitude = None
            state.lost_count = 0
            state.consecutive_good_fits = 0
            prev_was_lost = True

        search_anchor = state.last_center if state.last_center is not None else peak_spec.center_deg
        search_half_width = max(peak_spec.window_deg / 2, peak_spec.reacquire_window_deg or peak_spec.drift_tolerance_deg * 3)
        search_window = local_window(frame_corr, search_anchor, search_half_width)
        
        # STAGE 1: PRE-FIT VALIDATION - Check for peak presence
        has_peak, presence_reason, peak_snr = check_peak_presence(search_window, min_snr=peak_spec.min_snr)
        
        candidate = choose_candidate(search_window, search_anchor, min_prominence=peak_spec.min_prominence)

        fit_success = False
        fit_error: str | None = None
        fit_info: dict[str, Any] | None = None
        fit_status = "search" if not state.tracking_active else "track"
        is_auto_key = False
        validation_reason: str | None = None
        used_adaptive_bounds = False

        # If no peak detected by SNR check, mark as peak_absent
        if not has_peak:
            fit_error = f"Peak absent: {presence_reason}"
            fit_status = "peak_absent"
            state.tracking_active = False
            state.peak_present = False
            state.lost_count += 1
            prev_was_lost = True
            # No need to save plot or raise for SNR failures - the SNR value in CSV is sufficient diagnostic
            # Plot generation is expensive and there's nothing useful to visualize if SNR check fails
            if is_config_key:
                logger.warning(
                    "Peak absent (SNR check failed) at configured key frame for peak %s at frame %s: %s (SNR=%.2f)",
                    peak_spec.name, frame_index, presence_reason, peak_snr
                )
        
        elif candidate is None and state.last_center is not None:
            # Try expanded search window
            search_window = local_window(frame_corr, state.last_center, max(search_half_width, peak_spec.drift_tolerance_deg * 4))
            candidate = choose_candidate(search_window, state.last_center, min_prominence=peak_spec.min_prominence)

        if candidate is not None and has_peak:
            # Determine if we should use adaptive bounds
            use_adaptive = (
                peak_spec.use_adaptive_bounds
                and state.consecutive_good_fits >= peak_spec.adaptive_min_consecutive
                and state.last_sigma is not None
                and state.last_gamma is not None
                and state.last_amplitude is not None
            )
            
            previous_params = None
            if use_adaptive:
                previous_params = {
                    "sigma": state.last_sigma,
                    "gamma": state.last_gamma,
                    "amplitude": state.last_amplitude,
                }
                used_adaptive_bounds = True
            
            try:
                # Attempt fitting with adaptive bounds if enabled
                fit_info = fit_frame(
                    frame_corr, peak_spec, candidate["center"],
                    previous_params=previous_params,
                    use_adaptive=use_adaptive
                )
                
                # STAGE 2: POST-FIT VALIDATION - Check fit quality
                if peak_spec.validate_fit_quality:
                    is_valid, validation_reason = validate_fit_quality(
                        fit_info["fit_res"], peak_spec, search_window
                    )
                    
                    if not is_valid:
                        # Validation failed - try fallback if enabled
                        if use_adaptive and peak_spec.fallback_on_validation_failure:
                            logger.debug(
                                "Fit validation failed with adaptive bounds for %s idx=%s (%s), retrying with wide bounds",
                                peak_spec.name, frame_index, validation_reason
                            )
                            # Retry with wide bounds
                            fit_info = fit_frame(
                                frame_corr, peak_spec, candidate["center"],
                                previous_params=None,
                                use_adaptive=False
                            )
                            used_adaptive_bounds = False
                            # Re-validate
                            is_valid, validation_reason = validate_fit_quality(
                                fit_info["fit_res"], peak_spec, search_window
                            )
                        
                        if not is_valid:
                            # Still invalid after fallback
                            fit_error = f"Fit validation failed: {validation_reason}"
                            fit_status = "fit_invalid"
                            state.tracking_active = False
                            state.consecutive_good_fits = 0
                            state.lost_count += 1
                            prev_was_lost = True
                            raise RuntimeError(fit_error)
                
                # Fit succeeded and passed validation
                fit_success = True
                fit_status = "track" if state.tracking_active else "search"
                
                # Update tracking state with fitted parameters
                state.last_center = float(fit_info["fit_center"])
                best = fit_info["fit_res"].best_values
                state.last_sigma = float(best.get("peak_sigma", state.last_sigma or 0.05))
                state.last_gamma = float(best.get("peak_gamma", state.last_gamma or 0.05))
                state.last_amplitude = float(best.get("peak_amplitude", state.last_amplitude or 1.0))
                state.tracking_active = True
                state.peak_present = True
                state.lost_count = 0
                state.consecutive_good_fits += 1

                # Auto key-frame logic
                is_auto_key = False
                if not first_detected:
                    is_auto_key = True
                    first_detected = True
                    consecutive_tracked = 1
                else:
                    if prev_was_lost:
                        # reacquired after a loss
                        is_auto_key = True
                        prev_was_lost = False
                        consecutive_tracked = 1
                    else:
                        consecutive_tracked += 1
                        if peak_spec.auto_keyframe_every and consecutive_tracked >= peak_spec.auto_keyframe_every:
                            is_auto_key = True
                            consecutive_tracked = 0

                if is_auto_key:
                    auto_key_frames.append(frame_index)

            except Exception as exc:
                fit_error = str(exc)
                state.tracking_active = False
                state.consecutive_good_fits = 0
                state.lost_count += 1
                prev_was_lost = True
                # If this was a user-configured key frame, save diagnostic and raise — user provided explicit frame
                if is_config_key:
                    try:
                        _save_failed_fit_plot(run_dir, frame_corr, candidate.get("center") if candidate else None, peak_spec, frame_index, time_value, margin_deg=keyframe_plot_margin_deg)
                    except Exception:
                        logger.exception("Failed to save failed-fit plot for configured key frame %s idx=%s", peak_spec.name, frame_index)
                    raise ValueError(f"Configured key frame failed to fit for peak {peak_spec.name} at frame {frame_index}: {exc}")
        elif candidate is None:
            fit_error = "No candidate peak found"
            fit_status = "no_candidate"
            state.tracking_active = False
            state.consecutive_good_fits = 0
            state.lost_count += 1
            prev_was_lost = True
            # If no candidate on a user-configured key frame, save diagnostic and raise
            if is_config_key:
                try:
                    _save_failed_fit_plot(run_dir, frame_corr, None, peak_spec, frame_index, time_value, margin_deg=keyframe_plot_margin_deg)
                except Exception:
                    logger.exception("Failed to save failed-fit plot for configured key frame %s idx=%s", peak_spec.name, frame_index)
                raise ValueError(f"No candidate peak found at configured key frame for peak {peak_spec.name} at frame {frame_index}")

        row: dict[str, Any] = {
            "time": time_value,
            "frame_index": frame_index,
            "is_config_key_frame": is_config_key,
            "is_auto_key_frame": is_auto_key,
            "is_key_frame": bool(is_config_key or is_auto_key),
            "peak_name": peak_spec.name,
            "expected_center_deg": peak_spec.center_deg,
            "candidate_center_deg": candidate["center"] if candidate else np.nan,
            "fit_center_deg": np.nan,
            "drift_tolerance_deg": peak_spec.drift_tolerance_deg,
            "global_shift_deg": global_shift_deg,
            "fit_status": "lost" if fit_error else fit_status,
            "fit_success": False,
            "fit_model": peak_spec.fitter.kind,
            "fit_message": None,
            "raw_area": np.nan,
            "background_area": np.nan,
            "net_area": np.nan,
            "integration_left_deg": np.nan,
            "integration_right_deg": np.nan,
            "integration_width_deg": np.nan,
            "fit_amplitude": np.nan,
            "fit_sigma": np.nan,
            "fit_gamma": np.nan,
            "fit_slope": np.nan,
            "fit_offset": np.nan,
            "fit_chisqr": np.nan,
            "fit_redchi": np.nan,
            "fit_rmse": np.nan,
            "fit_rel_rmse": np.nan,
            "fit_r2": np.nan,
            "fit_quality_score": np.nan,
            "raw_y_max": float(np.nanmax(frame_corr.values)),
            "baseline_method": peak_spec.baseline_method or "none",
            "baseline_applied": baseline_da is not None,
            "fit_error": fit_error,
            "reacquire_anchor_deg": search_anchor,
            # New validation and adaptive bounds fields
            "peak_snr": peak_snr if has_peak else 0.0,
            "peak_presence_check": "passed" if has_peak else "failed",
            "validation_reason": validation_reason,
            "used_adaptive_bounds": used_adaptive_bounds,
            "consecutive_good_fits": state.consecutive_good_fits,
        }

        if fit_success and fit_info is not None:
            try:
                integration = integrate_peak(frame_corr, fit_info)
                best = fit_info["fit_res"].best_values
                quality = summarize_fit_quality(fit_info["fit_res"])
                row.update(
                    {
                        "fit_success": True,
                        "fit_status": fit_status,
                        "fit_message": fit_info["fit_res"].message,
                        "fit_center_deg": fit_info["fit_center"],
                        "raw_area": integration["raw_area"],
                        "background_area": integration["background_area"],
                        "net_area": integration["net_area"],
                        "integration_left_deg": integration["integration_left"],
                        "integration_right_deg": integration["integration_right"],
                        "integration_width_deg": integration["integration_right"] - integration["integration_left"],
                        "fit_amplitude": float(best.get("peak_amplitude", np.nan)),
                        "fit_sigma": float(best.get("peak_sigma", np.nan)),
                        "fit_gamma": float(best.get("peak_gamma", np.nan)),
                        "fit_slope": float(best.get("bkg_slope", 0.0)),
                        "fit_offset": float(best.get("bkg_intercept", best.get("bkg_c", 0.0))),
                        **quality,
                    }
                )
                # Save a per-frame fit plot for key frames (human inspection)
                try:
                    if row.get("is_key_frame", False):
                        _save_fit_plot(run_dir, frame_corr, fit_info, peak_spec, frame_index, time_value, margin_deg=keyframe_plot_margin_deg)
                except Exception:
                    logger.exception("Failed to save fit plot for %s idx=%s", peak_spec.name, frame_index)
            except Exception as exc:
                row["fit_error"] = str(exc)
                row["fit_success"] = False
                row["fit_status"] = "lost"
                state.tracking_active = False
                state.lost_count += 1

        rows.append(row)

    # Save auto-detected key frames summary for this peak
    try:
        summary = {
            "peak_name": peak_spec.name,
            "configured_key_frames": list(peak_spec.key_frames) if peak_spec.key_frames is not None else [],
            "auto_key_frames": auto_key_frames,
        }
        safe_name = peak_spec.name.replace(" ", "_").replace("/", "_")
        (Path(run_dir) / f"key_frames_{safe_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to write key frames summary for %s", peak_spec.name)

    return rows


def postprocess_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``normalized_raw_area`` and ``normalized_net_area`` columns to *df*.

    Each peak is normalized independently to its own maximum net/raw area
    across all frames. Rows with NaN or infinite areas are excluded from the
    normalization denominator.
    """
    df = df.copy()
    df["normalized_raw_area"] = np.nan
    df["normalized_net_area"] = np.nan
    for peak_name, group in df.groupby("peak_name", dropna=False):
        raw = group["raw_area"].replace([np.inf, -np.inf], np.nan).dropna()
        net = group["net_area"].replace([np.inf, -np.inf], np.nan).dropna()
        raw_denom = float(raw.max()) if not raw.empty else np.nan
        net_denom = float(net.max()) if not net.empty else np.nan
        if math.isfinite(raw_denom) and raw_denom not in (0.0, -0.0):
            df.loc[group.index, "normalized_raw_area"] = group["raw_area"] / raw_denom
        if math.isfinite(net_denom) and net_denom not in (0.0, -0.0):
            df.loc[group.index, "normalized_net_area"] = group["net_area"] / net_denom
    return df


def save_outputs(run_dir: Path, df_long: pd.DataFrame, config: RunConfig) -> None:
    """Write CSVs and run config JSON to *run_dir*.

    Writes:
    - ``tracking_results_long.csv``: all columns, all frames.
    - ``tracking_results_compact.csv``: subset of most-useful columns.
    - ``run_config.json``: serialised :class:`RunConfig` for reproducibility.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(run_dir / "tracking_results_long.csv", index=False)
    compact_cols = [
        "time",
        "frame_index",
        "is_key_frame",
        "is_config_key_frame",
        "is_auto_key_frame",
        "peak_name",
        "raw_area",
        "normalized_raw_area",
        "net_area",
        "normalized_net_area",
        "fit_r2",
        "fit_redchi",
        "fit_rel_rmse",
        "fit_quality_score",
        "background_area",
        "fit_center_deg",
        "integration_left_deg",
        "integration_right_deg",
        "fit_status",
        "fit_error",
    ]
    df_long.loc[:, [col for col in compact_cols if col in df_long.columns]].to_csv(run_dir / "tracking_results_compact.csv", index=False)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "name": config.name,
                "da_file": str(config.da_file),
                "global_shift_deg": config.global_shift_deg,
                "peaks": [asdict(p) for p in config.peaks],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def plot_results(run_dir: Path, df: pd.DataFrame) -> None:
    """Save summary plots to *run_dir*.

    Generates:
    - ``summary_peak_traces.png``: net area and normalised net area per peak
      on separate subplots.
    - ``summary_normalized_overlay.png``: all peak normalised net areas
      overlaid on one plot.
    """
    peaks = list(pd.unique(df["peak_name"]))
    if not peaks:
        return

    fig, axes = plt.subplots(len(peaks), 1, figsize=(10, max(3, 3.2 * len(peaks))), sharex=True, dpi=180)
    if len(peaks) == 1:
        axes = [axes]

    for ax, peak_name in zip(axes, peaks, strict=False):
        subset = df[df["peak_name"] == peak_name].sort_values("time")
        ax.plot(subset["time"], subset["net_area"], marker="o", markersize=3, linewidth=1.0, label="Net area")
        ax.plot(subset["time"], subset["normalized_net_area"], marker=".", markersize=2, linewidth=1.0, label="Normalized net area")
        ax.set_title(str(peak_name))
        ax.set_ylabel("Area")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize="small")

    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(run_dir / "summary_peak_traces.png", dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=180)
    for peak_name in peaks:
        subset = df[df["peak_name"] == peak_name].sort_values("time")
        ax.plot(subset["time"], subset["normalized_net_area"], label=str(peak_name), linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized net area")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(run_dir / "summary_normalized_overlay.png", dpi=250)
    plt.close(fig)


def run_peak_tracking(config: RunConfig) -> tuple[pd.DataFrame, Path]:
    """Execute a complete tracking run for all peaks defined in *config*.

    Creates a timestamped output directory, loads the time series, tracks each
    peak, normalises areas, saves outputs, and generates summary plots.

    Args:
        config: :class:`RunConfig` describing the data file and all peaks.

    Returns:
        Tuple of (results DataFrame, run output directory path).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_root / f"{config.name}_{timestamp}"
    configure_logging(run_dir)

    logger.info("Loading data from %s", config.da_file)
    series = load_time_series(config.da_file, wavelength_a=config.wavelength_a)
    series = apply_global_shift(series, config.global_shift_deg)

    rows: list[dict[str, Any]] = []
    for peak_spec in tqdm(config.peaks, desc="Peaks", unit="peak"):
        logger.info(
            "Tracking peak %s frames %s -> %s with drift tolerance %.3f deg",
            peak_spec.name,
            peak_spec.start_idx,
            peak_spec.stop_idx,
            peak_spec.drift_tolerance_deg,
        )
        peak_rows = track_peak_series(series, peak_spec, run_dir, global_shift_deg=config.global_shift_deg, keyframe_plot_margin_deg=config.keyframe_plot_margin_deg)
        for row in peak_rows:
            row["run_name"] = config.name
            row["global_shift_applied"] = config.global_shift_deg != 0.0
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = postprocess_normalization(df)
        save_outputs(run_dir, df, config)
        plot_results(run_dir, df)
        # Aggregate per-peak key frame summaries into a run-level CSV
        try:
            summaries = []
            for pjson in Path(run_dir).glob("key_frames_*.json"):
                try:
                    obj = json.loads(pjson.read_text(encoding="utf-8"))
                    summaries.append(
                        {
                            "peak_name": obj.get("peak_name"),
                            "configured_key_frames": ";".join(str(x) for x in obj.get("configured_key_frames", [])),
                            "auto_key_frames": ";".join(str(x) for x in obj.get("auto_key_frames", [])),
                        }
                    )
                except Exception:
                    logger.exception("Failed to read key frames summary %s", pjson)
            if summaries:
                kf_df = pd.DataFrame(summaries)
                kf_df.to_csv(Path(run_dir) / "key_frames_used.csv", index=False)
        except Exception:
            logger.exception("Failed to write run-level key_frames_used.csv")
    else:
        (run_dir / "empty_run.txt").write_text("No peak results were collected.", encoding="utf-8")

    logger.info("Finished run in %s", run_dir)
    return df, run_dir


def build_run_config_from_name(name: str, run_configs: dict[str, Any], default_output_root: Path, output_root: Path | None = None) -> RunConfig:
    """
    Build a RunConfig from a preset name in the config dictionary.
    
    Args:
        name: Name of the run preset
        run_configs: RUN_CONFIGS dictionary from the config file
        default_output_root: DEFAULT_OUTPUT_ROOT from the config file
        output_root: Optional override for output root
    
    Returns:
        RunConfig object
    """
    if name not in run_configs:
        raise KeyError(f"Unknown run preset: {name}")

    raw = run_configs[name]
    config = RunConfig(
        name=name,
        da_file=Path(raw["da_file"]),
        peaks=normalize_peak_entries(raw["peaks"]),
        global_shift_deg=float(raw.get("global_shift_deg", 0.0)),
        output_root=Path(output_root) if output_root is not None else default_output_root,
        baseline_method=raw.get("baseline_method"),
        baseline_kwargs=dict(raw.get("baseline_kwargs", {})),
        wavelength_a=float(raw.get("wavelength_a", DEFAULT_WAVELENGTH_A)),
        keyframe_plot_margin_deg=float(raw.get("keyframe_plot_margin_deg", 5.0)),
    )
    return config


def list_run_presets(run_configs: dict[str, Any]) -> None:
    """List all available run presets from the config dictionary."""
    for key, value in run_configs.items():
        print(f"{key}: {value['da_file']}")


def _run_one(name: str, run_configs: dict[str, Any], default_output_root: Path, output_root: Path, retries: int = 3) -> tuple[str, int, Path]:
    """Run a single preset and return (name, row_count, run_dir). Retries on failure."""
    last_exc: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            config = build_run_config_from_name(name, run_configs, default_output_root, output_root=output_root)
            df, run_dir = run_peak_tracking(config)
            return name, len(df), run_dir
        except Exception as exc:
            last_exc = exc
            logging.warning("Run %r failed on attempt %d/%d: %s", name, attempt, retries, exc)
    raise RuntimeError(f"Run {name!r} failed after {retries} attempts") from last_exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Peak tracking and integration workflow")
    parser.add_argument(
        "--config", 
        type=Path, 
        default=Path(__file__).parent / "peak_tracking_config.py",
        help="Path to peak tracking config file (default: peak_tracking_config.py in same directory)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", help="Run a single preset (name from config)")
    group.add_argument("--run-all", action="store_true", help="Run all presets in config in parallel")
    parser.add_argument("--list-runs", action="store_true", help="List available run presets")
    parser.add_argument("--output-root", type=Path, help="Override output root")
    parser.add_argument(
        "--jobs", type=int, default=-1,
        help="Number of parallel jobs for --run-all (default: -1 = all CPU cores)",
    )
    args = parser.parse_args()

    # Load config file
    try:
        default_output_root, run_configs = load_config_module(args.config)
    except Exception as exc:
        print(f"Error loading config file {args.config}: {exc}", file=sys.stderr)
        return 1

    output_root = args.output_root if args.output_root else default_output_root

    if args.list_runs:
        list_run_presets(run_configs)
        return 0

    if args.run_all:
        names = sorted(run_configs.keys())
        print(f"Running {len(names)} presets with {args.jobs} workers...")
        results = joblib.Parallel(n_jobs=args.jobs, backend="loky", verbose=10)(
            joblib.delayed(_run_one)(name, run_configs, default_output_root, output_root) for name in names
        )
        for name, nrows, run_dir in results:
            print(f"  {name}: {nrows} rows -> {run_dir}")
        return 0

    if not args.run:
        parser.error("--run or --run-all is required unless --list-runs is used")

    config = build_run_config_from_name(args.run, run_configs, default_output_root, output_root=output_root)
    df, run_dir = run_peak_tracking(config)
    print(f"Saved {len(df)} rows to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
