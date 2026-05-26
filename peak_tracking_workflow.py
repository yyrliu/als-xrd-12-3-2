from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
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

from peak_tracking_config import DEFAULT_OUTPUT_ROOT, RUN_CONFIGS
from vogit_width import voigt_width_at_height


DEFAULT_WAVELENGTH_A = 1.5418
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FitterSpec:
    kind: str = "voigt_linear"
    factory: str | Callable[..., Any] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    fit_half_width_deg: float | None = None


@dataclass(frozen=True)
class PeakSpec:
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


@dataclass(frozen=True)
class RunConfig:
    name: str
    da_file: Path
    peaks: list[PeakSpec]
    global_shift_deg: float = 0.0
    output_root: Path = DEFAULT_OUTPUT_ROOT
    baseline_method: str | None = None
    baseline_kwargs: dict[str, Any] = field(default_factory=dict)
    wavelength_a: float = DEFAULT_WAVELENGTH_A
    keyframe_plot_margin_deg: float = 5.0


@dataclass
class FitBundle:
    model: Model
    fit_kind: str
    component_names: tuple[str, ...] = ()


@dataclass
class TrackingState:
    last_center: float | None = None
    tracking_active: bool = False
    lost_count: int = 0


def configure_logging(run_dir: Path) -> None:
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
    if isinstance(data, xr.DataArray):
        return data

    for preferred_name in ("sample_norm_int", "intensity"):
        if preferred_name in data.data_vars:
            return data[preferred_name]

    if len(data.data_vars) == 1:
        return next(iter(data.data_vars.values()))

    raise ValueError("Could not identify an intensity data variable in the dataset.")


def load_time_series(da_file: Path, wavelength_a: float = DEFAULT_WAVELENGTH_A) -> xr.DataArray:
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
    if shift_deg == 0:
        return data
    return data.assign_coords(twoTheta_deg=data.twoTheta_deg + shift_deg)


def baseline_correct_frame(frame: xr.DataArray, method: str | None, **kwargs: Any) -> tuple[xr.DataArray, xr.DataArray | None]:
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
                )
            )
            continue

        raise TypeError(f"Unsupported peak entry: {type(entry)!r}")

    return normalized


def normalize_run_config(name: str, config: dict[str, Any]) -> RunConfig:
    return RunConfig(
        name=name,
        da_file=Path(config["da_file"]),
        peaks=normalize_peak_entries(config["peaks"]),
        global_shift_deg=float(config.get("global_shift_deg", 0.0)),
        output_root=Path(config.get("output_root", DEFAULT_OUTPUT_ROOT)),
        baseline_method=config.get("baseline_method"),
        baseline_kwargs=dict(config.get("baseline_kwargs", {})),
        wavelength_a=float(config.get("wavelength_a", DEFAULT_WAVELENGTH_A)),
        keyframe_plot_margin_deg=float(config.get("keyframe_plot_margin_deg", 5.0)),
    )


def inclusive_frame_indices(start_idx: int, stop_idx: int, size: int) -> list[int]:
    start = start_idx if start_idx >= 0 else size + start_idx
    stop = stop_idx if stop_idx >= 0 else size + stop_idx
    if start < 0 or start >= size or stop < 0 or stop >= size:
        raise IndexError(f"Frame range ({start_idx}, {stop_idx}) is outside series length {size}.")
    step = 1 if stop >= start else -1
    return list(range(start, stop + step, step))


def local_window(data: xr.DataArray, center: float, half_width: float) -> xr.DataArray:
    return data.sel(twoTheta_deg=slice(center - half_width, center + half_width))


def choose_candidate(
    window_data: xr.DataArray,
    center_guess: float,
    min_prominence: float = 0.0,
) -> dict[str, float] | None:
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


def _linear_background_guess(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float, float]:
    if len(x_vals) < 2:
        return 0.0, float(np.nanmedian(y_vals) if len(y_vals) else 0.0)
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    return float(slope), float(intercept)


def _exp_background(x: np.ndarray, amplitude: float, decay: float) -> np.ndarray:
    x0 = float(np.nanmin(x)) if np.size(x) else 0.0
    return amplitude * np.exp(decay * (x - x0))


def build_model_bundle(peak_spec: PeakSpec) -> FitBundle:
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
) -> Any:
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
    params["peak_amplitude"].set(value=amplitude_init, min=1e-6)
    params["peak_center"].set(
        value=candidate_center,
        min=candidate_center - peak_spec.drift_tolerance_deg,
        max=candidate_center + peak_spec.drift_tolerance_deg,
    )
    params["peak_sigma"].set(
        value=gamma_init,
        min=sigma_floor,
        max=sigma_ceiling,
    )
    params["peak_gamma"].set(
        value=gamma_init,
        min=sigma_floor,
        max=sigma_ceiling,
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


def fit_frame(frame: xr.DataArray, peak_spec: PeakSpec, candidate_center: float) -> dict[str, Any]:
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
    params = initialize_params(bundle, x_vals, y_vals, peak_spec, candidate_center)
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
    grid = np.linspace(left, right, num_points)
    values = frame.interp(twoTheta_deg=grid).values
    return grid, values


def integrate_peak(frame: xr.DataArray, fit_info: dict[str, Any]) -> dict[str, Any]:
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
    times = series.time.values
    frames = frame_indices_for_peak(peak_spec, len(times))
    state = TrackingState(last_center=peak_spec.center_deg, tracking_active=False, lost_count=0)
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
            state.lost_count = 0
            prev_was_lost = True

        search_anchor = state.last_center if state.last_center is not None else peak_spec.center_deg
        search_half_width = max(peak_spec.window_deg / 2, peak_spec.reacquire_window_deg or peak_spec.drift_tolerance_deg * 3)
        search_window = local_window(frame_corr, search_anchor, search_half_width)
        candidate = choose_candidate(search_window, search_anchor, min_prominence=peak_spec.min_prominence)

        fit_success = False
        fit_error: str | None = None
        fit_info: dict[str, Any] | None = None
        fit_status = "search" if not state.tracking_active else "track"
        is_auto_key = False

        if candidate is None and state.last_center is not None:
            search_window = local_window(frame_corr, state.last_center, max(search_half_width, peak_spec.drift_tolerance_deg * 4))
            candidate = choose_candidate(search_window, state.last_center, min_prominence=peak_spec.min_prominence)

        if candidate is not None:
            try:
                fit_info = fit_frame(frame_corr, peak_spec, candidate["center"])
                fit_success = True
                fit_status = "track" if state.tracking_active else "search"
                state.last_center = float(fit_info["fit_center"])
                state.tracking_active = True
                state.lost_count = 0

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
                state.lost_count += 1
                prev_was_lost = True
                # If this was a user-configured key frame, save diagnostic and raise — user provided explicit frame
                if is_config_key:
                    try:
                        _save_failed_fit_plot(run_dir, frame_corr, candidate.get("center") if candidate else None, peak_spec, frame_index, time_value, margin_deg=keyframe_plot_margin_deg)
                    except Exception:
                        logger.exception("Failed to save failed-fit plot for configured key frame %s idx=%s", peak_spec.name, frame_index)
                    raise ValueError(f"Configured key frame failed to fit for peak {peak_spec.name} at frame {frame_index}: {exc}")
        else:
            fit_error = "No candidate peak found"
            state.tracking_active = False
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


def build_run_config_from_name(name: str, output_root: Path | None = None) -> RunConfig:
    if name not in RUN_CONFIGS:
        raise KeyError(f"Unknown run preset: {name}")

    raw = RUN_CONFIGS[name]
    config = RunConfig(
        name=name,
        da_file=Path(raw["da_file"]),
        peaks=normalize_peak_entries(raw["peaks"]),
        global_shift_deg=float(raw.get("global_shift_deg", 0.0)),
        output_root=Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT,
        baseline_method=raw.get("baseline_method"),
        baseline_kwargs=dict(raw.get("baseline_kwargs", {})),
        wavelength_a=float(raw.get("wavelength_a", DEFAULT_WAVELENGTH_A)),
        keyframe_plot_margin_deg=float(raw.get("keyframe_plot_margin_deg", 5.0)),
    )
    return config


def list_run_presets() -> None:
    for key, value in RUN_CONFIGS.items():
        print(f"{key}: {value['da_file']}")


def _run_one(name: str, output_root: Path, retries: int = 3) -> tuple[str, int, Path]:
    """Run a single preset and return (name, row_count, run_dir). Retries on failure."""
    last_exc: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            config = build_run_config_from_name(name, output_root=output_root)
            df, run_dir = run_peak_tracking(config)
            return name, len(df), run_dir
        except Exception as exc:
            last_exc = exc
            logging.warning("Run %r failed on attempt %d/%d: %s", name, attempt, retries, exc)
    raise RuntimeError(f"Run {name!r} failed after {retries} attempts") from last_exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Peak tracking and integration workflow")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", choices=sorted(RUN_CONFIGS.keys()), help="Run a single preset")
    group.add_argument("--run-all", action="store_true", help="Run all presets in RUN_CONFIGS in parallel")
    parser.add_argument("--list-runs", action="store_true", help="List available run presets")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Override output root")
    parser.add_argument(
        "--jobs", type=int, default=-1,
        help="Number of parallel jobs for --run-all (default: -1 = all CPU cores)",
    )
    args = parser.parse_args()

    if args.list_runs:
        list_run_presets()
        return 0

    if args.run_all:
        names = sorted(RUN_CONFIGS.keys())
        print(f"Running {len(names)} presets with {args.jobs} workers...")
        results = joblib.Parallel(n_jobs=args.jobs, backend="loky", verbose=10)(
            joblib.delayed(_run_one)(name, args.output_root) for name in names
        )
        for name, nrows, run_dir in results:
            print(f"  {name}: {nrows} rows -> {run_dir}")
        return 0

    if not args.run:
        parser.error("--run or --run-all is required unless --list-runs is used")

    config = build_run_config_from_name(args.run, output_root=args.output_root)
    df, run_dir = run_peak_tracking(config)
    print(f"Saved {len(df)} rows to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
