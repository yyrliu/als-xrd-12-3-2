"""
Peak tracking configuration with CORRECTED validation bounds.

VERSION 2 - Fixes PbI2 false positive problem discovered in comprehensive analysis.

Key Changes from v1:
- Added peak-specific sigma_max bounds (was using global 0.5 for all peaks)
- PbI2: sigma_max = 0.10 (tightened from 0.15 → observed σ=0.040-0.084, need strict bounds)
- ITO: sigma_max = 0.25 (substrate peak, fairly narrow)
- 2D (002): sigma_max = 0.3 (perovskite, can vary during phase transitions)
- 1D (002): sigma_max = 0.3 (similar to 2D)
- MeOMBAI/ClMBAI: Keep sigma_max = 0.5 (intermittent, need flexibility)

Physical Rationale:
- PbI2 peaks are VERY narrow (σ ~ 0.040-0.084) due to excellent crystallinity
- ITO substrate peaks are narrow and stable (σ ~ 0.14-0.18)
- Perovskite 2D peaks can broaden during transitions (σ ~ 0.02-0.20)
- Organic peaks (MeOMBAI/ClMBAI) are weaker and more variable
"""
from __future__ import annotations

from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path(
    r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\peak_tracking_workflow_outputs_snr_v2"
)


def _mabai_peaks() -> list[tuple]:
    """MBAI peaks with corrected validation bounds."""
    return [
        (7.23, 2, (-1, 0), "2D (002)", {
            "drift_tolerance_deg": 0.05,
            "fitter": {"kind": "voigt_exp", "fit_half_width_deg": 0.4},
            # Peak-specific bounds
            "sigma_min": 0.01,
            "sigma_max": 0.3,  # Perovskite - allow broadening
            # Validation
            "min_snr": 3.0,
            "validate_fit_quality": True,
            # Adaptive bounds - moderate stability
            "use_adaptive_bounds": True,
            "adaptive_sigma_tolerance": 0.7,  # ±70% (wider for potential changes)
            "adaptive_min_consecutive": 10,
        }),
        (9.2, 1, (-1, 0), "1D (002)", {
            "drift_tolerance_deg": 0.05,
            # Peak-specific bounds
            "sigma_min": 0.01,
            "sigma_max": 0.3,  # Similar to 2D
            # Validation
            "min_snr": 3.0,
            "validate_fit_quality": True,
            # Adaptive bounds
            "use_adaptive_bounds": True,
            "adaptive_sigma_tolerance": 0.7,
            "adaptive_min_consecutive": 10,
        }),
        (13.6, 2, (0, -1), "PbI2", {
            "drift_tolerance_deg": 0.05,
            # Peak-specific bounds - CRITICAL FIX!
            "sigma_min": 0.02,
            "sigma_max": 0.10,  # ← KEY CHANGE: Observed σ=0.040-0.084, 0.10 gives margin
            # PbI2 is VERY narrow (σ ~ 0.040-0.084) due to excellent crystallinity
            # σ_max=0.10 is 2.5× the minimum, prevents false positives from broad fits
            # Validation
            "min_snr": 4.0,  # Higher (strong when present)
            "validate_fit_quality": True,
            # Adaptive bounds - reaction product
            "use_adaptive_bounds": True,
            "adaptive_sigma_tolerance": 0.5,  # ±50%
            "adaptive_min_consecutive": 10,  # Longer warm-up
        }),
        (30.3, 2, (0, -1), "ITO", {
            "drift_tolerance_deg": 0.05,
            # Peak-specific bounds
            "sigma_min": 0.05,
            "sigma_max": 0.25,  # ITO substrate peak, fairly narrow and stable
            # Validation
            "min_snr": 5.0,  # High (substrate, always strong)
            "validate_fit_quality": True,
            # Adaptive bounds - very stable
            "use_adaptive_bounds": True,
            "adaptive_sigma_tolerance": 0.3,  # ±30% (tight)
            "adaptive_amplitude_tolerance": 0.2,  # ±20% (very stable)
            "adaptive_min_consecutive": 5,  # Quick activation
        }),
    ]


def _meombai_peaks() -> list[tuple]:
    """MeOMBAI peaks with corrected bounds. Adaptive bounds ONLY for stable peaks."""
    return [
        (6.0, 0.5, (-1, 0), "2D (002)", {
            "drift_tolerance_deg": 0.15,
            # Peak-specific bounds
            "sigma_min": 0.01,
            "sigma_max": 0.3,  # Perovskite
            # Validation
            "min_snr": 3.0,
            "validate_fit_quality": True,
            # Adaptive bounds - moderate
            "use_adaptive_bounds": True,
            "adaptive_sigma_tolerance": 0.7,  # ±70% (wider)
            "adaptive_min_consecutive": 10,
        }),
        (13.3, 2.0, (0, -1), "PbI2", {
            "drift_tolerance_deg": 0.05,
            # Peak-specific bounds - CRITICAL FIX!
            "sigma_min": 0.02,
            "sigma_max": 0.10,  # ← KEY CHANGE: Very narrow bounds (observed σ=0.040-0.084)
            # Validation
            "min_snr": 4.0,  # Appears during reaction
            "validate_fit_quality": True,  # CRITICAL for appearance detection
            # Adaptive bounds - after it appears
            "use_adaptive_bounds": True,
            "adaptive_sigma_tolerance": 0.5,  # ±50%
            "adaptive_min_consecutive": 10,  # Wait for stability
        }),
        (6.8, 0.5, (-1, 0), "MeOMBAI", {
            "drift_tolerance_deg": 0.05,
            # Peak-specific bounds - keep wide for flexibility
            "sigma_min": 0.01,
            "sigma_max": 0.5,  # Keep wide - intermittent peak needs flexibility
            # Validation - CRITICAL for this weak peak!
            "min_snr": 2.5,  # Lower threshold (weak peak)
            "validate_fit_quality": True,  # ESSENTIAL - catches ~34% false positives
            # NO adaptive bounds - intermittent peak
            "use_adaptive_bounds": False,
        }),
        (30.3, 2, (0, -1), "ITO", {
            "drift_tolerance_deg": 0.05,
            # Peak-specific bounds
            "sigma_min": 0.05,
            "sigma_max": 0.25,  # ITO substrate
            # Validation
            "min_snr": 5.0,  # Very strong
            "validate_fit_quality": True,
            # Adaptive bounds - perfect candidate
            "use_adaptive_bounds": True,
            "adaptive_sigma_tolerance": 0.3,  # ±30% (tight)
            "adaptive_amplitude_tolerance": 0.2,  # ±20%
            "adaptive_min_consecutive": 5,
        }),
    ]

def _clmbai_peaks() -> list[tuple]:
    """ClMBAI peaks with corrected validation bounds."""
    return [
        (6.4, 1, (-1, 25), "2D (002)", {
            "drift_tolerance_deg": 0.05,
            "fitter": {"kind": "voigt_exp", "fit_half_width_deg": 0.4},
            # Peak-specific bounds
            "sigma_min": 0.01,
            "sigma_max": 0.3,  # Perovskite
            # Validation
            "min_snr": 3.0,
            "validate_fit_quality": True,
            # Adaptive bounds
            "use_adaptive_bounds": True,
            "adaptive_sigma_tolerance": 0.7,
            "adaptive_min_consecutive": 10,
        }),
        (7.3, 0.5, (-1, 0), "ClMBAI", {
            "drift_tolerance_deg": 0.05,
            # Peak-specific bounds - keep wide
            "sigma_min": 0.01,
            "sigma_max": 0.5,  # Keep wide for intermittent peak
            # Validation - may be weak/intermittent
            "min_snr": 2.5,
            "validate_fit_quality": True,
            # Conservative: no adaptive bounds
            "use_adaptive_bounds": False,
        }),
        (13.3, 2, (0, -1), "PbI2", {
            "drift_tolerance_deg": 0.05,
            # Peak-specific bounds - CRITICAL FIX!
            "sigma_min": 0.02,
            "sigma_max": 0.10,  # ← KEY CHANGE: Very narrow bounds (observed σ=0.040-0.084)
            # Validation
            "min_snr": 4.0,
            "validate_fit_quality": True,
            # Adaptive bounds
            "use_adaptive_bounds": True,
            "adaptive_sigma_tolerance": 0.5,
            "adaptive_min_consecutive": 10,
        }),
        (30.3, 2, (0, -1), "ITO", {
            "drift_tolerance_deg": 0.05,
            # Peak-specific bounds
            "sigma_min": 0.05,
            "sigma_max": 0.25,  # ITO substrate
            # Validation
            "min_snr": 5.0,
            "validate_fit_quality": True,
            # Adaptive bounds
            "use_adaptive_bounds": True,
            "adaptive_sigma_tolerance": 0.3,
            "adaptive_amplitude_tolerance": 0.2,
            "adaptive_min_consecutive": 5,
        }),
    ]


def _clmbai_260214_peaks() -> list[tuple]:
    """ClMBAI peaks for 260214 session with corrected bounds."""
    # Files from 260214 session (manual_calib.poni) have a slightly different
    # q→2θ mapping: the 2D (002) peak lands at ~5.79° instead of ~6.4°.
    # Other peaks scaled accordingly; ClMBAI/PbI2/ITO positions are approximate.
    return [
        (5.79, 1, (-1, 25), "2D (002)", {
            "drift_tolerance_deg": 0.05,
            "fitter": {"kind": "voigt_exp", "fit_half_width_deg": 0.4},
            # Peak-specific bounds
            "sigma_min": 0.01,
            "sigma_max": 0.3,
            # Validation
            "min_snr": 3.0,
            "validate_fit_quality": True,
            # Adaptive bounds
            "use_adaptive_bounds": True,
            "adaptive_sigma_tolerance": 0.7,
            "adaptive_min_consecutive": 10,
        }),
        (6.62, 0.5, (-1, 0), "ClMBAI", {
            "drift_tolerance_deg": 0.05,
            # Peak-specific bounds
            "sigma_min": 0.01,
            "sigma_max": 0.5,
            # Validation
            "min_snr": 2.5,
            "validate_fit_quality": True,
            # Conservative
            "use_adaptive_bounds": False,
        }),
        (13.3, 2, (0, -1), "PbI2", {
            "drift_tolerance_deg": 0.05,
            # Peak-specific bounds - CRITICAL FIX!
            "sigma_min": 0.02,
            "sigma_max": 0.10,  # ← KEY CHANGE: Very narrow (σ=0.040-0.084)
            # Validation
            "min_snr": 4.0,
            "validate_fit_quality": True,
            # Adaptive bounds
            "use_adaptive_bounds": True,
            "adaptive_sigma_tolerance": 0.5,
            "adaptive_min_consecutive": 10,
        }),
        (27.4, 2, (0, -1), "ITO", {
            "drift_tolerance_deg": 0.05,
            # Peak-specific bounds
            "sigma_min": 0.01,
            "sigma_max": 0.15,  # 260214 session: ITO appears narrower at 27.4°
            # Validation
            "min_snr": 5.0,
            "validate_fit_quality": True,
            # Adaptive bounds
            "use_adaptive_bounds": True,
            "adaptive_sigma_tolerance": 0.3,
            "adaptive_amplitude_tolerance": 0.2,
            "adaptive_min_consecutive": 5,
        }),
    ]


def _meombai_260214_peaks() -> list[tuple]:
    """MeOMBAI peaks for 260214 session with corrected bounds."""
    # Files from 260214 session (manual_calib.poni): 2D (002) peak at ~5.64°
    # instead of ~6.0°; other peaks scaled by the same factor (~0.94).
    return [
        (5.64, 0.5, (-1, 0), "2D (002)", {
            "drift_tolerance_deg": 0.15,
            # Peak-specific bounds
            "sigma_min": 0.01,
            "sigma_max": 0.3,
            # Validation
            "min_snr": 3.0,
            "validate_fit_quality": True,
            # Adaptive bounds
            "use_adaptive_bounds": True,
            "adaptive_sigma_tolerance": 0.7,
            "adaptive_min_consecutive": 10,
        }),
        (12.5, 2.0, (0, -1), "PbI2", {
            "drift_tolerance_deg": 0.05,
            # Peak-specific bounds - CRITICAL FIX!
            "sigma_min": 0.02,
            "sigma_max": 0.10,  # ← KEY CHANGE: Very narrow (σ=0.040-0.084)
            # Validation
            "min_snr": 4.0,
            "validate_fit_quality": True,
            # Adaptive bounds
            "use_adaptive_bounds": True,
            "adaptive_sigma_tolerance": 0.5,
            "adaptive_min_consecutive": 10,
        }),
        (6.38, 0.5, (-1, 0), "MeOMBAI", {
            "drift_tolerance_deg": 0.05,
            # Peak-specific bounds
            "sigma_min": 0.01,
            "sigma_max": 0.5,
            # Validation - CRITICAL
            "min_snr": 2.5,
            "validate_fit_quality": True,
            # NO adaptive bounds
            "use_adaptive_bounds": False,
        }),
    ]


RUN_CONFIGS = {
    ##### insitu runs #####
    "insitu_0.5M_MBAI": {
        "da_file": Path(
            r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\MBAI_0M5_3 004909 Images.nc"
        ),
        "global_shift_deg": 0.0,
        "keyframe_plot_margin_deg": 2.0,
        "peaks": _mabai_peaks(),
    },
    "insitu_0.5M_ClMBAI": {
        "da_file": Path(
            r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\ClMBAI_0M5_1 004910 Images.nc"
        ),
        "global_shift_deg": 0.0,
        "keyframe_plot_margin_deg": 2.0,
        "peaks": _clmbai_peaks(),
    },
    "insitu_0.5M_MeOMBAI":{
        "da_file": Path(
            r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\MeOMBAI_0M5_2 004912 Images.nc"
        ),
        "global_shift_deg": 0.0,
        "keyframe_plot_margin_deg": 2.0,
        "peaks": _meombai_peaks(),
    },
    ##### aging runs anneal 100c 120s #####
    "aging_100c_120s_anneal_MBAI": {
        "da_file": Path(
            r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\MBAI_aging_from_100C_2min_anneal 004913 Images.nc"
        ),
        "global_shift_deg": 0.0,
        "keyframe_plot_margin_deg": 2.0,
        "peaks": _mabai_peaks(),
    },
    "aging_100c_120s_anneal_MeOMBAI":{
            "da_file": Path(
                r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\MeOMBAI_aging_from_100C_2min_anneal 004914 Images.nc"
            ),
            "global_shift_deg": 0.0,
            "keyframe_plot_margin_deg": 2.0,
            "peaks": _meombai_peaks(),
    },
    "aging_100c_120s_anneal_ClMBAI":
    {
        "da_file": Path(
            r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\ClMBAI_aging_from_100C_2min_anneal 004915 Images.nc"
        ),
        "global_shift_deg": 0.0,
        "keyframe_plot_margin_deg": 2.0,
        "peaks": _clmbai_peaks(),
    },
    ##### aging runs anneal 120c 70s #####
    "aging_120c_70s_anneal_MeOMBAI": {
        "da_file": Path(
            r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\MeOMBAI_120c_aging_1 004868 Images.nc"
        ),
        "global_shift_deg": 0.0,
        "keyframe_plot_margin_deg": 2.0,
        "peaks": _meombai_260214_peaks(),  # 260214 session: peaks shifted to ~0.904× of standard 2θ
    },
    "aging_120c_70s_anneal_ClMBAI":
    {
        "da_file": Path(
            r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\ClMBAI_120c_aging_1 004871 Images.nc"
        ),
        "global_shift_deg": 0.0,
        "keyframe_plot_margin_deg": 2.0,
        "peaks": _clmbai_peaks(),
    },
    # aging runs anneal 120c 120s
    "aging_120c_120s_anneal_ClMBAI":
    {
        "da_file": Path(
            r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\ClMBAI_anneal_2min_120c_aging_1 004873 Images.nc"
        ),
        "global_shift_deg": 0.0,
        "keyframe_plot_margin_deg": 2.0,
        "peaks": _clmbai_260214_peaks(),  # 260214 session: 2D (002) at 5.79° not 6.4°
    },
    "aging_120c_120s_anneal_MeOMBAI": {
        "da_file": Path(
            r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\MeOMBAI_anneal_2min_120c_aging_1 004872 Images.nc"
        ),
        "global_shift_deg": 0.0,
        "keyframe_plot_margin_deg": 2.0,
        "peaks": _meombai_peaks(),
    },
}


PEAK_PRESETS = {
    "MBAI": _mabai_peaks(),
    "MeOMBAI": _meombai_peaks(),
    "ClMBAI": _clmbai_peaks(),
}
