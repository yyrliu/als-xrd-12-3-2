"""Peak tracking run configurations for the Yi-Ru Feb 2026 ALS beamtime.

This module is loaded dynamically by ``peak_tracking_workflow.py``. It must
define:

- ``DEFAULT_OUTPUT_ROOT`` (Path): Default directory for output folders.
- ``RUN_CONFIGS`` (dict[str, dict]): Mapping of run name → config dict.

Each config dict accepts the keys documented in :class:`RunConfig` and
:class:`PeakSpec`. Peaks can be supplied as tuples of the form::

    (center_deg, window_deg, (start_idx, stop_idx), name, extras_dict)

or as plain dicts matching :class:`PeakSpec` field names.

Usage::

    uv run python peak_tracking_workflow.py --config peak_tracking_config.py --list-runs
    uv run python peak_tracking_workflow.py --config peak_tracking_config.py --run insitu_0.5M_MeOMBAI

See PEAK_TRACKING_GUIDE.md for a complete parameter reference.

Notes on session calibration
-----------------------------
Files from the **260214 session** used a different poni file (``manual_calib.poni``),
which shifts all 2\u03b8 positions by ~0.94\u00d7 relative to the standard calibration.
Those runs use ``_260214_peaks()`` helpers with corrected center values; all other
runs use the standard ``_*_peaks()`` helpers.
"""
from __future__ import annotations

from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path(
    r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\peak_tracking_workflow_outputs"
)


def _mabai_peaks() -> list[tuple]:
    """Peak list for MBAI (methylammonium bismuth ammonium iodide) samples.

    Covers: 2D (002) perovskite, 1D (002), PbI2, and ITO substrate peaks.
    The 2D (002) peak uses a ``voigt_exp`` fitter with a narrow fit window
    (0.4°) to avoid the exponential background tail biasing the fit.
    """
    return [
        (7.23, 2, (-1, 0), "2D (002)", {
            "drift_tolerance_deg": 0.05,
            "fitter": {"kind": "voigt_exp", "fit_half_width_deg": 0.4},
        }),
        (9.2, 1, (-1, 0), "1D (002)", {"drift_tolerance_deg": 0.05}),
        (13.6, 2, (0, -1), "PbI2", {"drift_tolerance_deg": 0.05}),
        (30.3, 2, (0, -1), "ITO", {"drift_tolerance_deg": 0.05}),
    ]


def _meombai_peaks() -> list[tuple]:
    """Peak list for MeOMBAI (methoxy-methylbutylammonium iodide) samples.

    MeOMBAI appears as a weak, intermittent peak at ~6.8° that may be absent
    for extended periods. Validation is important; adaptive bounds are not
    recommended for this peak.
    """
    return [
        (6.0, 0.5, (-1, 0), "2D (002)", {"drift_tolerance_deg": 0.15}),
        (13.3, 2.0, (0, -1), "PbI2", {"drift_tolerance_deg": 0.05}),
        (6.8, 0.5, (-1, 0), "MeOMBAI", {"drift_tolerance_deg": 0.05}),
        (30.3, 2, (0, -1), "ITO", {"drift_tolerance_deg": 0.05}),
    ]

def _clmbai_peaks() -> list[tuple]:
    """Peak list for ClMBAI (chloro-methylbutylammonium iodide) samples.

    ClMBAI appears at ~7.3° and is similarly weak and intermittent to MeOMBAI.
    The 2D (002) peak uses a ``voigt_exp`` fitter to handle the background tail.
    Frame range for 2D (002) is restricted to frames -1 through 25 (backwards)
    where the peak is well-formed before being obscured by the growing 3D phase.
    """
    return [
        (6.4, 1, (-1, 25), "2D (002)", {
            "drift_tolerance_deg": 0.05,
            "fitter": {"kind": "voigt_exp", "fit_half_width_deg": 0.4},
        }),
        (7.3, 0.5, (-1, 0), "ClMBAI", {"drift_tolerance_deg": 0.05}),
        (13.3, 2, (0, -1), "PbI2", {"drift_tolerance_deg": 0.05}),
        (30.3, 2, (0, -1), "ITO", {"drift_tolerance_deg": 0.05}),
    ]


def _clmbai_260214_peaks() -> list[tuple]:
    """Peak list for ClMBAI samples from the 260214 calibration session.

    The 260214 session used ``manual_calib.poni``, which shifts all peaks to
    approximately 0.904× of their standard 2θ values. The 2D (002) peak lands
    at ~5.79° instead of the usual ~6.4°. The ITO substrate also shifts from
    30.3° to ~27.4°.
    """
    return [
        (5.79, 1, (-1, 25), "2D (002)", {
            "drift_tolerance_deg": 0.05,
            "fitter": {"kind": "voigt_exp", "fit_half_width_deg": 0.4},
        }),
        (6.62, 0.5, (-1, 0), "ClMBAI", {"drift_tolerance_deg": 0.05}),
        (13.3, 2, (0, -1), "PbI2", {"drift_tolerance_deg": 0.05}),
        (27.4, 2, (0, -1), "ITO", {"drift_tolerance_deg": 0.05}),
    ]


def _meombai_260214_peaks() -> list[tuple]:
    """Peak list for MeOMBAI samples from the 260214 calibration session.

    Same 0.904× scaling as :func:`_clmbai_260214_peaks`. The 2D (002) peak
    is at ~5.64° and the MeOMBAI peak at ~6.38°. ITO is not included because
    it falls outside the reliable 2θ range for this calibration.
    """
    return [
        (5.64, 0.5, (-1, 0), "2D (002)", {"drift_tolerance_deg": 0.15}),
        (12.5, 2.0, (0, -1), "PbI2", {"drift_tolerance_deg": 0.05}),
        (6.38, 0.5, (-1, 0), "MeOMBAI", {"drift_tolerance_deg": 0.05}),
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
