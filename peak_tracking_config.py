from __future__ import annotations

from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path(
    r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\peak_tracking_workflow_outputs"
)


def _mabai_peaks() -> list[tuple]:
    return [
        (7.23, 2, (-1, 0), "2D (002)", {"drift_tolerance_deg": 0.05}),
        (9.2, 1, (-1, 0), "1D (002)", {"drift_tolerance_deg": 0.05}),
        (13.6, 2, (0, -1), "PbI2", {"drift_tolerance_deg": 0.05}),
        (30.3, 2, (0, -1), "ITO", {"drift_tolerance_deg": 0.05}),
    ]


def _meombai_peaks() -> list[tuple]:
    return [
        (6.0, 0.5, (-1, 0), "2D (002)", {"drift_tolerance_deg": 0.15}),
        (13.3, 2.0, (0, -1), "PbI2", {"drift_tolerance_deg": 0.05}),
        (6.8, 0.5, (-1, 0), "MeOMBAI", {"drift_tolerance_deg": 0.05}),
    ]

def _clmbai_peaks() -> list[tuple]:
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
    # Files from 260214 session (manual_calib.poni) have a slightly different
    # q→2θ mapping: the 2D (002) peak lands at ~5.79° instead of ~6.4°.
    # Other peaks scaled accordingly; ClMBAI/PbI2/ITO positions are approximate.
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
    # Files from 260214 session (manual_calib.poni): 2D (002) peak at ~5.64°
    # instead of ~6.0°; other peaks scaled by the same factor (~0.94).
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
