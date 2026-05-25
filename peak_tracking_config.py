from __future__ import annotations

from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path(
    r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\agent_temp"
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


RUN_CONFIGS = {
    "MeOMBAI_120c_aging_1 004868 Images": {
        "da_file": Path(
            r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\MeOMBAI_120c_aging_1 004868 Images.nc"
        ),
        "global_shift_deg": 0.0,
        "keyframe_plot_margin_deg": 2.0,
        "peaks": _meombai_peaks(),
    },
    "MBAI_0M5_3 004909 Images": {
        "da_file": Path(
            r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\MBAI_0M5_3 004909 Images.nc"
        ),
        "global_shift_deg": 0.0,
        "keyframe_plot_margin_deg": 5.0,
        "peaks": _mabai_peaks(),
    },
    # Demo presets for additive vs restrict key_frame behavior
    "MeOMBAI_demo_additive": {
        "da_file": Path(r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\MeOMBAI_120c_aging_1 004868 Images.nc"),
        "global_shift_deg": 0.0,
        "keyframe_plot_margin_deg": 5.0,
        "peaks": [
            (6.0, 0.5, (0, -1), "2D (002)", {"key_frames": [0, 3], "key_frames_mode": "additive", "auto_keyframe_every": 15}),
        ],
    },
    "MeOMBAI_demo_restrict": {
        "da_file": Path(r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\MeOMBAI_120c_aging_1 004868 Images.nc"),
        "global_shift_deg": 0.0,
        "keyframe_plot_margin_deg": 5.0,
        "peaks": [
            (6.0, 0.5, (0, -1), "2D (002)", {"key_frames": [0, 2], "key_frames_mode": "restrict", "auto_keyframe_every": 15}),
        ],
    },
}


PEAK_PRESETS = {
    "MBAI": _mabai_peaks(),
    "MeOMBAI": _meombai_peaks(),
}
