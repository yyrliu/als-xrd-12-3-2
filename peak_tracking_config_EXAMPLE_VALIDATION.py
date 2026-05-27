"""
Example: How to add validation and adaptive bounds to peak_tracking_config.py

This shows how to use the new validation and adaptive bounds features with the 
existing tuple-based configuration format.
"""

from pathlib import Path

# Example 1: Enable adaptive bounds for stable peaks (ITO, PbI2)
def _meombai_peaks_with_validation() -> list[tuple]:
    return [
        # 2D (002) - Moderate stability, wider tolerance
        (6.0, 0.5, (-1, 0), "2D (002)", {
            "drift_tolerance_deg": 0.15,
            "min_snr": 3.0,                      # Moderate threshold
            "validate_fit_quality": True,
            "use_adaptive_bounds": True,         # Enable
            "adaptive_sigma_tolerance": 0.7,     # ±70% (wider, allows changes)
            "adaptive_min_consecutive": 10,      # Longer warm-up
        }),
        
        # PbI2 - Appears during reaction, then stable
        (13.3, 2.0, (0, -1), "PbI2", {
            "drift_tolerance_deg": 0.05,
            "min_snr": 4.0,                      # Higher (strong when present)
            "validate_fit_quality": True,        # Critical for appearance detection
            "use_adaptive_bounds": True,         # Enable after it appears
            "adaptive_sigma_tolerance": 0.5,     # ±50%
            "adaptive_amplitude_tolerance": 0.5, # ±50%
            "adaptive_min_consecutive": 10,      # Wait for stability
        }),
        
        # MeOMBAI - Weak, intermittent - DO NOT use adaptive bounds
        (6.8, 0.5, (-1, 0), "MeOMBAI", {
            "drift_tolerance_deg": 0.05,
            "min_snr": 2.5,                      # Lower threshold (weak peak)
            "validate_fit_quality": True,        # ESSENTIAL - catches false positives
            "use_adaptive_bounds": False,        # Keep wide bounds (intermittent)
        }),
        
        # ITO - Very stable substrate peak
        (30.3, 2, (0, -1), "ITO", {
            "drift_tolerance_deg": 0.05,
            "min_snr": 5.0,                      # High threshold (always strong)
            "validate_fit_quality": True,
            "use_adaptive_bounds": True,         # Perfect candidate
            "adaptive_sigma_tolerance": 0.3,     # ±30% (tight)
            "adaptive_amplitude_tolerance": 0.2, # ±20% (very stable)
            "adaptive_min_consecutive": 5,       # Quick activation
        }),
    ]


# Example 2: Conservative configuration (validation only, no adaptive bounds)
def _meombai_peaks_validation_only() -> list[tuple]:
    """Safe configuration - just adds validation, keeps wide bounds"""
    return [
        (6.0, 0.5, (-1, 0), "2D (002)", {
            "drift_tolerance_deg": 0.15,
            "min_snr": 3.0,
            "validate_fit_quality": True,
        }),
        (13.3, 2.0, (0, -1), "PbI2", {
            "drift_tolerance_deg": 0.05,
            "min_snr": 3.0,
            "validate_fit_quality": True,
        }),
        (6.8, 0.5, (-1, 0), "MeOMBAI", {
            "drift_tolerance_deg": 0.05,
            "min_snr": 2.5,                     # Lower for weak peak
            "validate_fit_quality": True,       # Critical!
        }),
        (30.3, 2, (0, -1), "ITO", {
            "drift_tolerance_deg": 0.05,
            "min_snr": 5.0,
            "validate_fit_quality": True,
        }),
    ]


# Example 3: Minimal changes to existing config
def _meombai_peaks_existing_plus_validation() -> list[tuple]:
    """Add only validation to existing peaks - minimal disruption"""
    return [
        # Just add min_snr to existing configs
        (6.0, 0.5, (-1, 0), "2D (002)", {
            "drift_tolerance_deg": 0.15,
            "min_snr": 3.0,                     # Add this
        }),
        (13.3, 2.0, (0, -1), "PbI2", {
            "drift_tolerance_deg": 0.05,
            "min_snr": 3.0,                     # Add this
        }),
        (6.8, 0.5, (-1, 0), "MeOMBAI", {
            "drift_tolerance_deg": 0.05,
            "min_snr": 2.5,                     # Add this (lower for weak)
        }),
        (30.3, 2, (0, -1), "ITO", {
            "drift_tolerance_deg": 0.05,
            "min_snr": 5.0,                     # Add this
        }),
    ]
    # Note: validate_fit_quality=True by default, so don't need to specify


# Full config example
RUN_CONFIGS_EXAMPLE = {
    "insitu_0.5M_MeOMBAI_with_validation": {
        "da_file": Path(
            r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\MeOMBAI_0M5_2 004912 Images.nc"
        ),
        "global_shift_deg": 0.0,
        "keyframe_plot_margin_deg": 2.0,
        "peaks": _meombai_peaks_with_validation(),  # Use the enhanced config
    },
}


"""
NOTES:

1. Backward Compatibility:
   - Existing configs work unchanged (all new params have defaults)
   - Just adding these params to the extras dict enables new features

2. Recommended Approach:
   - Start with validation only (Example 2 or 3)
   - Run once, review results
   - Then enable adaptive bounds for stable peaks (Example 1)

3. Key Decision Guide:
   
   Strong, always present (ITO, substrates):
   ├─ min_snr: 5.0
   ├─ validate_fit_quality: True
   └─ use_adaptive_bounds: True (with tight tolerance)

   Reaction products (PbI2, appears later):
   ├─ min_snr: 3.0-4.0
   ├─ validate_fit_quality: True
   └─ use_adaptive_bounds: True (with longer warm-up)

   Weak/intermittent (MeOMBAI):
   ├─ min_snr: 2.5
   ├─ validate_fit_quality: True
   └─ use_adaptive_bounds: False (keep flexible)

4. Testing Workflow:
   
   Step 1: Run with validation only
   >>> uv run python peak_tracking_workflow.py --run insitu_0.5M_MeOMBAI_with_validation
   
   Step 2: Review outputs
   >>> import pandas as pd
   >>> df = pd.read_csv("output/peak_ITO.csv")
   >>> print(df['peak_snr'].describe())
   >>> print(df['fit_status'].value_counts())
   
   Step 3: Enable adaptive bounds for stable peaks
   (Edit config, re-run)
   
   Step 4: Compare performance
   >>> print(df['used_adaptive_bounds'].sum(), "/ ", len(df), "frames")
"""
