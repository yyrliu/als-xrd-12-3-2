# Example Configuration: Using Validation & Adaptive Bounds

This file shows a complete example of how to configure peaks with the new validation and adaptive bounds features.

## Complete Example Configuration

```python
# In peak_tracking_config.py - RUN_CONFIGS dictionary

"example_with_validation": {
    "description": "Example showing validation + adaptive bounds for mixed peak types",
    "data_path": "path/to/your/data.nc",
    "output_base": "output",
    "global_shift_deg": 0.1,
    "min_prominence": 5,
    "peaks": [
        # EXAMPLE 1: Strong, stable substrate peak
        # - Always present, high intensity, smooth evolution
        # - ENABLE adaptive bounds for efficiency
        {
            "name": "ITO",
            "center_deg": 30.6,
            "window_deg": 1.0,
            "start_idx": 0,
            "stop_idx": -1,
            "drift_tolerance_deg": 0.05,
            "model_type": "voigt",
            "baseline_order": 0,
            "apply_baseline_correction": False,
            "keyframe_plot_margin_deg": 0.5,
            
            # VALIDATION SETTINGS
            "min_snr": 5.0,                      # High threshold (strong peak)
            "validate_fit_quality": True,        # Catch any issues
            
            # ADAPTIVE BOUNDS (ENABLED)
            "use_adaptive_bounds": True,         # Enable for stable peak
            "adaptive_min_consecutive": 5,       # 5 frame warm-up
            "adaptive_sigma_tolerance": 0.3,     # ±30% (tight)
            "adaptive_amplitude_tolerance": 0.2, # ±20% (very stable)
            "fallback_on_validation_failure": True,  # Safe fallback
        },
        
        # EXAMPLE 2: Reaction product that appears mid-experiment
        # - Initially absent, then appears and grows
        # - ENABLE adaptive bounds AFTER it appears
        {
            "name": "PbI2",
            "center_deg": 12.7,
            "window_deg": 1.0,
            "start_idx": 0,
            "stop_idx": -1,
            "drift_tolerance_deg": 0.08,
            "model_type": "voigt",
            "baseline_order": 1,
            "apply_baseline_correction": True,
            "keyframe_plot_margin_deg": 0.5,
            
            # VALIDATION SETTINGS
            "min_snr": 3.0,                      # Lower threshold (grows over time)
            "validate_fit_quality": True,        # Critical for appearance detection
            
            # ADAPTIVE BOUNDS (ENABLED with longer warm-up)
            "use_adaptive_bounds": True,         # Enable once it appears
            "adaptive_min_consecutive": 10,      # Longer warm-up (10 frames)
            "adaptive_sigma_tolerance": 0.5,     # ±50% (moderate)
            "adaptive_amplitude_tolerance": 0.5, # ±50% (allows growth)
            "fallback_on_validation_failure": True,
        },
        
        # EXAMPLE 3: Weak, intermittent peak
        # - Low intensity, may appear/disappear
        # - DISABLE adaptive bounds (needs full flexibility)
        {
            "name": "MeOMBAI",
            "center_deg": 7.5,
            "window_deg": 1.5,
            "start_idx": 0,
            "stop_idx": -1,
            "drift_tolerance_deg": 0.05,
            "model_type": "pseudovoigt",
            "baseline_order": 1,
            "apply_baseline_correction": True,
            "keyframe_plot_margin_deg": 0.8,
            
            # VALIDATION SETTINGS (CRITICAL!)
            "min_snr": 2.5,                      # Lower threshold (weak peak)
            "validate_fit_quality": True,        # ESSENTIAL - catches false positives
            
            # ADAPTIVE BOUNDS (DISABLED)
            "use_adaptive_bounds": False,        # Keep wide bounds (intermittent)
            # (other adaptive params ignored when disabled)
        },
        
        # EXAMPLE 4: Peak with known phase transition
        # - Changes properties at specific frames
        # - Use key frames + wider adaptive tolerance
        {
            "name": "2D_002",
            "center_deg": 6.0,
            "window_deg": 1.0,
            "start_idx": 0,
            "stop_idx": -1,
            "drift_tolerance_deg": 0.15,
            "key_frames": [0, 200, 400, 600],    # Reset at transitions
            "model_type": "gaussian",
            "baseline_order": 1,
            "apply_baseline_correction": True,
            "keyframe_plot_margin_deg": 0.6,
            
            # VALIDATION SETTINGS
            "min_snr": 3.0,                      # Moderate threshold
            "validate_fit_quality": True,
            
            # ADAPTIVE BOUNDS (with wide tolerance)
            "use_adaptive_bounds": True,
            "adaptive_min_consecutive": 10,      # Longer warm-up
            "adaptive_sigma_tolerance": 0.7,     # ±70% (wide tolerance)
            "adaptive_amplitude_tolerance": 0.7, # ±70% (allows transitions)
            "fallback_on_validation_failure": True,
        },
    ],
},
```

## Understanding Each Parameter

### Validation Parameters

```python
# min_snr: Signal-to-noise ratio threshold for pre-fit check
# - Higher values (5.0+) = only fit strong peaks
# - Lower values (2.0-3.0) = fit weak peaks too
# - Too low = may fit noise
# - Too high = may miss real but weak peaks
"min_snr": 3.0,

# validate_fit_quality: Enable post-fit validation
# - Always True unless debugging
# - Catches: sigma at bounds, amp≈0, weak peaks, poor fits
"validate_fit_quality": True,
```

### Adaptive Bounds Parameters

```python
# use_adaptive_bounds: Main on/off switch
# - Enable for: stable, continuous peaks
# - Disable for: intermittent, weak, or noisy peaks
"use_adaptive_bounds": True,

# adaptive_min_consecutive: Warm-up period
# - Number of consecutive successful fits before enabling
# - Typical: 5-10 frames
# - Longer for: peaks that appear mid-experiment
"adaptive_min_consecutive": 5,

# adaptive_sigma_tolerance: Width constraint
# - 0.3 = ±30% (tight, for very stable peaks)
# - 0.5 = ±50% (moderate, typical)
# - 0.7 = ±70% (wide, for changing peaks)
"adaptive_sigma_tolerance": 0.5,

# adaptive_amplitude_tolerance: Height constraint
# - Similar to sigma_tolerance
# - Can be tighter if peak height is very stable
"adaptive_amplitude_tolerance": 0.5,

# fallback_on_validation_failure: Safety net
# - If validation fails with adaptive bounds, retry with wide bounds
# - Should always be True (default)
"fallback_on_validation_failure": True,
```

## Decision Tree for Configuration

```
START: What type of peak do you have?
│
├─── STRONG & ALWAYS PRESENT (ITO, substrate peaks)
│    ├─ min_snr: 5.0
│    ├─ validate_fit_quality: True
│    ├─ use_adaptive_bounds: True
│    ├─ adaptive_min_consecutive: 5
│    └─ adaptive_sigma_tolerance: 0.3-0.5
│
├─── REACTION PRODUCT (appears during experiment)
│    ├─ min_snr: 3.0-4.0
│    ├─ validate_fit_quality: True (CRITICAL)
│    ├─ use_adaptive_bounds: True
│    ├─ adaptive_min_consecutive: 10 (longer warm-up)
│    └─ adaptive_sigma_tolerance: 0.5-0.7
│
├─── WEAK / INTERMITTENT (low intensity, on/off)
│    ├─ min_snr: 2.5-3.0
│    ├─ validate_fit_quality: True (ESSENTIAL)
│    └─ use_adaptive_bounds: False (keep flexible)
│
└─── PHASE TRANSITION (changes properties)
     ├─ key_frames: [0, 100, 200, ...] (reset points)
     ├─ min_snr: 3.0
     ├─ validate_fit_quality: True
     ├─ use_adaptive_bounds: True
     ├─ adaptive_min_consecutive: 10
     └─ adaptive_sigma_tolerance: 0.7 (wide)
```

## Common Patterns

### Pattern 1: Default Safe Configuration
Use this when unsure:
```python
{
    "min_snr": 3.0,                      # Moderate
    "validate_fit_quality": True,        # Always safe
    "use_adaptive_bounds": False,        # Start conservative
}
```

### Pattern 2: Maximum Performance (stable peaks)
Use for known stable peaks:
```python
{
    "min_snr": 5.0,                      # High bar
    "validate_fit_quality": True,
    "use_adaptive_bounds": True,         # Enable
    "adaptive_min_consecutive": 5,
    "adaptive_sigma_tolerance": 0.3,     # Tight
}
```

### Pattern 3: Catch False Positives (weak peaks)
Use for low-intensity peaks:
```python
{
    "min_snr": 2.5,                      # Low (detects weak peaks)
    "validate_fit_quality": True,        # CRITICAL (catches noise fits)
    "use_adaptive_bounds": False,        # Keep flexible
}
```

## Testing Your Configuration

### Step 1: Run with defaults
```bash
uv run python peak_tracking_workflow.py --run your_config
```

### Step 2: Review outputs
```python
import pandas as pd

df = pd.read_csv("output/your_run/peak_ITO.csv")

# Check SNR distribution
print(df['peak_snr'].describe())

# Count statuses
print(df['fit_status'].value_counts())

# Check adaptive activation
print(f"Adaptive used: {df['used_adaptive_bounds'].sum()} / {len(df)} frames")
```

### Step 3: Visual inspection
- Open `fit_plots/fit_ITO_idx*.png`
- Verify `peak_absent` frames actually have no peak
- Verify `fit_invalid` frames are correctly rejected
- Check that `track` status frames look good

### Step 4: Iterate
- Too many `peak_absent`? → Lower `min_snr`
- Too many `fit_invalid`? → Check `validation_reason`, may be correct
- Want faster performance? → Enable adaptive bounds for stable peaks

## Migrating Existing Configurations

If you have an existing configuration without these parameters, it will still work! All new parameters are optional with safe defaults:

```python
# Your existing config:
{
    "name": "ITO",
    "center_deg": 30.6,
    "window_deg": 1.0,
    # ... other existing params ...
}

# Automatically uses these defaults:
# min_snr: 3.0
# validate_fit_quality: True
# use_adaptive_bounds: False
# fallback_on_validation_failure: True
```

To opt-in to new features, just add the parameters you want to change.
