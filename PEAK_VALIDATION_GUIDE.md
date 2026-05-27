# Peak Validation & Adaptive Bounds Guide

## Overview

This guide explains the two-stage validation system and adaptive parameter bounds added to the peak tracking workflow. These features help distinguish real peaks from noise and improve tracking stability.

## Problem Statement

**Before**: The workflow would mark fits as "successful" even when:
- ❌ No peak actually exists (fitting noise/background)
- ❌ Optimizer parameters hit bounds (σ = 0.5, indicating struggle)
- ❌ Amplitude ≈ 0 (essentially fitting nothing)

**Result**: ~34% of "successful" MeOMBAI fits were false positives!

## Solution: Two-Stage Validation

### Stage 1: Peak Presence Check (Pre-Fit)

**Before fitting**, check if there's actually a peak worth fitting:

```python
PeakSpec(
    name="MeOMBAI",
    min_snr=3.0,  # Minimum signal-to-noise ratio
)
```

**How it works**:
1. Estimates baseline from bottom 10% of data
2. Calculates noise from lower quartile
3. Computes SNR = (peak_max - baseline) / noise
4. If SNR < threshold → Mark as `peak_absent` (skip fitting)

**Benefits**:
- ✅ Saves compute time (no fitting noise)
- ✅ Cleaner tracking status (`peak_absent` vs `fit_failed`)
- ✅ Can track when peaks appear/disappear in time-series

### Stage 2: Fit Quality Validation (Post-Fit)

**After fitting**, validate the result makes physical sense:

```python
PeakSpec(
    name="MeOMBAI",
    validate_fit_quality=True,  # Default: True
)
```

**Rejection criteria**:
1. **Sigma/gamma at bounds** (within 5%)
   - Indicates optimizer struggled to find meaningful peak
2. **Amplitude < 1e-4**
   - Essentially zero, fitting noise
3. **Peak height < 5% of data maximum**
   - Too weak relative to data scale
4. **Reduced chi-square > 20**
   - Poor fit quality

**Benefits**:
- ✅ Rejects false positives (fitted noise)
- ✅ Status: `fit_invalid` (different from `peak_absent`)
- ✅ Can retry with wider bounds (fallback mechanism)

## Adaptive Parameter Bounds

### Motivation

**Observation from real data**:
- **ITO, PbI2** (strong peaks): σ varies only 5-28% frame-to-frame
- **MeOMBAI** (weak peak): σ varies 100%+, appears/disappears

**Insight**: Strong peaks don't need wide bounds every frame!

### How It Works

```python
PeakSpec(
    name="ITO",
    use_adaptive_bounds=True,              # Enable adaptive bounds
    adaptive_min_consecutive=5,             # Wait for 5 good fits
    adaptive_sigma_tolerance=0.5,          # Allow ±50% change
    adaptive_amplitude_tolerance=0.5,      # Allow ±50% change
    fallback_on_validation_failure=True,   # Retry if validation fails
)
```

**Workflow**:
1. **Warm-up**: First N frames use wide bounds (default: 5)
2. **Activation**: After N consecutive successful fits, tighten bounds:
   - σ_new ∈ [σ_prev × 0.5, σ_prev × 1.5]
   - γ_new ∈ [γ_prev × 0.5, γ_prev × 1.5]
   - amp_new ∈ [amp_prev × 0.5, amp_prev × 1.5]
3. **Fallback**: If validation fails with tight bounds, retry with wide bounds
4. **Reset**: Key frames reset to wide bounds

**Benefits**:
- ✅ Faster convergence (fewer optimizer iterations)
- ✅ Prevents unrealistic parameter jumps
- ✅ Maintains flexibility for real changes
- ✅ Safe fallback if bounds too tight

### When to Use

| Peak Type | Adaptive Bounds | Reason |
|-----------|----------------|---------|
| **Strong, stable** (ITO, PbI2) | ✅ YES | Consistent signal, smooth evolution |
| **Weak/transient** (MeOMBAI) | ❌ NO | Appears/disappears, needs flexibility |
| **With phase transitions** | ⚠️ MAYBE | Use wider tolerance + key frames |

## Configuration Examples

### Example 1: Strong Substrate Peak (ITO)

```python
{
    "name": "ITO",
    "center_deg": 30.6,
    "window_deg": 1.0,
    "start_idx": 0,
    "stop_idx": -1,
    "drift_tolerance_deg": 0.05,
    # Validation
    "min_snr": 5.0,                        # Strong peak, higher threshold
    "validate_fit_quality": True,
    # Adaptive bounds
    "use_adaptive_bounds": True,           # Enable for efficiency
    "adaptive_min_consecutive": 5,
    "adaptive_sigma_tolerance": 0.3,       # Tight tolerance (30%)
    "adaptive_amplitude_tolerance": 0.2,   # Very stable amplitude
}
```

### Example 2: Weak, Transient Peak (MeOMBAI)

```python
{
    "name": "MeOMBAI",
    "center_deg": 7.5,
    "window_deg": 1.5,
    "start_idx": 0,
    "stop_idx": -1,
    "drift_tolerance_deg": 0.05,
    # Validation
    "min_snr": 2.5,                        # Lower threshold (weak peak)
    "validate_fit_quality": True,          # CRITICAL: catch bad fits
    # NO adaptive bounds
    "use_adaptive_bounds": False,          # Peak appears/disappears
}
```

### Example 3: Peak with Phase Transition (2D Perovskite)

```python
{
    "name": "2D_002",
    "center_deg": 6.0,
    "window_deg": 1.0,
    "start_idx": 0,
    "stop_idx": -1,
    "drift_tolerance_deg": 0.15,
    "key_frames": [0, 200, 400, 600],      # Reset at transitions
    # Validation
    "min_snr": 3.0,
    "validate_fit_quality": True,
    # Adaptive with wider tolerance
    "use_adaptive_bounds": True,
    "adaptive_min_consecutive": 10,        # Longer warm-up
    "adaptive_sigma_tolerance": 0.7,       # 70% tolerance (wider)
    "adaptive_amplitude_tolerance": 0.5,
}
```

## Output Interpretation

### New CSV Columns

| Column | Type | Description |
|--------|------|-------------|
| `peak_snr` | float | Signal-to-noise ratio from pre-fit check |
| `peak_presence_check` | str | "passed" or "failed" |
| `validation_reason` | str | Why post-fit validation failed (if applicable) |
| `used_adaptive_bounds` | bool | Whether adaptive bounds were used |
| `consecutive_good_fits` | int | Counter for tracking stability |

### Status Codes

| fit_status | Meaning | Action |
|-----------|---------|--------|
| `peak_absent` | No peak detected (low SNR) | Expected if peak hasn't formed yet |
| `fit_invalid` | Fit succeeded but validation failed | Check fit_plots, may need parameter tuning |
| `no_candidate` | No peak found by scipy.find_peaks | Check min_prominence threshold |
| `track` | Successfully tracking peak | Normal operation |
| `search` | Initial search phase | Normal at start |

## Troubleshooting

### Too Many `peak_absent` Frames

**Symptom**: Peak is clearly visible in data but marked absent

**Solutions**:
1. Lower `min_snr` threshold (try 2.0-2.5)
2. Check if baseline correction is too aggressive
3. Verify `min_prominence` isn't too high

### Too Many `fit_invalid` Frames

**Symptom**: Fits are rejected but look reasonable in plots

**Solutions**:
1. Disable validation temporarily: `validate_fit_quality=False`
2. Check specific `validation_reason` in CSV
3. If "Sigma at bound", widen `window_deg` or adjust `drift_tolerance_deg`
4. If "Peak too weak", this may be correct - peak is genuinely weak

### Adaptive Bounds Not Activating

**Symptom**: `used_adaptive_bounds` always False

**Check**:
1. `use_adaptive_bounds=True` in config
2. `consecutive_good_fits` reaching `adaptive_min_consecutive`
3. Peak is actually tracking successfully (not intermittent failures)

### Adaptive Bounds Too Restrictive

**Symptom**: Fit quality degrades after adaptive bounds activate

**Solutions**:
1. Increase tolerance: `adaptive_sigma_tolerance=0.7` (70%)
2. Increase warm-up: `adaptive_min_consecutive=10`
3. Enable fallback: `fallback_on_validation_failure=True` (default)
4. Consider disabling for this peak

## Performance Impact

### Computational Cost

- **Pre-fit SNR check**: +0.1ms per frame (negligible)
- **Post-fit validation**: +0.5ms per frame (negligible)
- **Adaptive bounds**: -10-30% fitting time (faster convergence!)

### Overall

✅ **Net positive**: Validation overhead < time saved by skipping bad fits + faster convergence

## Best Practices

1. **Start conservative**: Enable validation, disable adaptive bounds
2. **Analyze one run**: Check `peak_snr`, `validation_reason` distributions
3. **Enable adaptive selectively**: Only for stable peaks
4. **Monitor diagnostics**: Track `consecutive_good_fits`, `used_adaptive_bounds`
5. **Iterate**: Adjust thresholds based on fit_plots and physical expectations

## References

- Implementation: `peak_tracking_workflow.py`
- Development log: `DEVELOPMENT_LOG.md`
- Analysis findings: `TRACKING_ANALYSIS_FINDINGS.md`
