# Quick Reference: Peak Validation & Adaptive Bounds

## Quick Decision Guide

```
Is your peak strong and always present? (ITO, substrate peaks)
├─ YES → Enable adaptive bounds
│         use_adaptive_bounds: True
│         adaptive_min_consecutive: 5
│         adaptive_sigma_tolerance: 0.3-0.5
│
└─ NO → Is it weak or appears/disappears? (MeOMBAI, reaction products)
         ├─ YES → Keep wide bounds, enable validation
         │         use_adaptive_bounds: False
         │         validate_fit_quality: True
         │         min_snr: 2.5-3.0
         │
         └─ MAYBE → Has phase transitions? (2D perovskites)
                     use_adaptive_bounds: True
                     adaptive_sigma_tolerance: 0.7  # wider
                     key_frames: [0, 100, 200, ...]  # reset points
```

## Configuration Templates

### Strong, Stable Peak (ITO, PbI2)
```python
{
    "name": "ITO",
    "center_deg": 30.6,
    "window_deg": 1.0,
    "min_snr": 5.0,                      # High threshold
    "validate_fit_quality": True,
    "use_adaptive_bounds": True,          # ENABLE
    "adaptive_min_consecutive": 5,
    "adaptive_sigma_tolerance": 0.3,     # Tight (30%)
}
```

### Weak, Transient Peak (MeOMBAI)
```python
{
    "name": "MeOMBAI",
    "center_deg": 7.5,
    "window_deg": 1.5,
    "min_snr": 2.5,                      # Low threshold
    "validate_fit_quality": True,         # CRITICAL
    "use_adaptive_bounds": False,         # DISABLED
}
```

### Peak with Transitions (2D Perovskite)
```python
{
    "name": "2D_002",
    "center_deg": 6.0,
    "window_deg": 1.0,
    "key_frames": [0, 200, 400],         # Reset points
    "min_snr": 3.0,
    "validate_fit_quality": True,
    "use_adaptive_bounds": True,
    "adaptive_sigma_tolerance": 0.7,     # Wide (70%)
    "adaptive_min_consecutive": 10,      # Longer warm-up
}
```

## All New Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Validation** |
| `min_snr` | 3.0 | Minimum signal-to-noise ratio |
| `validate_fit_quality` | True | Enable post-fit validation |
| **Adaptive Bounds** |
| `use_adaptive_bounds` | False | Enable adaptive constraints |
| `adaptive_sigma_tolerance` | 0.5 | ±50% change in σ/γ |
| `adaptive_amplitude_tolerance` | 0.5 | ±50% change in amplitude |
| `adaptive_min_consecutive` | 5 | Warm-up frames before activation |
| `fallback_on_validation_failure` | True | Retry with wide bounds |

## New CSV Columns

| Column | Description | Use Case |
|--------|-------------|----------|
| `peak_snr` | Signal-to-noise ratio | Check if threshold appropriate |
| `peak_presence_check` | passed/failed | Track when peak appears |
| `validation_reason` | Why fit rejected | Debug invalid fits |
| `used_adaptive_bounds` | True/False | Monitor adaptive activation |
| `consecutive_good_fits` | Counter | Track stability |

## Status Codes Cheat Sheet

| fit_status | What it means | What to do |
|-----------|---------------|------------|
| `peak_absent` | SNR < threshold | ✅ Normal if peak hasn't formed |
| `fit_invalid` | Failed validation | ⚠️ Check fit_plots, tune params |
| `no_candidate` | No peak found | ⚠️ Check min_prominence |
| `track` | Tracking successfully | ✅ Good! |
| `search` | Initial search | ✅ Normal at start |
| `lost` | Generic failure | ⚠️ Check fit_error column |

## Troubleshooting Flowchart

```
Too many peak_absent?
├─ Lower min_snr (try 2.0-2.5)
├─ Check baseline correction intensity
└─ Verify min_prominence not too high

Too many fit_invalid?
├─ Check validation_reason in CSV
├─ If "Sigma at bound" → widen window_deg
├─ If "Peak too weak" → may be correct!
└─ Temporarily disable: validate_fit_quality=False

Adaptive bounds not activating?
├─ Check: use_adaptive_bounds=True
├─ Check: consecutive_good_fits reaching threshold
└─ Check: Peak tracking successfully (no failures)

Adaptive bounds too tight?
├─ Increase tolerance: 0.7 (70%)
├─ Increase warm-up: 10 frames
└─ Check fallback enabled (default: True)
```

## Typical Workflow

1. **First run**: Validation only (adaptive bounds OFF)
   ```bash
   uv run python peak_tracking_workflow.py --run your_config
   ```

2. **Review outputs**:
   - Check `peak_absent` counts vs expected
   - Review `fit_invalid` cases in fit_plots
   - Check `peak_snr` distribution

3. **Enable adaptive** for stable peaks:
   ```python
   # Edit config
   "use_adaptive_bounds": True  # for ITO, PbI2, etc.
   ```

4. **Re-run and compare**:
   - Monitor `used_adaptive_bounds` column
   - Check `consecutive_good_fits` progression
   - Verify fit quality maintained

5. **Iterate** on thresholds if needed

## Performance Expectations

- **Validation overhead**: < 1ms/frame (negligible)
- **Adaptive speedup**: 10-30% faster (strong peaks)
- **Net effect**: ✅ FASTER overall

## Key Insights from Analysis

✅ **ITO, PbI2**: σ varies only 5-28% → Perfect for adaptive bounds  
❌ **MeOMBAI**: σ varies 100%+, appears/disappears → Keep wide bounds  
⚠️ **34% of MeOMBAI "successes"** were fitting noise → Validation catches these!

## Documentation Files

- **PEAK_VALIDATION_GUIDE.md** - Detailed guide with examples
- **DEVELOPMENT_LOG.md** - Technical implementation details
- **TRACKING_ANALYSIS_FINDINGS.md** - Data analysis results
- **IMPLEMENTATION_SUMMARY.md** - Implementation checklist

## Getting Help

1. Check fit_plots visually
2. Review validation_reason in CSV
3. Try disabling features one at a time
4. Start with conservative defaults (above)
5. Iterate based on your specific peaks
