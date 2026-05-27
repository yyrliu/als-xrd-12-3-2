# Peak Tracking Analysis: Visual Inspection Results

## Date: May 26, 2026
## Dataset: insitu_0.5M_MeOMBAI

---

## Key Findings from Visual Inspection

### 1. **Strong Peaks (ITO, PbI2): Excellent Tracking** ✅
- **ITO**: Consistent, strong signal throughout all 903 frames
- **PbI2**: Sharp, well-defined peaks with excellent fits
- **Characteristics**:
  - Clear peak presence in all frames
  - Smooth parameter evolution (σ CV: 5-28%)
  - Minimal fitting errors (100% success rate)
  - **These peaks WOULD benefit from adaptive bounds**

### 2. **2D (002): Good Tracking with Few Issues** ✅⚠️
- **97.8% success rate** (883/903 frames)
- Most fits are excellent (sharp peaks, good quality)
- **Early frames (idx 2-10)**: Peak is ABSENT or extremely weak
  - Sigma maxes out at 0.6 trying to fit noise
  - These are genuine "no peak" situations
- **Later frames**: Strong, well-defined peak that tracks smoothly

### 3. **MeOMBAI: Major Issues - Fitting Non-Existent Peaks** ❌

#### Visual Evidence:
- **idx 4**: Real peak present, noisy but fittable (Amp: 0.0195, σ: 0.017)
- **idx 100**: **NO PEAK EXISTS** - flat background, but fit reports:
  - Amplitude: 0.0000 (essentially zero)
  - Sigma: 0.490 (maxed out)
  - **Marked as "successful fit"** ❌
- **idx 902**: **NO PEAK EXISTS** - fitting tail of different feature
  - Amplitude: 0.2628
  - Sigma: 0.500 (maxed out)
  - Center: 6.86 deg (drifted 0.63 deg from expected 7.5)
  - **Marked as "successful fit"** ❌

#### Quantitative Analysis:
- **210 out of 614 "successful" fits** (34%) have σ > 0.4
  - These are hitting the upper bound → fitting very broad features (NOT the target peak)
  - Amplitudes ≈ 0.000001 (essentially zero) yet report net_area 7-12
- **Physical interpretation**: The MeOMBAI peak genuinely appears and disappears during the experiment
  - This is REAL sample behavior, not a tracking failure
  - The algorithm cannot distinguish "peak absent" from "peak weak"

---

## Critical Problem: False Positives

The current workflow has **no validation to reject fits of non-existent peaks**:

1. ✅ Optimizer converges → "success"
2. ❌ No check: "Is this actually a peak?"
3. ❌ No check: "Is amplitude above noise floor?"
4. ❌ No check: "Did sigma hit bounds?" (indicates poor fit)

**Result**: ~34% of MeOMBAI "successful" fits are fitting noise/background.

---

## Why This Matters for Adaptive Bounds

### If we implement adaptive bounds blindly:

**Scenario 1: Peak disappears (frame 100)**
1. Frame 99: Real peak, σ = 0.02
2. Frame 100: No peak present
3. Adaptive bounds: σ constrained to 0.014-0.026 (30% tolerance)
4. **Result**: Fit fails completely OR forces unrealistic narrow fit to noise
5. Tracking lost permanently ❌

**Scenario 2: Peak reappears with different width**
1. Frames 100-200: No peak, tracking lost
2. Frame 201: Peak returns but broader (σ = 0.08)
3. Previous σ was 0.02, adaptive bounds: 0.014-0.026
4. **Result**: Cannot fit the real peak because bounds too tight ❌

---

## Revised Recommendations

### A. Add Fit Quality Validation (CRITICAL - Do This First)

Add these checks to `fit_frame()` or post-processing:

```python
def validate_fit_quality(fit_res, peak_spec, candidate_center):
    """Return True if fit represents a real peak, False if fitting noise."""
    best = fit_res.best_values
    
    # 1. Check if sigma hit bounds (indicates poor fit)
    sigma = best.get("peak_sigma", 0)
    sigma_floor = max(0.01, peak_spec.window_deg / 50)
    sigma_ceiling = max(peak_spec.window_deg, peak_spec.drift_tolerance_deg * 4)
    if sigma <= sigma_floor * 1.1 or sigma >= sigma_ceiling * 0.9:
        return False, "Sigma at bound"
    
    # 2. Check if amplitude is above noise floor
    amplitude = best.get("peak_amplitude", 0)
    # Estimate noise from residuals
    noise_level = np.std(fit_res.residual)
    if amplitude < noise_level * 3:  # SNR < 3
        return False, "Amplitude below noise"
    
    # 3. Check peak height (not just area)
    # For Voigt: height ≈ amplitude / (π * gamma)
    gamma = best.get("peak_gamma", 1)
    peak_height = amplitude / (np.pi * gamma) if gamma > 0 else 0
    data_max = np.max(fit_res.data)
    if peak_height < data_max * 0.1:  # Peak < 10% of max signal
        return False, "Peak too weak"
    
    # 4. Check fit quality
    if fit_res.redchi > 10:  # Poor reduced chi-square
        return False, "Poor fit quality"
    
    return True, None
```

### B. Implement Adaptive Bounds SELECTIVELY

```python
@dataclass
class PeakSpec:
    # ... existing fields ...
    use_adaptive_bounds: bool = False  # Enable per-peak
    adaptive_sigma_tolerance: float = 0.5  # 50% change allowed (not 30%)
    adaptive_min_frames: int = 10  # Need N consecutive good fits before tightening
    fallback_on_failure: bool = True  # Retry with wide bounds if adaptive fails
```

**Enable adaptive bounds ONLY for:**
- ✅ ITO (100% success, stable)
- ✅ PbI2 (100% success, stable)
- ❌ MeOMBAI (use wide bounds, it appears/disappears)
- ⚠️ 2D (002) (maybe after initial detection)

### C. Add Peak Presence Detection

Before fitting, check if a peak is actually present:

```python
def has_candidate_peak(window_data, min_snr=3.0):
    """Check if window contains a real peak above noise."""
    y = window_data.values
    baseline = np.percentile(y, 10)  # Bottom 10% is noise
    peak_max = np.max(y)
    noise = np.std(y[y < np.percentile(y, 25)])
    
    snr = (peak_max - baseline) / noise if noise > 0 else 0
    return snr > min_snr
```

### D. Separate "Peak Absent" from "Fit Failed"

Modify tracking state:
```python
@dataclass
class TrackingState:
    last_center: float | None = None
    last_sigma: float | None = None
    last_amplitude: float | None = None
    tracking_active: bool = False
    lost_count: int = 0
    peak_present: bool = True  # NEW: distinguish absence from failure
```

---

## Summary

**Your suspicion was 100% correct**: Many "successful" fits are fitting non-existent peaks.

**Priority Actions**:
1. **Add fit validation** (reject σ at bounds, amplitude ≈ 0)
2. **Add peak presence detection** (SNR check before fitting)
3. **Only then** consider adaptive bounds for strong, stable peaks
4. **Keep wide bounds** for peaks that appear/disappear

**Adaptive bounds are helpful BUT ONLY after we can reliably distinguish real peaks from noise.**
