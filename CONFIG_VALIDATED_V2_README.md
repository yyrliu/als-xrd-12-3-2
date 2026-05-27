# Validated Config V2 - Corrected PbI2 Bounds

**Version**: 2.0  
**Date**: 2026-05-26  
**Status**: Ready for testing  
**Output Directory**: `peak_tracking_workflow_outputs_snr_v2`

---

## What's Fixed in V2

### 🚨 Critical Fix: PbI2 False Positives

**The Problem** (discovered in comprehensive analysis of v1):
- V1 was **accepting bad PbI2 fits** with σ = 0.45-0.48
- Two MBAI datasets severely affected:
  * `insitu_0.5M_MBAI`: 62.3% of validated fits had σ > 0.4 (worse than original!)
  * `aging_100c_120s_anneal_MBAI`: 38.4% of validated fits had σ > 0.4
- Root cause: `sigma_max = 0.5` is **too lenient** for narrow peaks

**The Solution** (v2):
```python
# V1 (broken):
PbI2: sigma_max = 0.5  # Accepts σ up to 0.475 via validation
      → Accepts fits 4-9× wider than physical expectations

# V2 (fixed):
PbI2: sigma_max = 0.15  # Physical upper limit for PbI2
      → Rejects σ > 0.14, catches bad fits that v1 missed
```

---

## Peak-Specific Bounds in V2

| Peak Type | sigma_min | sigma_max | Rationale |
|-----------|-----------|-----------|-----------|
| **PbI2** | 0.02 | **0.15** | Narrow crystalline peak (typical σ: 0.05-0.10) |
| **ITO** | 0.05 | **0.25** | Substrate peak, narrow and stable (typical σ: 0.14-0.18) |
| **2D (002)** | 0.01 | **0.30** | Perovskite, can broaden during transitions |
| **1D (002)** | 0.01 | **0.30** | Similar to 2D |
| **MeOMBAI** | 0.01 | **0.50** | Intermittent weak peak, needs flexibility |
| **ClMBAI** | 0.01 | **0.50** | Intermittent weak peak, needs flexibility |

### Physical Reasoning

**Narrow peaks (PbI2, ITO)**:
- Good crystallinity → narrow diffraction peaks
- Stable chemical environment → consistent σ
- **Tight bounds prevent false positives**

**Perovskite peaks (2D/1D 002)**:
- Can broaden during phase transitions
- Moderate bounds (0.3) allow physical changes
- Still reject extreme outliers (σ > 0.3)

**Organic peaks (MeOMBAI, ClMBAI)**:
- Weak, intermittent signals
- More variability due to lower crystallinity
- **Wide bounds necessary** to catch real peaks

---

## Expected Improvements Over V1

### PbI2 in MBAI Datasets

| Dataset | V1 High σ (>0.4) | V2 Expected | Improvement |
|---------|-----------------|-------------|-------------|
| `insitu_0.5M_MBAI` | 62.3% ⚠️ | **< 1%** ✅ | **Fix ~430 false positives** |
| `aging_100c_120s_MBAI` | 38.4% ⚠️ | **< 1%** ✅ | **Fix ~200 false positives** |

With `sigma_max = 0.15`, validation will properly reject these bad fits.

### Other Peaks (Maintained)

| Peak Type | V1 Performance | V2 Expected |
|-----------|----------------|-------------|
| **MeOMBAI** | Perfect (0% false positives) | **Maintained** ✅ |
| **ClMBAI** | Excellent (0.6% false positives) | **Maintained** ✅ |
| **ITO** | Good (99.9% success) | **Maintained** ✅ |
| **2D (002)** | Good (0% false positives) | **Maintained** ✅ |

---

## Usage

### Test on Problem Datasets First

Start with the two datasets that had issues in v1:

```bash
# Test PbI2 fix on MBAI dataset with worst false positives
uv run python peak_tracking_workflow.py \
    --config peak_tracking_config_validated_v2.py \
    --run insitu_0.5M_MBAI

# Check results
# Expected: PbI2 false positive rate drops from 62.3% to <1%
```

### Compare V1 vs V2

```python
# In Python
import pandas as pd

# Load both results
v1 = pd.read_csv("peak_tracking_workflow_outputs_snr/insitu_0.5M_MBAI_.../tracking_results_long.csv")
v2 = pd.read_csv("peak_tracking_workflow_outputs_snr_v2/insitu_0.5M_MBAI_.../tracking_results_long.csv")

# Check PbI2 sigma distribution
pbi2_v1 = v1[(v1['peak_name'] == 'PbI2') & (v1['fit_success'])]
pbi2_v2 = v2[(v2['peak_name'] == 'PbI2') & (v2['fit_success'])]

print(f"V1: {(pbi2_v1['fit_sigma'] > 0.4).sum()} / {len(pbi2_v1)} fits with σ>0.4")
print(f"V2: {(pbi2_v2['fit_sigma'] > 0.4).sum()} / {len(pbi2_v2)} fits with σ>0.4")
# Expected V2: 0 or very few
```

### Full Batch Run

Once verified on problem datasets:

```bash
# Run all datasets with corrected config
uv run python peak_tracking_workflow.py \
    --config peak_tracking_config_validated_v2.py \
    --batch-all
```

---

## Validation Logic

V2 uses the same two-stage validation as V1, but with corrected bounds:

### Stage 1: Pre-Fit SNR Check
```python
if SNR < min_snr:
    return "peak_absent"  # Don't waste time fitting noise
```

### Stage 2: Post-Fit Quality Validation (CORRECTED)
```python
# Check sigma against PEAK-SPECIFIC bounds
if sigma >= 0.95 * sigma_max:  # Now uses peak-specific sigma_max
    return "fit_invalid"
    
# For PbI2: sigma_max = 0.15, so rejects σ ≥ 0.1425
# For V1:   sigma_max = 0.50, so accepts σ up to 0.475 (TOO LENIENT!)

if amplitude < 1e-4:
    return "fit_invalid"
if peak_height < 0.05 * data_max:
    return "fit_invalid"
if redchi > 20:
    return "fit_invalid"
```

---

## What to Check After Running V2

### 1. PbI2 False Positive Rate

```python
# For each dataset with PbI2
pbi2 = df[(df['peak_name'] == 'PbI2') & (df['fit_success'])]

high_sigma_count = (pbi2['fit_sigma'] > 0.15).sum()
high_sigma_pct = high_sigma_count / len(pbi2) * 100

print(f"PbI2 high sigma (>0.15): {high_sigma_count}/{len(pbi2)} ({high_sigma_pct:.1f}%)")
# Expected: 0 or very close to 0
```

### 2. Validation Reason Distribution

```python
# Check why fits are being rejected
df['validation_reason'].value_counts()

# Should see:
# - "Valid fit": most successful fits
# - "Sigma at upper bound: X.XX ≥ 0.1425": PbI2 rejections (if any)
# - Very few "Sigma at upper bound: X.XX ≥ 0.4750": only for organic peaks
```

### 3. Success Rate Changes

Expected changes from V1 → V2:

| Peak | Expected Change |
|------|----------------|
| PbI2 (MBAI datasets) | **Lower success rate** (rejecting bad fits) |
| PbI2 (other datasets) | Similar (already good) |
| All other peaks | No change (bounds unchanged or relaxed) |

---

## Troubleshooting

### Issue: PbI2 success rate drops too much

**Possible causes**:
1. σ_max = 0.15 is too restrictive for your data
2. Real PbI2 peaks are broader than typical

**Solution**:
```python
# In peak_tracking_config_validated_v2.py
PbI2: sigma_max = 0.20  # Increase if needed
```

### Issue: Still seeing high sigma PbI2 fits

**Possible causes**:
1. Fits converging just below threshold (σ = 0.14)
2. Need to check gamma bounds too

**Solution**:
```python
# Add gamma bounds
PbI2: {
    "sigma_max": 0.15,
    "gamma_max": 0.10,  # Also constrain Lorentzian width
}
```

---

## Summary of Changes

| Item | V1 (Broken) | V2 (Fixed) |
|------|------------|-----------|
| **PbI2 sigma_max** | 0.5 (implicit) | **0.15** ← KEY FIX |
| **ITO sigma_max** | 0.5 (implicit) | **0.25** |
| **2D/1D (002) sigma_max** | 0.5 (implicit) | **0.30** |
| **Organic peaks** | 0.5 (implicit) | **0.50** (explicit) |
| **Output directory** | `..._snr` | `..._snr_v2` |
| **Expected PbI2 false positives** | 10-60% ⚠️ | **<1%** ✅ |

---

## Next Steps

1. ✅ Created corrected config (this file)
2. ⏳ **Test on `insitu_0.5M_MBAI`** (worst case, 62.3% false positives in v1)
3. ⏳ **Verify fix**: PbI2 false positive rate should drop to <1%
4. ⏳ **Test on `aging_100c_120s_MBAI`** (38.4% false positives in v1)
5. ⏳ Run full batch if tests pass
6. ⏳ Update COMPREHENSIVE_RESULTS_ANALYSIS.md with v2 results

---

## Files

- **Config**: [peak_tracking_config_validated_v2.py](peak_tracking_config_validated_v2.py)
- **This README**: [CONFIG_VALIDATED_V2_README.md](CONFIG_VALIDATED_V2_README.md)
- **V1 Analysis**: [COMPREHENSIVE_RESULTS_ANALYSIS.md](COMPREHENSIVE_RESULTS_ANALYSIS.md) (shows the problem)
- **Original validated config**: [peak_tracking_config_validated.py](peak_tracking_config_validated.py) (deprecated, use v2)

**Status**: Ready for testing ✅
