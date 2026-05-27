"""
Quick visual check: Compare a few fit plots between original and validated.
"""
import pandas as pd
from pathlib import Path

# Paths
orig_base = Path(r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\peak_tracking_workflow_outputs")
valid_base = Path(r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\peak_tracking_workflow_outputs_snr")

# Load results
orig_dir = orig_base / "insitu_0.5M_MeOMBAI_20260526_001029"
valid_dir = valid_base / "insitu_0.5M_MeOMBAI_20260526_123252"

orig = pd.read_csv(orig_dir / "tracking_results_long.csv")
valid = pd.read_csv(valid_dir / "tracking_results_long.csv")

print("="*80)
print("VISUAL COMPARISON GUIDE")
print("="*80)
print()

# MeOMBAI: Find frames with high sigma in original but rejected in validated
meombai_orig = orig[orig['peak_name'] == 'MeOMBAI']
meombai_valid = valid[valid['peak_name'] == 'MeOMBAI']

high_sigma_orig = meombai_orig[(meombai_orig['fit_success']) & (meombai_orig['fit_sigma'] > 0.4)]

print("MeOMBAI - Examples of FALSE POSITIVES in original (removed in validated):")
print("-"*80)
print("\nFrames where original claimed success but was likely fitting noise:")
print("(These should be rejected in validated run)")
print()

sample_frames = high_sigma_orig.head(10)['frame_index'].values
for idx in sample_frames:
    orig_row = meombai_orig[meombai_orig['frame_index'] == idx].iloc[0]
    valid_row = meombai_valid[meombai_valid['frame_index'] == idx].iloc[0]
    
    print(f"Frame {idx}:")
    print(f"  Original: fit_success={orig_row['fit_success']}, sigma={orig_row['fit_sigma']:.4f}, status={orig_row['fit_status']}")
    print(f"  Validated: fit_success={valid_row['fit_success']}, sigma={valid_row.get('fit_sigma', 'N/A')}, status={valid_row['fit_status']}")
    if 'peak_snr' in valid_row and pd.notna(valid_row['peak_snr']):
        print(f"  SNR: {valid_row['peak_snr']:.2f}")
    print(f"  → Check plots: fit_MeOMBAI_idx{idx}.png in both fit_plots folders")
    print()

# Check where fit plots exist
orig_plots = orig_dir / "fit_plots"
valid_plots = valid_dir / "fit_plots"

print("\n" + "="*80)
print("TO VISUALLY VERIFY:")
print("="*80)
print()
print("1. Navigate to original fit_plots:")
print(f"   {orig_plots}")
print()
print("2. Navigate to validated fit_plots:")  
print(f"   {valid_plots}")
print()
print("3. Compare the same frame indices listed above")
print()
print("EXPECTED OBSERVATIONS:")
print("  ✓ Original plots show wide, flat peaks fitting noise/background")
print("  ✓ Validated plots either:")
print("    - Don't exist (frame marked as peak_absent/lost)")
print("    - Show tighter, more realistic peaks (if peak was real)")
print()

# ITO: Check adaptive bounds activation
ito_valid = valid[valid['peak_name'] == 'ITO']
if 'used_adaptive_bounds' in ito_valid.columns:
    first_adaptive = ito_valid[ito_valid['used_adaptive_bounds']].iloc[0]['frame_index']
    print("\n" + "="*80)
    print("ITO - Adaptive Bounds Activation:")
    print("="*80)
    print(f"\nFirst frame with adaptive bounds: {first_adaptive}")
    print(f"Check ITO fit plots around frame {first_adaptive}:")
    print("  - Before: Wide parameter bounds")
    print("  - After: Tight adaptive bounds (should look similar but faster convergence)")
    print()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print()
print("Key visual checks:")
print("  1. MeOMBAI bad frames in original → absent/rejected in validated")
print("  2. ITO adaptive bounds → parameters stay stable, fits look identical")
print("  3. All validated plots should show physically reasonable peaks")
print()
