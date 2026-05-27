"""Check validation reasons for rejected frames"""
import pandas as pd
from pathlib import Path

valid_dir = Path(r'G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\peak_tracking_workflow_outputs_snr\insitu_0.5M_MeOMBAI_20260526_123252')
valid = pd.read_csv(valid_dir / 'tracking_results_long.csv')

meombai = valid[valid['peak_name'] == 'MeOMBAI']

# Check validation reasons for those high-sigma frames that were rejected
frames = [902, 897, 895, 888, 886, 879, 878, 876, 865, 854]

print('Validation details for rejected frames:')
print('='*80)
for idx in frames:
    row = meombai[meombai['frame_index'] == idx].iloc[0]
    print(f"Frame {idx}:")
    print(f"  status: {row['fit_status']}")
    print(f"  SNR: {row['peak_snr']:.2f}")
    if 'validation_reason' in row and pd.notna(row['validation_reason']):
        print(f"  validation_reason: {row['validation_reason']}")
    if 'peak_presence_check' in row and pd.notna(row['peak_presence_check']):
        print(f"  presence_check: {row['peak_presence_check']}")
    print()

# Check if validation_reason column exists and what values it has
if 'validation_reason' in meombai.columns:
    print('\nAll validation reasons in dataset:')
    print(meombai['validation_reason'].value_counts())
    print()
    
# Check lost vs peak_absent vs fit_invalid breakdown
print('\nStatus breakdown for MeOMBAI:')
print(meombai['fit_status'].value_counts())
