import pandas as pd
import numpy as np

df = pd.read_csv(r'G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\peak_tracking_workflow_outputs\insitu_0.5M_MeOMBAI_20260526_001029\tracking_results_long.csv')

print('Checking if fitted parameters hit bounds:')
print('=' * 70)

for peak in df['peak_name'].unique():
    peak_data = df[(df['peak_name'] == peak) & (df['fit_success'])].copy()
    
    if len(peak_data) > 0:
        print(f'\n{peak}:')
        
        # Check sigma bounds (typical bounds based on code)
        sigma_min = 0.01  # sigma_floor
        sigma_vals = peak_data['fit_sigma']
        at_floor = (sigma_vals <= sigma_min + 0.001).sum()
        print(f'  Sigma at floor (≤{sigma_min+0.001}): {at_floor}/{len(peak_data)} ({100*at_floor/len(peak_data):.1f}%)')
        print(f'  Sigma range: {sigma_vals.min():.4f} to {sigma_vals.max():.4f}')
        
        # Check gamma bounds
        gamma_vals = peak_data['fit_gamma']
        at_floor_g = (gamma_vals <= sigma_min + 0.001).sum()
        print(f'  Gamma at floor (≤{sigma_min+0.001}): {at_floor_g}/{len(peak_data)} ({100*at_floor_g/len(peak_data):.1f}%)')
        print(f'  Gamma range: {gamma_vals.min():.4f} to {gamma_vals.max():.4f}')
        
        # Check center drift relative to tolerance
        tolerance = peak_data['drift_tolerance_deg'].iloc[0]
        center_vals = peak_data['fit_center_deg']
        expected_center = peak_data['expected_center_deg'].iloc[0]
        drift = (center_vals - expected_center).abs()
        near_limit = (drift > tolerance * 0.8).sum()
        print(f'  Center near drift limit (>80% of {tolerance} deg): {near_limit}/{len(peak_data)} ({100*near_limit/len(peak_data):.1f}%)')
        print(f'  Max drift from expected: {drift.max():.4f} deg')

print('\n' + '=' * 70)
print('\nChecking error patterns:')
for peak in df['peak_name'].unique():
    peak_data = df[df['peak_name'] == peak].copy()
    errors = peak_data[~peak_data['fit_success']]['fit_error'].value_counts()
    
    if len(errors) > 0:
        print(f'\n{peak}: {len(errors)} unique error types')
        for error, count in errors.head(5).items():
            print(f'  "{error}": {count} times')
