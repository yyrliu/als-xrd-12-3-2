import pandas as pd
import numpy as np

df = pd.read_csv(r'G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\peak_tracking_workflow_outputs\insitu_0.5M_MeOMBAI_20260526_001029\tracking_results_long.csv')
df_success = df[df['fit_success']].copy()

print('Frame-to-frame parameter changes (successful fits only):')
print('=' * 70)

for peak in df_success['peak_name'].unique():
    peak_data = df_success[df_success['peak_name'] == peak].sort_values('frame_index')
    
    if len(peak_data) > 1:
        center_diff = peak_data['fit_center_deg'].diff().abs()
        sigma_diff = peak_data['fit_sigma'].diff().abs()
        amp_diff = (peak_data['fit_amplitude'].diff().abs() / peak_data['fit_amplitude'].shift(1) * 100)
        
        print(f'\n{peak}:')
        print(f'  Center shift: mean={center_diff.mean():.5f} deg, max={center_diff.max():.5f} deg, std={center_diff.std():.5f}')
        print(f'  Sigma change: mean={sigma_diff.mean():.5f}, max={sigma_diff.max():.5f}, std={sigma_diff.std():.5f}')
        print(f'  Amplitude change: mean={amp_diff.mean():.1f}%, max={amp_diff.max():.1f}%, std={amp_diff.std():.1f}%')
        
        # Check for large jumps (outliers)
        center_q95 = center_diff.quantile(0.95)
        sigma_q95 = sigma_diff.quantile(0.95)
        large_center_jumps = (center_diff > center_q95 * 2).sum()
        large_sigma_jumps = (sigma_diff > sigma_q95 * 2).sum()
        
        print(f'  Large jumps: {large_center_jumps} center outliers, {large_sigma_jumps} sigma outliers')

print('\n' + '=' * 70)
print('\nLooking at consecutive failed fits:')
for peak in df['peak_name'].unique():
    peak_data = df[df['peak_name'] == peak].sort_values('frame_index')
    
    # Find consecutive failures
    peak_data['failed'] = ~peak_data['fit_success']
    peak_data['fail_group'] = (peak_data['failed'] != peak_data['failed'].shift()).cumsum()
    
    fail_runs = peak_data[peak_data['failed']].groupby('fail_group').size()
    if len(fail_runs) > 0:
        print(f'\n{peak}: {len(fail_runs)} failure episodes, max consecutive={fail_runs.max()}, total failed={peak_data["failed"].sum()}')
