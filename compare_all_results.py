"""
Compare original vs validated tracking results across ALL datasets.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Paths
orig_base = Path(r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\peak_tracking_workflow_outputs")
valid_base = Path(r"G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\peak_tracking_workflow_outputs_snr")

# Find matching runs - extract base name without timestamp (format: name_YYYYMMDD_HHMMSS)
def extract_run_name(dir_name):
    """Extract base run name by removing _YYYYMMDD_HHMMSS suffix"""
    parts = dir_name.rsplit('_', 2)  # Split from right, max 2 splits
    if len(parts) >= 3 and parts[-2].isdigit() and len(parts[-2]) == 8:
        # Has timestamp format
        return '_'.join(parts[:-2])
    return dir_name

orig_dirs = {extract_run_name(d.name): d for d in orig_base.iterdir() if d.is_dir()}
valid_dirs = {extract_run_name(d.name): d for d in valid_base.iterdir() if d.is_dir()}

# Find common runs
common_runs = sorted(set(orig_dirs.keys()) & set(valid_dirs.keys()))

print('='*100)
print('COMPREHENSIVE COMPARISON: All 10 Datasets')
print('='*100)
print(f'\nFound {len(common_runs)} matching runs\n')

# Aggregate statistics
agg_stats = defaultdict(lambda: defaultdict(list))

for run_name in common_runs:
    print(f'\n{"="*100}')
    print(f'{run_name.upper()}')
    print('='*100)
    
    # Load CSVs
    orig_csv = orig_dirs[run_name] / "tracking_results_long.csv"
    valid_csv = valid_dirs[run_name] / "tracking_results_long.csv"
    
    if not orig_csv.exists() or not valid_csv.exists():
        print(f'  ⚠️  Missing CSV files, skipping...')
        continue
    
    orig = pd.read_csv(orig_csv)
    valid = pd.read_csv(valid_csv)
    
    # Get unique peaks
    peaks = sorted(orig['peak_name'].unique())
    
    for peak in peaks:
        orig_peak = orig[orig['peak_name'] == peak]
        valid_peak = valid[valid['peak_name'] == peak]
        
        print(f'\n  {peak}:')
        
        # Success rates
        orig_success = orig_peak['fit_success'].sum()
        orig_total = len(orig_peak)
        valid_success = valid_peak['fit_success'].sum()
        valid_total = len(valid_peak)
        
        orig_rate = orig_success / orig_total * 100
        valid_rate = valid_success / valid_total * 100
        
        print(f'    Success: {orig_success}/{orig_total} ({orig_rate:.1f}%) → {valid_success}/{valid_total} ({valid_rate:.1f}%)')
        
        # Aggregate
        agg_stats[peak]['orig_rate'].append(orig_rate)
        agg_stats[peak]['valid_rate'].append(valid_rate)
        agg_stats[peak]['orig_success'].append(orig_success)
        agg_stats[peak]['valid_success'].append(valid_success)
        
        # Sigma analysis for successful fits
        orig_success_data = orig_peak[orig_peak['fit_success']]
        valid_success_data = valid_peak[valid_peak['fit_success']]
        
        if len(orig_success_data) > 0:
            orig_sigma_mean = orig_success_data['fit_sigma'].mean()
            orig_sigma_std = orig_success_data['fit_sigma'].std()
            orig_high_sigma = (orig_success_data['fit_sigma'] > 0.4).sum()
            orig_high_sigma_pct = orig_high_sigma / len(orig_success_data) * 100
            
            print(f'    Sigma (orig): μ={orig_sigma_mean:.4f}, σ={orig_sigma_std:.4f}, >0.4: {orig_high_sigma} ({orig_high_sigma_pct:.1f}%)')
            
            agg_stats[peak]['orig_sigma_mean'].append(orig_sigma_mean)
            agg_stats[peak]['orig_sigma_std'].append(orig_sigma_std)
            agg_stats[peak]['orig_high_sigma_pct'].append(orig_high_sigma_pct)
        
        if len(valid_success_data) > 0:
            valid_sigma_mean = valid_success_data['fit_sigma'].mean()
            valid_sigma_std = valid_success_data['fit_sigma'].std()
            valid_high_sigma = (valid_success_data['fit_sigma'] > 0.4).sum()
            valid_high_sigma_pct = valid_high_sigma / len(valid_success_data) * 100 if len(valid_success_data) > 0 else 0
            
            print(f'    Sigma (valid): μ={valid_sigma_mean:.4f}, σ={valid_sigma_std:.4f}, >0.4: {valid_high_sigma} ({valid_high_sigma_pct:.1f}%)')
            
            agg_stats[peak]['valid_sigma_mean'].append(valid_sigma_mean)
            agg_stats[peak]['valid_sigma_std'].append(valid_sigma_std)
            agg_stats[peak]['valid_high_sigma_pct'].append(valid_high_sigma_pct)
            
            # Adaptive bounds usage
            if 'used_adaptive_bounds' in valid_peak.columns:
                adaptive_count = valid_peak['used_adaptive_bounds'].sum()
                adaptive_pct = adaptive_count / len(valid_peak) * 100
                print(f'    Adaptive: {adaptive_count}/{len(valid_peak)} ({adaptive_pct:.1f}%)')
                agg_stats[peak]['adaptive_pct'].append(adaptive_pct)

# Summary statistics
print(f'\n\n{"="*100}')
print('AGGREGATE STATISTICS ACROSS ALL DATASETS')
print('='*100)

for peak in sorted(agg_stats.keys()):
    print(f'\n{peak}:')
    print(f'  {"Metric":<30} {"Original":<20} {"Validated":<20} {"Change"}')
    print(f'  {"-"*90}')
    
    # Success rate
    orig_rates = agg_stats[peak]['orig_rate']
    valid_rates = agg_stats[peak]['valid_rate']
    if orig_rates and valid_rates:
        print(f'  {"Success Rate (avg)":<30} {np.mean(orig_rates):>6.1f}% ± {np.std(orig_rates):.1f}   {np.mean(valid_rates):>6.1f}% ± {np.std(valid_rates):.1f}   {np.mean(valid_rates) - np.mean(orig_rates):+6.1f}%')
    
    # High sigma percentage
    orig_high = agg_stats[peak]['orig_high_sigma_pct']
    valid_high = agg_stats[peak]['valid_high_sigma_pct']
    if orig_high and valid_high:
        print(f'  {"False Positive Rate (σ>0.4)":<30} {np.mean(orig_high):>6.1f}% ± {np.std(orig_high):.1f}   {np.mean(valid_high):>6.1f}% ± {np.std(valid_high):.1f}   {np.mean(valid_high) - np.mean(orig_high):+6.1f}%')
    
    # Sigma stability
    orig_sigma = agg_stats[peak]['orig_sigma_std']
    valid_sigma = agg_stats[peak]['valid_sigma_std']
    if orig_sigma and valid_sigma:
        print(f'  {"Sigma Std Dev (avg)":<30} {np.mean(orig_sigma):>10.4f}       {np.mean(valid_sigma):>10.4f}       {(1 - np.mean(valid_sigma)/np.mean(orig_sigma))*100:+5.1f}% better')
    
    # Adaptive bounds
    adaptive = agg_stats[peak]['adaptive_pct']
    if adaptive:
        print(f'  {"Adaptive Bounds Usage":<30} {"N/A":<20} {np.mean(adaptive):>6.1f}% ± {np.std(adaptive):.1f}')

print('\n' + '='*100)
print('SUMMARY')
print('='*100)
print('''
Key Findings:
  1. Check if false positive reduction is consistent across all datasets
  2. Verify adaptive bounds activation for stable peaks (ITO, PbI2)
  3. Confirm sigma stability improvements
  4. Identify any dataset-specific issues

Next: Review the numbers above to ensure validation improvements are universal.
''')
