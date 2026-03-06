import numpy as np
import pandas as pd
import xarray as xr
import logging
import pathlib
import matplotlib.pyplot as plt
from scipy.signal import peak_widths, find_peaks
from scipy.optimize import curve_fit, brentq
from scipy.special import wofz
from pybaselines import Baseline

# Configure logger
logger = logging.getLogger(__name__)

def baseline_correction(da: xr.DataArray, sample=None, method=None, **kwargs):
    """Perform baseline correction on an xarray DataArray using specified method.
    Parameters:
        da (xr.DataArray): Input data array to be baseline corrected.
        sample (str, optional): Type of sample to determine default method and parameters.
        method (str, optional): Baseline correction method to use.
    Returns:
        corrected_da (xr.DataArray): Baseline corrected data array.
        baseline_da (xr.DataArray): Calculated baseline data array.
    """

    if sample is not None and method is not None:
        raise ValueError("Either method or sample type must be specified, not both.")
    
    if method is None:
        if sample == "PXRD":
            method = "asls"
            kwargs = {"lam": 5e3}
        elif sample == "UV-Vis":
            method = "aspls"
            kwargs = {"lam": 7e5}
        else:
            raise ValueError("Unknown sample type. Please specify method directly.")
        
    baseline_finder = Baseline()
    func = getattr(baseline_finder, method)
    baseline, _ = func(da.values, **kwargs)
    corrected = da.values - baseline

    corrected_da = xr.DataArray(corrected, coords=da.coords, dims=da.dims)
    baseline_da = xr.DataArray(baseline, coords=da.coords, dims=da.dims)
    return corrected_da, baseline_da

def voigt_profile_func(x, amp, cen, sigma, gamma, slope, offset):
    """
    Voigt profile function with linear background.
    """
    z = ((x - cen) + 1j * gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi)) + slope * (x-cen) + offset

def find_peaks_in_window(da: xr.DataArray, x: str, target=None, window_size=None, **find_peaks_kwargs):
    """
    Find peaks in an xarray DataArray, optionally restricting to a window around a target.
    """
    if x not in da.dims and x not in da.coords:
        raise ValueError(f"{x} must be a dimension or coordinate in the DataArray.")

    if target is not None and window_size is not None:
        target_da = da.sel(**{x: slice(target - window_size / 2, target + window_size / 2)})
        da = target_da 
    
    peaks_indices, properties = find_peaks(da.values, **find_peaks_kwargs)
    half_widths = peak_widths(da.values, peaks_indices, rel_height=0.5)

    # convert indices back to original x values (wavelength or twoTheta)
    peaks_x = da[x].values[peaks_indices]
    # get intensities at peak positions
    peaks_intensities = da.values[peaks_indices]
    # convert from indices to x units
    half_widths = half_widths[0] * np.abs(da[x].values[1] - da[x].values[0]) 
    # add half widths to properties
    properties['half_widths'] = half_widths

    # Convert properties from dict of arrays to array of dicts
    properties = [dict(zip(properties,t)) for t in zip(*properties.values())]

    return peaks_x, peaks_intensities, properties

def voigt_fit(da: xr.DataArray, x: str, initial_guess: (float|list), window_size: float, no_slope=True, bounds=None):
    """
    Fit a Voigt profile to a peak in the DataArray.
    initial_guess could be peak_pos, or a list of parameters [amp, cen, sigma, gamma, slope, offset] (slope and offset optional if no_slope=True).
    bounds: tuple of (min, max) arrays for curve_fit. If None, defaults are used.
    """
    # set bounds for parameters: amp > 0, sigma > 0, gamma > 0
    # amp, cen, sigma, gamma, slope, offset

    if isinstance(initial_guess, float):
        peak_pos = initial_guess
        peak_intensity = da.sel(**{x: peak_pos}, method='nearest').values
        
        # Handle case where peak intensity itself is NaN/Inf
        if not np.isfinite(peak_intensity):
            peak_intensity = 0.001 # safe fallback

        # guess: amp, cen, sigma, gamma, slope, offset, offset is set later after fit_da is generated
        initial_guess = [np.clip(peak_intensity, 1e-6, None), peak_pos, 0.1, 0.1, 0, None]
    elif isinstance(initial_guess, list) and len(initial_guess) >= 4:
        # If passed a list, we assume it's a good guess, but we need peak_pos for window/bounds defaults
        peak_pos = initial_guess[1]
    else:
        raise ValueError("initial_guess must be either a float (peak position) or a list of parameters with at least 4 elements (amp, cen, sigma, gamma).")

    if bounds is None:
        if no_slope:
            bounds = (
                [0, peak_pos - 0.5, 0, 0, -1e-9, -np.inf], 
                [np.inf, peak_pos + 0.5, np.inf, np.inf, 1e-9, np.inf]
            )
        else:
            bounds = (
                [0, peak_pos - 0.5, 0, 0, -np.inf, -np.inf], 
                [np.inf, peak_pos + 0.5, np.inf, np.inf, np.inf, np.inf]
            )

    fit_da = da.sel(**{x: slice(peak_pos - window_size / 2, peak_pos + window_size / 2)})
    
    # Ensure we use only finite data for fitting
    # xarray's dropna usually handles NaNs, but we also want to exclude infs
    x_vals = fit_da[x].values
    y_vals = fit_da.values
    
    mask = np.isfinite(y_vals)
    x_data = x_vals[mask]
    y_data = y_vals[mask]
    
    if len(y_data) < 4: # Need at least as many points as parameters (roughly)
         raise RuntimeError(f"Not enough valid data points in fitting window for peak at {peak_pos:.2f}° (found {len(y_data)}).")

    # If offset is not provided, set it to the minimum of the y_data in the fitting window
    if initial_guess[5] is None:
        initial_guess[5] = np.min(y_data)

    try:
        popt, pcov = curve_fit(voigt_profile_func, x_data, y_data, p0=initial_guess, bounds=bounds)
    except RuntimeError as e:
        logger.error(f"Voigt fit failed for peak at {peak_pos:.2f}: {e}")
        raise

    fitted_curve = voigt_profile_func(da[x].values, *popt)
    fitted_da = xr.DataArray(fitted_curve, coords=da.coords, dims=da.dims)

    return fitted_da, popt

def calculate_fwhm(sigma, gamma):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a Voigt profile given its Gaussian (sigma) and Lorentzian (gamma) parameters.
    """
    fwhm_L = 2 * gamma
    fwhm_G = 2.3548 * sigma
    fwhm = 0.5346 * fwhm_L + np.sqrt(0.2166 * fwhm_L**2 + fwhm_G**2)
    return fwhm

def get_confidence_bounds(amp, cen, sigma, gamma, confidence=0.95, cut_off=2):
    """
    Calculate the x-values that bound `confidence` (fraction) of the Voigt profile area.
    """
    # Estimate FWHM to define a sufficient grid
    fwhm = calculate_fwhm(sigma, gamma)
    
    # Create a grid wide enough to capture the tails (20 * FWHM)
    window = 5 * fwhm
    if window == 0: window = 1.0 
    
    x_grid = np.linspace(cen - window, cen + window, 5000)
    y_grid = voigt_profile_func(x_grid, amp, cen, sigma, gamma, 0, 0)
    
    cum_area = np.cumsum(y_grid) * (x_grid[1] - x_grid[0])
    total_area = cum_area[-1]
    if total_area == 0:
        return cen - 0.1, cen + 0.1
    
    alpha = 1.0 - confidence
    lower_q = alpha / 2.0
    upper_q = 1.0 - (alpha / 2.0)
    
    norm_cum = cum_area / total_area
    idx_low = np.searchsorted(norm_cum, lower_q)
    idx_high = np.searchsorted(norm_cum, upper_q)

    l, r = x_grid[idx_low], x_grid[idx_high]
    l, r = np.clip(l, cen - cut_off * fwhm, cen), np.clip(r, cen, cen + cut_off * fwhm)
    
    return l, r

def calculate_peak_areas(da: xr.DataArray, x: str, fit_results: list, confidence=0.95, tolerance=0.025, integration_max_span=3.0):
    """
    Calculate peak areas using non-overlapping integration ranges based on Voigt fits.
    """
    sorted_results = sorted(fit_results, key=lambda r: r['popt'][1])
    processed_results = []
    x_min, x_max = da[x].min().item(), da[x].max().item()

    intersections = {}
    for i in range(len(sorted_results) - 1):
        popt_curr = sorted_results[i]['popt']
        popt_next = sorted_results[i+1]['popt']
        
        c_amp, c_cen, c_sigma, c_gamma = popt_curr[:4]
        n_amp, n_cen, n_sigma, n_gamma = popt_next[:4]

        def diff_func(val):
            val_curr = voigt_profile_func(val, c_amp, c_cen, c_sigma, c_gamma, 0, 0)
            val_next = voigt_profile_func(val, n_amp, n_cen, n_sigma, n_gamma, 0, 0)
            return val_curr - val_next
        
        try:
            root = brentq(diff_func, c_cen, n_cen)
            intersections[i] = root
        except ValueError:
            intersections[i] = (c_cen + n_cen) / 2

    for i, res in enumerate(sorted_results):
        popt = res['popt']
        if len(popt) == 6:
            amp, cen, sigma, gamma, slope, offset = popt
        else:
            amp, cen, sigma, gamma = popt[:4]
            slope, offset = 0, 0
            
        l_conf, r_conf = get_confidence_bounds(amp, cen, sigma, gamma, confidence=confidence)
        
        if i == 0:
            l_final = max(l_conf, x_min)
        else:
            l_final = max(intersections[i-1], l_conf)

        if i == len(sorted_results) - 1:
            r_final = min(r_conf, x_max)
        else:
            r_final = min(intersections[i], r_conf)

        if (r_final - l_final) > integration_max_span:
            half_span = integration_max_span / 2
            l_prop = cen - half_span
            r_prop = cen + half_span
            if i > 0: l_prop = max(l_prop, intersections[i-1])
            if i < len(sorted_results) - 1: r_prop = min(r_prop, intersections[i])
            l_final, r_final = l_prop, r_prop

        if l_final >= r_final:
            l_final = cen - 0.05
            r_final = cen + 0.05

        l_integ = max(l_final, x_min)
        r_integ = min(r_final, x_max)

        subset_x = da.sel(**{x: slice(l_integ, r_integ)})[x].values
        
        if len(subset_x) < 2 or np.abs(subset_x[0] - l_integ) > tolerance or np.abs(subset_x[-1] - r_integ) > tolerance:
             grid_x = np.unique(np.concatenate(([l_integ], subset_x, [r_integ])))
             grid_x = grid_x[(grid_x >= l_integ) & (grid_x <= r_integ)] 
             if len(grid_x) > 1:
                 extended_y = da.interp({x: grid_x}).values
                 subset = xr.DataArray(extended_y, coords={x: grid_x}, dims=[x])
             else:
                 subset = xr.DataArray([], coords={x: []}, dims=[x])
        else:
            subset = da.sel(**{x: subset_x})
            
        if subset.sizes[x] > 1:
            background = slope * (subset[x] - cen) + offset
            corrected_subset = subset - background
            # Handle NaN in integration by filling. Or should we warn?
            # Integration with NaNs returns NaN.
            if np.isnan(corrected_subset).any():
                # Interpolate over NaNs or just fill with 0?
                # Filling with 0 (after bg correction) assumes no intensity where NaN is.
                # But corrected_subset = subset - background. If subset is NaN, corrected is NaN.
                # If we fill subset with 0, we might get negative peaks if background is > 0.
                # Better to interpolate subset.
                subset = subset.interpolate_na(dim=x, method='linear', fill_value="extrapolate")
                corrected_subset = subset - background
            
            try:
                area = corrected_subset.integrate(coord=x).item()
            except:
                 area = np.nan
        else:
            area = 0.0

        res_copy = res.copy()
        res_copy.update({
            'area': area,
            'integration_range': (l_integ, r_integ),
            'offset_used': offset,
            'slope_used': slope
        })
        processed_results.append(res_copy)

    return processed_results

def process_time_series(time_series_da: xr.DataArray, 
                        peaks_of_interest: list, 
                        sample_name: str = "sample",
                        output_dir: str = "time_series_results",
                        shift_threshold: float = 0.3, 
                        slope_threshold: float = 10.0,
                        perform_baseline_correction: bool = False,
                        debug: bool = False,
                        **baseline_kwargs) -> tuple[pd.DataFrame, list]:
    """
    Process a time series DataArray to track peak evolution.
    
    Parameters:
    -----------
    time_series_da : xr.DataArray
        DataArray with 'time' dimension.
    peaks_of_interest : list of tuples
        List of (center_position, window_size) for peaks to track.
    sample_name : str
         Name of sample for output directory structure.
    output_dir : str
         Base output directory.
    shift_threshold : float
        Threshold for reporting warnings about peak shifting.
    slope_threshold : float
        Position threshold above which a linear slope is allowed in fit.
    perform_baseline_correction : bool
        Whether to perform baseline correction at each step.
    debug : bool
        If True, prints progress and error details.
    **baseline_kwargs :
        Arguments passed to `baseline_correction`.

    Returns:
    --------
    df_evolution : pd.DataFrame
        Pivot table of Time vs Peak Center with Area values.
    shift_warnings : list
        List of warning strings describing large peak shifts.
    """
    
    times = time_series_da.time.values
    all_peak_data = []
    shift_warnings = []
    
    # Prepare output directory for visualizations if needed
    viz_dir = pathlib.Path(output_dir) / sample_name / "visualized_step"
    viz_dir_created = False

    if debug:
        logger.info(f"Processing {len(times)} time steps...")

    for i, t in enumerate(times):
        try:
            da_t = time_series_da.sel(time=t, method='nearest')
            
            if perform_baseline_correction:
                corrected_t, _ = baseline_correction(da_t, **baseline_kwargs)
            else:
                corrected_t = da_t.copy() 

            current_step_fits = []
            
            for peak_pos, window_size in peaks_of_interest:
                try:
                    peaks_x, intensity, _ = find_peaks_in_window(
                        corrected_t,
                        x="twoTheta_deg",
                        initial_guess=peak_pos,
                        window_size=window_size,
                        height=0.001,
                        # prominence=0.01
                    )
                    if len(peaks_x) == 0:
                        continue
                    
                    peak_x = peaks_x[np.argmax(intensity)]
                    no_slope = peak_pos > slope_threshold 
                    _, popt = voigt_fit(corrected_t, x="twoTheta_deg", peak_pos=peak_x, window_size=window_size*2, no_slope=no_slope)
                    
                    if popt[0] < 0.01:
                        continue
                        
                    current_step_fits.append({'popt': popt})
                except Exception as ex:
                     logger.debug(f"Fitting failed for window {peak_pos} at time {t}: {ex}")
                     pass

            if current_step_fits:
                areas_info = calculate_peak_areas(corrected_t, "twoTheta_deg", current_step_fits, confidence=0.95)
                
                visualize_step = False
                large_shifts = []

                for info in areas_info:
                    fitted_center = info['popt'][1]
                    
                    all_peak_data.append({
                        'Time': t,
                        'Position': fitted_center,
                        'Area': info['area']
                    })
                    
                    shifts = [np.abs(fitted_center - target) for target in [p[0] for p in peaks_of_interest]]
                    min_shift = min(shifts)
                    
                    if min_shift > shift_threshold:
                         visualize_step = True
                         closest_target = [p[0] for p in peaks_of_interest][np.argmin(shifts)]
                         msg = f"Time {t:.1f}s: Peak {closest_target}° -> {fitted_center:.3f}° (shift {min_shift:.3f})"
                         shift_warnings.append(msg)
                         large_shifts.append(msg)

                if visualize_step or i % 10 == 0:
                    if not viz_dir_created:
                         viz_dir.mkdir(parents=True, exist_ok=True)
                         viz_dir_created = True

                    fig, ax = plt.subplots(figsize=(10, 4))
                    corrected_t.plot(ax=ax, label='Data', color='black', linewidth=1)
                    
                    colors = plt.cm.tab10(np.linspace(0, 1, len(areas_info)))
                    
                    for i, info in enumerate(areas_info):
                        l, r = info['integration_range']
                        cen = info['popt'][1]
                        
                        if len(info['popt']) == 6:
                            offset = info['popt'][5] 
                            slope = info['popt'][4]
                        else:
                            offset = 0
                            slope = 0
                        
                        background_func = lambda x: slope * (x - cen) + offset
                        
                        color = colors[i]
                        
                        # Fill area
                        subset_x = corrected_t.sel(twoTheta_deg=slice(l, r))
                        if len(subset_x) > 1:
                            bg_vals = background_func(subset_x.twoTheta_deg)
                            ax.fill_between(subset_x.twoTheta_deg, subset_x, bg_vals, 
                                            where=(subset_x > bg_vals),
                                            alpha=0.3, color=color)
                        
                        # Fit
                        fit_x_vals = corrected_t.sel(twoTheta_deg=slice(cen-1, cen+1)).twoTheta_deg.values
                        if len(fit_x_vals) > 0:
                            y_fit = voigt_profile_func(fit_x_vals, *info['popt'])
                            ax.plot(fit_x_vals, y_fit, '--', color=color, alpha=0.8)
                            
                        ax.axvline(l, linestyle=':', color=color, alpha=0.5)
                        ax.axvline(r, linestyle=':', color=color, alpha=0.5)

                    ax.set_title(f"Shift Detected at t={t}s")
                    ax.set_yscale('log')
                    # add large_shifts msg as text box
                    if large_shifts:
                        textstr = "\n".join(large_shifts)
                        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                                verticalalignment='top', bbox=props)
                    
                    # Save plot
                    plot_path = viz_dir / f"shift_t_{t:.2f}s.png"
                    fig.savefig(plot_path)
                    plt.close(fig)
                    logger.info(f"Saved shift visualization to {plot_path}")

        except Exception as e:
            if debug:
                logger.error(f"Error at time {t}: {e}")

    if all_peak_data:
        df_raw = pd.DataFrame(all_peak_data)
        df_raw['PeakBin'] = df_raw['Position'].round(1)
        df_evolution = df_raw.pivot_table(index='Time', columns='PeakBin', values='Area')
        df_evolution.columns.name = 'Peak Center'
        df_evolution = df_evolution.sort_index(axis=1)
    else:
        df_evolution = pd.DataFrame()

    return df_evolution, shift_warnings


def process_time_series_by_peak(
    time_series_da: xr.DataArray,
    peaks_definition: list,
    sample_name: str,
    output_dir: str,
    debug: bool = False,
    **baseline_kwargs
) -> pd.DataFrame:
    """
    Process time series data peak-by-peak with bi-directional tracking.
    
    Parameters:
    peaks_definition : list of tuples
        Format: [(target_pos, window_size, (start_idx, end_idx), peak_name), ...]
        start_idx: Index to start tracking from.
        end_idx: Index to end at (inclusive). 
                 If start < end, tracks forward. If start > end, tracks backward.
    """
    times = time_series_da.time.values
    num_times = len(times)
    
    exp_peak_results = []
    
    # Prepare viz directory
    viz_dir = pathlib.Path(output_dir) / sample_name / "peak_tracking_viz"
    if viz_dir.exists():
    # use n+1 to avoid overwriting existing results
        existing_dirs = [d for d in viz_dir.parent.iterdir() if d.is_dir() and d.name.startswith(viz_dir.name)]
        viz_dir = viz_dir.parent / f"{viz_dir.name}_{len(existing_dirs)+1}"
    viz_dir.mkdir(parents=True, exist_ok=False)
    
    for (peak_target, window_size, (start_idx, end_idx), peak_name) in peaks_definition:
        if debug:
            logger.info(f"Tracking Peak '{peak_name}' ({start_idx} -> {end_idx})...")

        # Handle negative indices
        if start_idx < 0: start_idx += num_times
        if end_idx < 0: end_idx += num_times
        
        # Determine Direction and Range
        step = 1 if end_idx >= start_idx else -1
        
        # Generate indices to process (inclusive of end_idx)
        # Python range stop is exclusive, so we add step
        processing_indices = range(start_idx, end_idx + step, step)
        
        peak_lost = True
        last_popt = None
        last_ref_time = None
        first_peak_found = False
        
        for t_idx in processing_indices:
            if t_idx < 0 or t_idx >= num_times: continue
            
            t = times[t_idx]
            da_t = time_series_da.sel(time=t, method='nearest')
            
            # Use raw data (baseline correction handled inside if needed, assuming user passes kwargs)
            if baseline_kwargs.get('method'):
                 da_t, _ = baseline_correction(da_t, **baseline_kwargs)

            fit_success = False
            current_popt = None
            
            # --- Phase: TRACK (Optimization) ---
            # Try to track if we have a previous fix
            if not peak_lost:
                popt_arr = np.array(last_popt)
                tolerance = 0.2
                
                # Bounds: +/- 20%, but enforce a minimum absolute breathing room of 0.001
                # This prevents "pincering" parameters that are near zero (like gamma/sigma/slope)
                delta = np.maximum(np.abs(popt_arr) * tolerance, 1e-3)
                
                lower_bounds = popt_arr - delta
                upper_bounds = popt_arr + delta
                
                # Physical Constraints
                # [amp, cen, sigma, gamma, slope, offset]
                lower_bounds[0] = max(0, lower_bounds[0])       # amp > 0
                lower_bounds[2] = max(1e-6, lower_bounds[2])    # sigma > 0
                lower_bounds[3] = max(1e-6, lower_bounds[3])    # gamma > 0
                
                # Ensure initial_guess is within bounds to avoid ValueError in curve_fit
                initial_guess_clamped = np.clip(last_popt, lower_bounds, upper_bounds).tolist()

                # Fit window is 2x the original FWHM estimate from previous fit, but at least 0.5° to allow for some movement
                window_size = max(0.5, calculate_fwhm(popt_arr[2], popt_arr[3]) * 3)
                
                try:
                    _, popt = voigt_fit(
                        da_t, x="twoTheta_deg",
                        initial_guess=initial_guess_clamped,
                        window_size=window_size,
                        bounds=(lower_bounds, upper_bounds)
                    )
                    current_popt = popt
                    fit_success = True
                except (RuntimeError, ValueError) as e:
                    fit_success = False
                    peak_lost = True 
                    if debug: 
                        logger.info(f"  Tracking lost at t={t:.1f}s (Error: {e}). Switching to search.")

            # --- Phase: SEARCH (Discovery) ---
            # If peak is lost (or tracking failed just now), we search
            if peak_lost:
                peaks_x, intensity, props = find_peaks_in_window(
                    da_t, x="twoTheta_deg", 
                    target=peak_target, window_size=window_size, 
                    height=0.001
                )
                
                if len(peaks_x) > 0:
                    best_idx = np.argmax(intensity)
                    peak_pos_guess = peaks_x[best_idx]
                    peak_width = props[best_idx]['half_widths']
                    
                    try:
                        no_slope = peak_pos_guess > 10.0
                        _, popt = voigt_fit(
                            da_t, x="twoTheta_deg", 
                            initial_guess=peak_pos_guess, 
                            window_size=max(0.5, peak_width * 4), 
                            no_slope=no_slope
                        )
                        current_popt = popt
                        fit_success = True
                        
                    except RuntimeError:
                        pass

            # --- Capture Result ---
            if fit_success:
                try:
                    fit_result_obj = {'popt': current_popt}
                    peak_area_res = calculate_peak_areas(da_t, "twoTheta_deg", [fit_result_obj], confidence=0.95)[0]
                    
                    ref_val = last_ref_time if not peak_lost else "Search"
                    
                    if debug: logger.info(f"Captured peak '{peak_name}' at t={t:.1f}, Area={peak_area_res['area']:.4f}")

                    exp_peak_results.append({
                        'Time': t,
                        'PeakName': peak_name,
                        'Position': peak_area_res['popt'][1],
                        'Area': peak_area_res['area'],
                        'RefTime': ref_val
                    })
                    
                    peak_lost = False
                    last_popt = current_popt
                    last_ref_time = t
                except Exception as e:
                    msg = f"Failed to capture result at t={t}: {e}"
                    logger.error(msg)
                    peak_lost = True 
            else:
                peak_lost = True
                last_popt = None

            # --- Visualization (Found / Refound) ---
            if fit_success and (peak_lost or not first_peak_found or t_idx % 10 == 0):
                try:
                    popt = current_popt # Ensure we use the current fit
                    fig, ax = plt.subplots(figsize=(6, 4))
                    plot_min, plot_max = peak_target - window_size, peak_target + window_size
                    da_t.sel(twoTheta_deg=slice(plot_min, plot_max)).plot(ax=ax, label='Data')
                    # fill area that was integrated
                    cen, slope, offset = popt[1], popt[4] if len(popt) == 6 else 0, popt[5] if len(popt) == 6 else 0
                    l, r = peak_area_res['integration_range']
                    background_func = lambda x: slope * (x - cen) + offset
                    subset_x = da_t.sel(twoTheta_deg=slice(l, r))
                    sunset_bg_vals = background_func(subset_x.twoTheta_deg)

                    ax.fill_between(subset_x.twoTheta_deg, subset_x.values, sunset_bg_vals,
                                    where=(subset_x.values > sunset_bg_vals),
                                    alpha=0.3, color='orange', label='Integrated Area')
                    # Use the fit window (3*FWHM) for plotting the fit, but limit to data range
                    fwhm = calculate_fwhm(popt[2], popt[3])
                    fit_x = np.linspace(max(plot_min, cen - 2 * fwhm), min(plot_max, cen + 2 * fwhm), 100)
                    ax.plot(fit_x, voigt_profile_func(fit_x, *popt), 'r--', label='Fit')
                    ax.axvline(l, linestyle=':', alpha=0.5)
                    ax.axvline(r, linestyle=':', alpha=0.5)
                    
                    # add the fitting parameters as text
                    param_text = f"Amp: {popt[0]:.4f}\nCen: {popt[1]:.4f}\nSigma: {popt[2]:.3f}\nGamma: {popt[3]:.3f}"
                    if len(popt) == 6:
                        param_text += f"\nSlope: {popt[4]:.4f}\nOffset: {popt[5]:.4f}"
                    ax.text(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    status_text = "Refound" if peak_lost and first_peak_found else "Found"
                    ax.set_title(f"'{peak_name}' {status_text} at t={t:.1f}s (idx {t_idx})")
                    fig.savefig(viz_dir / f"{peak_name}_{status_text.lower()}_idx{t_idx}.png")
                    
                    first_peak_found = True
                except Exception as e:
                    if debug: logger.warning(f"Visualization failed: {e}")
                    raise
                finally:
                    plt.close(fig)




    # Format Output
    if debug: logger.info(f"Total results captured: {len(exp_peak_results)}")
    
    if exp_peak_results:
        df_raw = pd.DataFrame(exp_peak_results)
        # Check if 'Area' is all NaN or something?
        if debug: 
             logger.info("\n--- Raw Results Preview ---")
             logger.info(df_raw.head())
             logger.info(f"Unique Peaks: {df_raw['PeakName'].unique()}")
             
        df_pivot = df_raw.pivot_table(index='Time', columns='PeakName', values='Area')
        return df_pivot
    else:
        if debug: logger.warning("No results collected for this experiment.")
        return pd.DataFrame()
