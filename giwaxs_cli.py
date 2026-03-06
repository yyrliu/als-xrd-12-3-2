import argparse
from pathlib import Path
import fabio
from natsort import natsorted
import pyFAI
from pyFAI.calibrant import Calibrant
from pyFAI.geometry import Geometry
from pyFAI.goniometer import SingleGeometry
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy import percentile
import numpy as np
import xarray as xr
import datetime
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
import json

def get_file_creation_time(file_path):
    p = Path(file_path)
    stat_result = p.stat()
    
    # Use st_birthtime if available (macOS/some Unix), otherwise use st_ctime (Windows/metadata change time on Unix)
    if hasattr(stat_result, 'st_birthtime'):
        timestamp = stat_result.st_birthtime
    else:
        timestamp = stat_result.st_ctime
        
    return datetime.datetime.fromtimestamp(timestamp)

def process_frame(frame_file, ai, radial_range=None, azimuth_range=None):
    frame_timestamp = get_file_creation_time(frame_file)
    frame = fabio.open(str(frame_file)).data
    res = ai.integrate1d(
        frame,
        npt=2000,
        radial_range=radial_range,
        azimuth_range=azimuth_range,
        unit="q_A^-1",
        method=("full", "histogram", "cython"),
        polarization_factor=0.95,
    )
    return (res.radial, res.intensity, frame_timestamp, int(frame_file.stem.split("_")[-1]))


def process_frames(frame_files, poni_file, radial_range=(0.3, 4), azimuth_range=(-35, 35), num_workers=None):
    ai = pyFAI.load(poni_file)

    if num_workers != 1:

        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)

        results = Parallel(n_jobs=num_workers, prefer="threads")(
            delayed(process_frame)(frame_file, ai, radial_range, azimuth_range) for frame_file in frame_files
        )

    else:
        results = []
        for frame_file in tqdm(frame_files, desc="Processing frames"):
            res = process_frame(frame_file, ai, radial_range, azimuth_range)
            results.append(res)

    radial = results[0][0]
    intensity = [res[1] for res in results]
    timestamps = [res[2] for res in results]
    frame_indices = [res[3] for res in results]

    # The resolution of timestamps may not be sufficient, so we can use frame indices if needed
    # if multiple frames have the same timestamp, we use np.linespace to create unique times
    # create elapsed seconds array from timestamps
    times_seconds = np.array([(t - timestamps[0]).total_seconds() for t in timestamps], dtype=float)

    # ensure uniqueness: if multiple frames share the same timestamp, spread them by small offsets
    unique_vals, counts = np.unique(times_seconds, return_counts=True)
    dup_vals = unique_vals[counts > 1]
    for val in dup_vals:
        idxs = np.where(times_seconds == val)[0]
        # order duplicates by frame index to preserve acquisition order
        order = np.argsort([frame_indices[i] for i in idxs])
        idxs_sorted = idxs[order]
        count = len(idxs_sorted)

        if idxs_sorted[0] == 0:
            if len(idxs_sorted) > 0 and len(times_seconds) > idxs_sorted[-1]+1:
                 new_vals = np.linspace(0, times_seconds[idxs_sorted[-1]+1], count, endpoint=False)
            else:
                 new_vals = np.linspace(0, 1, count, endpoint=False) # Fallback if only one group or end of array
        else:
            # spread evenly between previous and next unique time
            new_vals = np.linspace(times_seconds[idxs_sorted[0]-1], val, count + 1)[1:]

        times_seconds[idxs_sorted] = new_vals
        # for idx in idxs_sorted:
        #    print(f"Adjusted time for frame {frame_indices[idx]} from {val} to {times_seconds[idx]} seconds")

    data_array = xr.DataArray(
        data=intensity,
        dims=["time", "q_A^-1"],
        coords={"time": times_seconds, "q_A^-1": radial},
        attrs={"description": "GIWAXS integrated intensity over time"},
    )

    # assign frame indices as coordinate
    data_array = data_array.assign_coords(frame=("time", frame_indices))

    return data_array

def refine_geometry(image_file, calibrant_file, initial_poni, ax=None):
    print(f"Refining geometry with {str(Path(image_file).name)}")
    calibrant = Calibrant(filename=str(calibrant_file))
    initial_geometry = Geometry()
    initial_geometry.load(str(initial_poni))
    detector = initial_geometry.detector
    image = fabio.open(str(image_file)).data

    sg_label = "Recalibration_" + str(Path(image_file).parent.name)

    sg = SingleGeometry(sg_label, image, calibrant=calibrant, detector=detector, geometry=initial_geometry)
    sg.extract_cp(max_rings=5)
    sg.geometry_refinement.refine2(fix=["wavelength"])
    sg.get_ai()

    # jupyter.display(sg=sg) # This is for notebook
    
    refined_poni_path = str(Path(image_file).parent / f"{sg_label}.poni")
    if Path(refined_poni_path).is_file():
        Path(refined_poni_path).unlink()
    sg.geometry_refinement.save(refined_poni_path)
    print(f"Refined geometry saved to {refined_poni_path}")
    
    return refined_poni_path, sg

def create_plots(exp_dir, frame_files, first_image, refined_poni, data_array, nc_file_path, sg=None):
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4) # 2 rows, 4 cols
    
    # Plot 1: First Image raw
    ax1 = fig.add_subplot(gs[0, 0])
    frame_data = fabio.open(str(first_image)).data
    ax1.imshow(frame_data, norm=LogNorm(), cmap='viridis')
    ax1.set_title(f"First Image: {first_image.name}")

    # Plot 2: Recalibration (with rings)
    ax2 = fig.add_subplot(gs[0, 1])
    if sg:
        ax2.imshow(sg.image, norm=LogNorm(), cmap='viridis')
        try:
            ai = sg.get_ai()
            shape = sg.image.shape
            tth = ai.center_array(shape, unit="2th_rad")
            # Get expected rings from calibrant
            rings = sg.calibrant.get_2th()
            # Filter rings that are within the image view vaguely (approx max tth)
            max_tth = tth.max()
            rings_visible = [r for r in rings if r < max_tth]
            
            if rings_visible:
                 ax2.contour(tth, levels=rings_visible, colors='r', linewidths=1)
            ax2.set_title("Recalibration (Rings)")
        except Exception as e:
            print(f"Could not plot rings: {e}")
            ax2.set_title("Recalibration (Image Only)")
    else:
        ax2.text(0.5, 0.5, "No Geometry Object", ha='center', va='center')
        ax2.set_title("Recalibration")

    # Plot 3: 1D Integration Sample (mid-way)
    ax3 = fig.add_subplot(gs[0, 2])
    ai_refined = pyFAI.load(str(refined_poni))
    mid_idx = len(frame_files) // 2
    res_1d = ai_refined.integrate1d(fabio.open(str(frame_files[mid_idx])).data, 500, unit="q_A^-1", azimuth_range=(-35, 35), radial_range=(0.3, 4), polarization_factor=0.95)
    ax3.plot(res_1d.radial, res_1d.intensity)
    ax3.set_yscale("log")
    ax3.set_xlabel("q ($A^{-1}$)")
    ax3.set_ylabel("Intensity")
    ax3.set_title(f"1D Integration (Frame {mid_idx})")

    # Plot 4: 2D Integration Sample
    ax4 = fig.add_subplot(gs[0, 3])
    res_2d, tth, azi = ai_refined.integrate2d(fabio.open(str(first_image)).data, 500, 500, unit="2th_deg", polarization_factor=0.95)
    ax4.imshow(res_2d, origin='lower', extent=[tth.min(), tth.max(), azi.min(), azi.max()], aspect='auto', cmap='viridis', norm=LogNorm())
    ax4.set_title("2D Integration (First Image)")
    ax4.set_xlabel("2Theta (deg)")
    ax4.set_ylabel("Azimuth (deg)")

    # Plot 5: Time Evolution (The main result) - Spanning bottom
    ax5 = fig.add_subplot(gs[1, :])
    
    # Replicating original plot logic
    data_array.plot.imshow(x="time", y="q_A^-1", cmap="viridis", norm=LogNorm(vmin=percentile(data_array, 50), vmax=percentile(data_array, 99)), ax=ax5)
    # data_array.plot.imshow automatically handles colorbars usually, but we want to customize?
    ax5.set_ylabel(r"q ($\AA^{-1}$)")
    ax5.set_xlabel("Time (s)")
    ax5.set_title( f"Evolution: {exp_dir.stem}")

    plt.tight_layout()
    composite_plot_path = nc_file_path.parent / f"{exp_dir.stem}_summary.png"
    fig.savefig(composite_plot_path, dpi=300)
    print(f"Saved summary plot to {composite_plot_path}")
    
    # Original specific plot save (only the time evolution)
    fig_daily, ax_daily = plt.subplots(figsize=(8, 6))
    data_array.plot.imshow(x="time", y="q_A^-1", cmap="viridis", norm=LogNorm(vmin=percentile(data_array, 50), vmax=percentile(data_array, 99)), ax=ax_daily)
    # ax_daily.images[0].colorbar.set_label("Intensity") # xarray handles this
    ax_daily.set_ylabel(r"q ($\AA^{-1}$)")
    ax_daily.set_xlabel("Time (s)")
    ax_daily.set_title(exp_dir.stem)
    fig_daily.tight_layout()
    daily_plot_path = nc_file_path.with_suffix('.png')
    fig_daily.savefig(daily_plot_path, dpi=300)
    print(f"Saved standard plot to {daily_plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Process GIWAXS experiment data.")
    parser.add_argument("exp_dir", type=str, help="Experiment directory path")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers (default: all cores minus one)")
    parser.add_argument("--save_csv", action="store_true", help="Whether to save integrated data as CSV in addition to NetCDF")
    parser.add_argument("--output_dir", type=str, default='./processed', help="Directory to save outputs (default: ./processed)")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    poni_file = ""
    calibrant_file = ""

    # Find PONI file first in dir then parent dir
    try:
        poni_file = next(exp_dir.glob("*.poni"))
    except StopIteration:
        try:
            poni_file = next(exp_dir.parent.glob("*.poni"))
        except StopIteration:
            print(f"No PONI file found in {exp_dir} or its parent directory. Using default geometry parameters.")
            poni_file = Path('./default_calibrations/default_geometry.poni')
            if not poni_file.is_file():
                raise FileNotFoundError("Default PONI file not found at './default_calibrations/default_geometry.poni'. Please provide a PONI file in the experiment directory or its parent directory.")

    # Find Calibrant file
    calibrant_file = Path(exp_dir.parents[1], "ito_calibrant.D")
    if not calibrant_file.is_file():
        calibrant_file = Path('./default_calibrations/ito_calibrant.D')
        if not calibrant_file.is_file():
            raise FileNotFoundError(f"Calibrant file not found at {calibrant_file} or {Path('./default_calibrations/ito_calibrant.D')}. Please provide the calibrant file in the expected location.")

    frame_files = natsorted(list(exp_dir.glob("*.tif")))
    if not frame_files:
        raise FileNotFoundError(f"No .tif files found in {exp_dir}")

    first_image = frame_files[0]
    
    # 1. Refine Geometry
    # If poni file is in the same dir as images, skip refinement
    if poni_file.parent == exp_dir:
        print(f"PONI file found in experiment directory; skipping refinement.")
        refined_poni = poni_file
        sg = None
    else:
        refined_poni_path, sg = refine_geometry(first_image, calibrant_file, poni_file)
        refined_poni = Path(refined_poni_path)

    # 2. Process Frames
    print(f"Processing {len(frame_files)} frames with {args.num_workers or 'all available'} workers...")
    data_array = process_frames(frame_files, str(refined_poni))
    print(f"Processed {len(frame_files)} frames.")
    print(f"DataArray shape: {data_array.shape}, q range: {data_array.coords['q_A^-1'].values[0]} to {data_array.coords['q_A^-1'].values[-1]}")

    # 3. Save NetCDF
    data_array.attrs["title"] = f"{exp_dir.stem}"
    data_array.attrs["base_poni_file"] = poni_file.as_posix()
    data_array.attrs["poni_file"] = refined_poni.as_posix()
    data_array.attrs["source_dir"] = exp_dir.as_posix()
    data_array.attrs["tif_tags"] = json.dumps(fabio.open(str(first_image)).header)
    
    # save_dir = Path(r"G:\Shared drives\Sutter-Fella Lab\ECRP-Project\results\beamtime_Dec2025") # From notebook, maybe generalize?
    # Using parent folder of exp_dir as typical default or the exp_dir itself? 
    # The notebook saved to: G:\Shared drives\Sutter-Fella Lab\ECRP-Project\results\beamtime_Dec2025
    # which is exp_dir.parent

    if args.output_dir:
        if not Path(args.output_dir).is_absolute():
            save_dir = exp_dir.parent / args.output_dir
        else:
            save_dir = Path(args.output_dir)
    else:
        save_dir = exp_dir.parent

    if not save_dir.exists():
        save_dir.mkdir(parents=True)
        print(f"Created output directory at {save_dir}")

    nc_file_path = save_dir / f"{exp_dir.stem}.nc"
    if nc_file_path.is_file():
        nc_file_path.unlink()
        # double check deletion
        if nc_file_path.is_file():
            raise FileExistsError(f"Could not delete existing NetCDF file at {nc_file_path}")
        
    data_array.to_netcdf(nc_file_path)
    print(f"Saved NetCDF to {nc_file_path}")
        
    # save as csv for quick access (optional)
    if args.save_csv:
        csv_file_path = nc_file_path.with_suffix('.csv')
        with open(csv_file_path, 'w') as f:
            # write metadata as comments
            for key, value in data_array.attrs.items():
                f.write(f"# {key}: {value}\n")

        data_array.to_pandas().to_csv(csv_file_path, mode='a')
        print(f"Saved CSV to {csv_file_path}")

    # 4. Create Plots
    create_plots(exp_dir, frame_files, first_image, refined_poni, data_array, nc_file_path, sg=sg)

if __name__ == "__main__":
    main()
