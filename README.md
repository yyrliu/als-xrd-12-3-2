# GIWAXS Processing Tool

## 1. How to set up uv

`uv` is an extremely fast Python package installer and resolver.

**Installation:**

*   **macOS / Linux:**
    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

*   **Windows:**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

For more details, refer to the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

## 2. How to set up the project environment

Once `uv` is installed, set up the project environment and install dependencies by running:

```sh
uv sync
```

This will create a virtual environment and install all required packages specified in `pyproject.toml`.

## 3. How to create the required PONI file

You need a `.poni` file which defines the geometry of the experimental setup. You can generate this using the `pyFAI-calib2` GUI tool.

Run the tool with:
```sh
uv run pyFAI-calib2
```

**Brief Intro:**
`pyFAI-calib2` is a graphical interface that helps you determine the geometry of your detector (distance, center, rotation) using a calibration image (e.g., Al2O3 or ITO). You will need to select your detector, specify the calibrant, and pick peaks on the image to refine the geometry.

For a step-by-step guide, please see: [pyFAI Calibration Cookbook](https://pyfai.readthedocs.io/en/stable/usage/cookbook/calib-gui/index.html)

## 4. How to use `giwaxs_cli.py`

The `giwaxs_cli.py` tool processes your experiment data (TIFF files), performs integration, and saves the results.

### Expected File Structure

The tool relies on a specific folder hierarchy to automatically locate the `.poni` file and the calibrant file (`ito_calibrant.D`).

Ensure your files are organized as follows:

```text
root_directory/                <-- Work from this directory
├── ito_calibrant.D            <-- Calibrant file (Required)
├── <sample_group_folder>/     <-- Parent folder for experiments
│   ├── calibration.poni       <-- Initial PONI file (generated in step 3)
│   └── <experiment_folder>/   <-- The folder containing your .tif files
│       ├── image_0001.tif
│       ├── image_0002.tif
│       └── ...
```

*   **`ito_calibrant.D`**: Must be in the grandparent directory of the experiment images.
*   **`.poni` file**: Must be in the parent directory of the experiment images (same level as the experiment folder).
*   **Experiment Folder**: Contains the sequence of `.tif` images to process.

### Running the Tool

To process an experiment, run the following command, pointing to the **experiment folder**:

```sh
uv run giwaxs_cli.py "path/to/sample_group_folder/experiment_folder"
```

**Example:**
If your structure is `data/ito_calibrant.D`, `data/run1/calib.poni`, and `data/run1/sampleA/`, you would run:
```sh
uv run giwaxs_cli.py "data/run1/sampleA"
```

**Outputs:**
*   A refined `.poni` file is saved in the `<sample_group_folder>`.
*   A NetCDF (`.nc`) file containing the processed data is saved in the `root_directory`.
*   Summary plots (`.png`) are generated in the `root_directory`.

## 5. Output Structure & Visualization

The processing tool generates a NetCDF (`.nc`) file containing the integrated GIWAXS data, structured as an `xarray.DataArray`.

### NetCDF File Structure
The file is organized with the following dimensions and coordinates:

*   **Dimensions:**
    *   `time`: Elapsed time in seconds from the first frame.
    *   `q_A^-1`: Radial scattering vector ($q$) in $\AA^{-1}$.
*   **Coordinates:**
    *   `time`: Array of time stamps.
    *   `q_A^-1`: Array of $q$ values.
    *   `frame`: (Auxiliary coordinate) The original file index corresponding to each time point.
*   **Data:** Integrated intensity values.
*   **Attributes:** Metadata covering source directories, PONI files used, and experimental tags.

### Visualizing Results

The results can be easily visualized using Python with `xarray` and `matplotlib`.

**Example:**

```python
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# 1. Load the data
data = xr.open_dataarray("path/to/experiment.nc")

# 2. Plot the time evolution (2D Map)
plt.figure(figsize=(10, 6))
data.plot.imshow(norm=LogNorm(), cmap='viridis')
plt.title("GIWAXS Time Evolution")
plt.show()

# 3. Plot a single frame (1D Profile)
# Select the 10th time point
plt.figure()
data.isel(time=10).plot(yscale="log")
plt.title("Integration at Frame 10")
plt.show()
```

