# CLI Usage Guide

## Basic Commands

### List available runs
```bash
# Using default config file (peak_tracking_config.py)
python peak_tracking_workflow.py --list-runs

# Using custom config file
python peak_tracking_workflow.py --config my_config.py --list-runs
```

### Run a single preset
```bash
# Using default config
python peak_tracking_workflow.py --run insitu_0.5M_MeOMBAI

# Using custom config file
python peak_tracking_workflow.py --config experiments/feb2026_config.py --run insitu_0.5M_MeOMBAI

# Override output directory
python peak_tracking_workflow.py --run insitu_0.5M_MeOMBAI --output-root ./results
```

### Run all presets in parallel
```bash
# Run all with default workers (all CPU cores)
python peak_tracking_workflow.py --run-all

# Run with specific number of parallel jobs
python peak_tracking_workflow.py --run-all --jobs 4

# With custom config and output
python peak_tracking_workflow.py --config my_config.py --run-all --jobs 2 --output-root ./batch_results
```

## New Features (2026-05-26)

### Custom Config File
You can now specify a different config file for different experiments:

```bash
# February 2026 beamtime
python peak_tracking_workflow.py --config configs/feb2026.py --run insitu_0.5M_MeOMBAI

# Different experimental conditions
python peak_tracking_workflow.py --config configs/high_temp_experiments.py --run-all

# Testing new validation parameters
python peak_tracking_workflow.py --config configs/test_validation.py --run insitu_0.5M_MeOMBAI
```

### Config File Requirements

Your config file must be a Python module (.py) that defines:

**Required:**
- `RUN_CONFIGS`: Dictionary of run configurations

**Optional:**
- `DEFAULT_OUTPUT_ROOT`: Default output directory (defaults to `./output` if not specified)

Example minimal config:
```python
from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path("./my_results")

RUN_CONFIGS = {
    "my_run": {
        "da_file": Path("data/my_data.nc"),
        "global_shift_deg": 0.0,
        "keyframe_plot_margin_deg": 2.0,
        "peaks": [
            (30.6, 1.0, (-1, 0), "ITO", {
                "drift_tolerance_deg": 0.05,
                "min_snr": 5.0,
                "use_adaptive_bounds": True,
            }),
        ],
    },
}
```

## Using with uv

If using `uv` for dependency management:

```bash
# Default config
uv run python peak_tracking_workflow.py --run insitu_0.5M_MeOMBAI

# Custom config
uv run python peak_tracking_workflow.py --config my_config.py --run insitu_0.5M_MeOMBAI

# Run all
uv run python peak_tracking_workflow.py --config my_config.py --run-all
```

## Common Use Cases

### Development/Testing
```bash
# Test with validation only (no adaptive bounds)
python peak_tracking_workflow.py \
  --config configs/validation_test.py \
  --run insitu_0.5M_MeOMBAI \
  --output-root ./test_results
```

### Production Run
```bash
# Full pipeline with adaptive bounds enabled
python peak_tracking_workflow.py \
  --config peak_tracking_config.py \
  --run-all \
  --jobs -1 \
  --output-root /path/to/production/results
```

### Quick Experiment
```bash
# Create temporary config on the fly
cat > temp_config.py << 'EOF'
from pathlib import Path

RUN_CONFIGS = {
    "quick_test": {
        "da_file": Path("data/test.nc"),
        "peaks": [(30.6, 1.0, (0, 100), "ITO", {})],
    }
}
EOF

python peak_tracking_workflow.py --config temp_config.py --run quick_test
```

## Error Handling

If config file is not found:
```
Error loading config file my_config.py: [Errno 2] No such file or directory: 'my_config.py'
```

If config file missing RUN_CONFIGS:
```
Error loading config file my_config.py: Config file my_config.py must define RUN_CONFIGS
```

If run preset not found:
```
KeyError: Unknown run preset: my_run
```

## Migration from Old Usage

**Before (hardcoded config):**
```bash
python peak_tracking_workflow.py --run insitu_0.5M_MeOMBAI
```

**After (same behavior, explicit default):**
```bash
python peak_tracking_workflow.py --config peak_tracking_config.py --run insitu_0.5M_MeOMBAI
```

The default behavior is unchanged - if you don't specify `--config`, it uses `peak_tracking_config.py` from the same directory as the script.
