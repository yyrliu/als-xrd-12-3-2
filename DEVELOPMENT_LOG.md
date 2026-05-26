# Development Log

## 2026-05-25: ClMBAI fit quality investigation and fitter upgrade

1. Observed that the ClMBAI 2D (002) key-frame inspection plot looked poor even though the fit was landing near the expected peak center.
2. Checked the fit metrics and found the problem was not the plot window: the fit quality was noticeably worse than comparable MeOMBAI fits, and `sigma` often collapsed to the lower bound.
3. Added explicit fit-quality columns to the tracking CSV output so peak quality can be compared numerically:
   - `fit_r2`
   - `fit_redchi`
   - `fit_rmse`
   - `fit_rel_rmse`
   - `fit_quality_score`
   - `fit_chisqr`
4. Concluded that introducing a separate `fit_window_deg` would add confusion because `window_deg` already defines the fitting/search neighborhood.
5. Added new built-in fitters to better match ClMBAI peak/background structure:
   - `voigt_exp`: target Voigt + exponential background
   - `double_voigt_linear`: target Voigt + background Voigt + linear term
6. Updated the background integration/plotting logic so all non-target model components are treated as background in the outputs and inspection plots.
7. Added ClMBAI demo presets in `peak_tracking_config.py` that use `key_frames_mode: "restrict"` with a few hand-picked frames so users can inspect and compare fit quality without running every frame.
8. Kept user-configured key frames additive/manual-reset for the main workflow, but used restrict-mode demos specifically to compare fitter behavior on selected frames.
9. Next validation step is to run the new ClMBAI demo preset and inspect the key-frame plot and fit-quality metrics side by side.
10. Validation result: `voigt_exp` failed on the first configured ClMBAI key frame with a fit tolerance / covariance message, so it is not yet a stable default for this dataset.
11. Validation result: `double_voigt_linear` succeeded on the same ClMBAI key frames and improved the fit-quality metrics materially (`fit_r2` ~0.93-0.94, `fit_redchi` ~1.35-1.68), so it is the better demo fitter for showing why `key_frames_mode: "restrict"` is useful.
12. After tightening the target peak width floor and re-checking the key-frame plot, `voigt_exp` became the cleaner visual choice for ClMBAI: the peak overlay is physically reasonable and the background stays monotonic, while `double_voigt_linear` still tends to overfit the shoulder.
13. **Root cause identified** for persistent amplitude underestimation across all fitters:
    - `peak_amplitude` was initialized as the raw data peak *height* value, but lmfit's `VoigtModel.amplitude` is the profile *area* (integral). Starting from a height value put the optimizer ~8× too high, forcing it into a bad local minimum (wide, short peak + inflated background).
    - `peak_gamma` was initialized to `window_deg / 8 = 0.125°`, producing a Lorentzian FWHM of 0.25° that is roughly 2× wider than the actual ClMBAI 2D (002) peak. The optimizer could not recover from this starting point without sacrificing amplitude accuracy.
    - The fit window (`fit_half_width = window_deg / 2 = 0.5°`) extended into the steep exponential shoulder at ~5.9°, giving the background model a high anchor point that further depressed the peak amplitude.
14. **Fixes applied** (all three needed to resolve the gap):
    - `peak_gamma` init changed to `window_deg / 16` (0.0625° for `window_deg=1.0`) — physically closer to real perovskite peak widths.
    - `peak_amplitude` init changed to `peak_height * π * gamma_init` — correct area-domain starting point for the Lorentzian limit.
    - Added `fit_half_width_deg` field to `FitterSpec` so custom fitters can specify their own fit range independently of `window_deg`. Both ClMBAI demo presets now use `fit_half_width_deg: 0.4` to exclude the steepest shoulder region from the fit.
15. **Final validation — fit is now good** (run `ClMBAI_demo_restrict_voigt_exp_20260525_175331`): the integrated area fills the data peak with no visible gap across all three key frames.

## Iteration metrics summary

All runs used `key_frames_mode: "restrict"` on frames [453, 738, 828] of `ClMBAI_0M5_1 004910 Images.nc`.  
Plot paths are relative to `G:\Shared drives\Sutter-Fella Lab\ALS_Beamtimes\2026\Yi-Ru_Feb2026\processed\agent_temp\`.

| Run timestamp | Fitter | Frame | fit_r2 | fit_redchi | fit_rel_rmse | fit_sigma | fit_gamma | Visual | Plot |
|---|---|---|---|---|---|---|---|---|---|
| 170723 | voigt_exp | 453 | 0.834 | 3.28 | 0.105 | 1e-6 | 0.125 | ❌ gap | [plot](ClMBAI_demo_restrict_voigt_exp_20260525_170723/fit_plots/fit_2D_(002)_idx453.png) |
| 170723 | voigt_exp | 738 | 0.824 | 3.94 | 0.105 | 1e-6 | 0.125 | ❌ gap | [plot](ClMBAI_demo_restrict_voigt_exp_20260525_170723/fit_plots/fit_2D_(002)_idx738.png) |
| 170723 | voigt_exp | 828 | — | — | — | 1e-6 | 0.125 | ❌ gap | [plot](ClMBAI_demo_restrict_voigt_exp_20260525_170723/fit_plots/fit_2D_(002)_idx828.png) |
| 170708 | double_voigt_linear | 453 | 0.907 | 2.03 | 0.078 | 1e-6 | 0.125 | ❌ gap | [plot](ClMBAI_demo_restrict_double_voigt_20260525_170708/fit_plots/fit_2D_(002)_idx453.png) |
| 170708 | double_voigt_linear | 738 | 0.918 | 2.01 | 0.072 | 1e-6 | 0.125 | ❌ gap | [plot](ClMBAI_demo_restrict_double_voigt_20260525_170708/fit_plots/fit_2D_(002)_idx738.png) |
| 171135 | voigt_exp + sigma_floor | 453 | 0.820 | 3.56 | 0.109 | 0.02 | 0.125 | ❌ gap | [plot](ClMBAI_demo_restrict_voigt_exp_20260525_171135/fit_plots/fit_2D_(002)_idx453.png) |
| 171135 | voigt_exp + sigma_floor | 738 | 0.811 | 4.25 | 0.109 | 0.02 | 0.125 | ❌ gap | [plot](ClMBAI_demo_restrict_voigt_exp_20260525_171135/fit_plots/fit_2D_(002)_idx738.png) |
| **175331** | **voigt_exp + area init + narrow gamma + fit_half_width_deg=0.4** | **453** | **0.987** | **0.306** | **0.031** | 0.02 | 0.062 | ✅ good | [plot](ClMBAI_demo_restrict_voigt_exp_20260525_175331/fit_plots/fit_2D_(002)_idx453.png) |
| **175331** | **voigt_exp + area init + narrow gamma + fit_half_width_deg=0.4** | **738** | **0.986** | **0.382** | **0.032** | 0.02 | 0.062 | ✅ good | [plot](ClMBAI_demo_restrict_voigt_exp_20260525_175331/fit_plots/fit_2D_(002)_idx738.png) |
| **175331** | **voigt_exp + area init + narrow gamma + fit_half_width_deg=0.4** | **828** | **0.984** | **0.425** | **0.034** | 0.02 | 0.062 | ✅ good | [plot](ClMBAI_demo_restrict_voigt_exp_20260525_175331/fit_plots/fit_2D_(002)_idx828.png) |
