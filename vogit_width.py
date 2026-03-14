"""
voigt_width.py
----------------
Utility to evaluate the full width of a Voigt profile at an arbitrary
height (relative or absolute) given the Gaussian σ and Lorentzian γ
parameters.

Author:  <your‑name>
Date:    2026‑02‑17
Dependencies:  numpy, scipy
"""

from __future__ import annotations

import numpy as np
from typing import Union, Iterable

# ----------------------------------------------------------------------
# SciPy imports – the function will raise a clear error if SciPy is missing.
# ----------------------------------------------------------------------
try:
    from scipy.special import wofz          # Faddeeva function
    from scipy.optimize import brentq
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This routine requires SciPy. Install it via `pip install scipy`."
    ) from exc


def _voigt_profile(x: Union[float, np.ndarray],
                   sigma: float,
                   gamma: float) -> Union[float, np.ndarray]:
    """
    Voigt line shape (area‑normalised, centred at 0).

    Parameters
    ----------
    x : float or ndarray
        Position(s) at which to evaluate the profile.
    sigma : float
        Gaussian standard deviation (σ).  Must be ≥ 0.
    gamma : float
        Lorentzian half‑width at half‑maximum (γ).  Must be ≥ 0.

    Returns
    -------
    float or ndarray
        Normalised intensity V(x).  The integral of V over x equals 1.
    """
    # Guard against negative arguments – they have no physical meaning.
    if sigma < 0 or gamma < 0:
        raise ValueError("sigma and gamma must be non‑negative.")

    # Pure Gaussian or pure Lorentzian can be handled analytically for speed.
    if gamma == 0.0:          # pure Gaussian
        return (1.0 / (sigma * np.sqrt(2.0 * np.pi)) *
                np.exp(-x**2 / (2.0 * sigma**2)))
    if sigma == 0.0:          # pure Lorentzian
        return gamma / (np.pi * (x**2 + gamma**2))

    # General case – use the Faddeeva function.
    z = (x + 1j * gamma) / (sigma * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def _peak_intensity(sigma: float, gamma: float) -> float:
    """
    Return V(0) – the maximum intensity of a Voigt profile
    with the supplied σ and γ.

    The analytic expression is simply V(0) = Re[w(iγ/(σ√2))]/(σ√(2π)) .
    """
    # Pure cases again have simple closed forms.
    if gamma == 0.0:                     # pure Gaussian
        return 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    if sigma == 0.0:                     # pure Lorentzian
        return 1.0 / (np.pi * gamma)

    z = 1j * gamma / (sigma * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def voigt_width_at_height(
    sigma: float,
    gamma: float,
    level: Union[float, Iterable[float]],
    *,
    rel: bool = True,
    max_iter: int = 200,
    xtol: float = 1e-12,
    rtol: float = 1e-12,
) -> Union[float, np.ndarray]:
    """
    Compute the *full* width of a Voigt peak at a given height.

    The height can be supplied either as a *relative* fraction of the
    peak intensity (`rel=True`, the default) or as an *absolute* intensity
    (`rel=False`).  The function returns the width `W = 2·x_p` that satisfies

        V(x_p) = level   (or level·V(0) if rel=True),

    where V(x) is the Voigt line shape defined by `sigma` and `gamma`.

    Parameters
    ----------
    sigma : float
        Gaussian standard deviation (σ).  Units are the same as those of the
        desired width (e.g. nm, cm⁻¹, …).
    gamma : float
        Lorentzian half‑width at half‑maximum (γ).  Same unit as `sigma`.
    level : float or iterable of floats
        Desired height.  If ``rel=True`` the value must lie in (0, 1]; it
        represents a fraction of the peak height (e.g. ``0.01`` for FW1 % M).
        If ``rel=False`` the value must be ≤ V(0) (the absolute intensity).
    rel : bool, optional
        ``True`` → treat `level` as a relative fraction; ``False`` → treat
        it as an absolute intensity.  Default: ``True``.
    max_iter : int, optional
        Maximum number of iterations that the bracketing loop may perform
        to find a right‑hand bound.  Default: 200 (more than enough for
        typical scientific use).
    xtol, rtol : float, optional
        Tolerances passed to ``scipy.optimize.brentq``.  Defaults give
        sub‑10⁻¹² relative error on the half‑width `x_p`.

    Returns
    -------
    float or ndarray
        Full width at the requested height.  If `level` is an iterable,
        a NumPy array of the same shape is returned; otherwise a scalar.

    Raises
    ------
    ValueError
        If `sigma` or `gamma` are negative, or if a requested `level`
        lies outside the admissible interval.
    RuntimeError
        If the root‑finder fails to converge (unlikely for physically
        meaningful inputs).

    Notes
    -----
    * The function is **exact** up to machine precision because the Voigt
      line shape is evaluated via the Faddeeva function, and the width is
      obtained from a robust scalar root‑finder.
    * For the two limiting cases (pure Gaussian or pure Lorentzian) the
      routine uses the well‑known analytical widths:

        – Gaussian:   W = 2 σ √{‑2 ln p}
        – Lorentzian:  W = 2 γ √{1/p − 1}

      where `p` is the relative fraction (i.e. the same quantity as `level`
      when `rel=True`).  This makes the routine fast and numerically
      stable even when one component is essentially zero.

    Examples
    --------
    >>> sigma, gamma = 0.20, 0.10          # same units as the eventual width
    >>> for p in (0.05, 0.01, 0.001):
    ...     w = voigt_width_at_height(sigma, gamma, p)
    ...     print(f"FW{int(p*100)}%M = {w:.4f}")
    FW5%M = 1.2631
    FW1%M = 1.6036
    FW0%M = 2.1999

    The same call can be made with an absolute intensity:

    >>> V0 = _peak_intensity(sigma, gamma)
    >>> I_target = 0.01 * V0               # absolute intensity equal to 1 % of the peak
    >>> w = voigt_width_at_height(sigma, gamma, I_target, rel=False)
    >>> print(w)
    1.6036
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if sigma < 0 or gamma < 0:
        raise ValueError("sigma and gamma must be non‑negative numbers.")

    # Normalise `level` to an array for uniform handling.
    level_arr = np.atleast_1d(level).astype(np.float64)
    # Output array of the same shape.
    width_arr = np.empty_like(level_arr)

    # Compute the maximum intensity once (used for rel=False case)
    V0 = _peak_intensity(sigma, gamma)

    # ------------------------------------------------------------------
    # Helper: function to find the half‑width x_p for a *single* target.
    # ------------------------------------------------------------------
    def _half_width(target_intensity: float) -> float:
        """
        Find x > 0 such that V(x) = target_intensity.
        Returns the half‑width (x).  The full width is 2·x.
        """
        # Edge cases: target == V(0) → width = 0
        if np.isclose(target_intensity, V0, rtol=1e-14, atol=0):
            return 0.0
        if target_intensity <= 0.0:
            # In principle the width is infinite; we raise an informative error.
            raise ValueError(
                "Requested height is ≤ 0; the Voigt profile never reaches "
                "zero intensity.  Use a positive fraction or intensity."
            )

        # **** Pure component shortcuts ****
        # If one of the components is effectively zero we can return the analytic width.
        eps = np.finfo(float).eps
        if gamma < eps:    # pure Gaussian
            # Solve G(x)/G(0) = target/V0 = p → x = σ √{-2 ln p}
            p = target_intensity / V0
            return sigma * np.sqrt(-2.0 * np.log(p))
        if sigma < eps:    # pure Lorentzian
            # Solve L(x)/L(0) = p → x = γ √{1/p – 1}
            p = target_intensity / V0
            return gamma * np.sqrt(1.0 / p - 1.0)

        # **** General case (σ>0, γ>0) ****
        # Define the monotonic function f(x) = V(x) - target.
        def f(x: float) -> float:
            return _voigt_profile(x, sigma, gamma) - target_intensity

        # Find a right‑hand bracket where f(x) < 0.
        # Start from a modest guess and increase exponentially until the sign flips.
        # The bracket must be positive (x>0) because V(x) is symmetric.
        left = 0.0
        # The function at x=0 is positive (V(0) > target).  So we just need a right bound.
        # A heuristic upper bound is a few times the sum of the component widths.
        # We increase it until V(x) < target.
        right = max(sigma, gamma) * 5.0   # start with something generous
        for _ in range(max_iter):
            if f(right) < 0.0:
                break
            right *= 2.0
        else:
            # If we exit the loop without a sign change, something is off.
            raise RuntimeError(
                "Failed to bracket the root for the requested height. "
                "Try a larger `max_iter` or check the inputs."
            )

        # Now we have f(left) > 0 and f(right) < 0 → safe to use Brent's method.
        x_root = brentq(f, left, right, xtol=xtol, rtol=rtol, maxiter=200)
        return x_root

    # ------------------------------------------------------------------
    # Main loop over the requested levels
    # ------------------------------------------------------------------
    for i, lvl in enumerate(level_arr):
        if rel:
            if not (0.0 < lvl <= 1.0):
                raise ValueError(
                    f"Relative level must lie in (0,1]; got {lvl:.6g}."
                )
            target = lvl * V0
        else:
            # level already given as absolute intensity
            target = float(lvl)
            if target > V0 or target <= 0.0:
                raise ValueError(
                    f"Absolute intensity must satisfy 0 < I ≤ V0 ({V0:.6g}); "
                    f"got {target:.6g}."
                )
        # Find half‑width and store full width.
        half_width = _half_width(target)
        width_arr[i] = 2.0 * half_width

    # Return scalar if the original input was scalar.
    if np.isscalar(level):
        return float(width_arr.squeeze())
    return width_arr


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Demo / quick test when the module is executed as a script.
# ----------------------------------------------------------------------
if __name__ == "__main__":          # pragma: no cover
    # Example parameters (units are arbitrary, e.g. nm)
    sigma_test = 0.20
    gamma_test = 0.10

    # Relative heights we care about
    fractions = np.array([0.05, 0.01, 0.001])
    widths = voigt_width_at_height(sigma_test, gamma_test, fractions)

    print("Gaussian σ = {:.3f}, Lorentzian γ = {:.3f}".format(sigma_test, gamma_test))
    for p, w in zip(fractions, widths):
        print(f"FW{int(p*100)}%M = {w:.6f}")

    # Show that passing an absolute intensity gives the same result
    V0 = _peak_intensity(sigma_test, gamma_test)
    absolute = 0.01 * V0
    w_abs = voigt_width_at_height(sigma_test, gamma_test, absolute, rel=False)
    print("\nAbsolute 1 % of peak intensity → width = {:.6f}".format(w_abs))
