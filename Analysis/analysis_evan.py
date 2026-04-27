from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import chi2


@dataclass
class SpeData:
    """Container for one spectrum loaded from an ORTEC/Maestro .Spe file."""

    path: str
    counts: np.ndarray
    bins: np.ndarray
    live_time_s: Optional[float]
    real_time_s: Optional[float]
    metadata: Dict[str, str]
    n_channels_raw: int
    rebin_factor: int


@dataclass
class PeakCandidates:
    """Result of initial candidate-finding pass."""

    candidate_bins: np.ndarray
    candidate_heights: np.ndarray
    candidate_prominences: np.ndarray
    indices: np.ndarray


@dataclass
class LocalPeakFit:
    """Local parametric peak fit using Gaussian + linear background."""

    candidate_bin: float
    fit_center: float
    fit_center_err: float
    sigma: float
    sigma_err: float
    amplitude: float
    amplitude_err: float
    background_intercept: float
    background_slope: float
    chi2_val: float
    ndof: int
    p_value: float
    fit_range: Tuple[int, int]
    success: bool
    covariance: Optional[np.ndarray] = None


def poisson_errors(counts: np.ndarray) -> np.ndarray:
    """Poisson counting uncertainty with floor to avoid divide-by-zero."""
    return np.sqrt(np.maximum(counts, 1.0))


def gaussian_plus_linear(x: np.ndarray, A: float, mu: float, sigma: float, b0: float, b1: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + b0 + b1 * x


def compute_chi2(
    y: np.ndarray, yfit: np.ndarray, yerr: np.ndarray, n_params: int
) -> Tuple[float, int, float]:
    residuals = (y - yfit) / yerr
    chi2_val = float(np.sum(residuals**2))
    ndof = int(len(y) - n_params)
    p_value = float(chi2.sf(chi2_val, ndof)) if ndof > 0 else np.nan
    return chi2_val, ndof, p_value


def rebin_counts(counts: np.ndarray, factor: int = 2) -> np.ndarray:
    """Rebin by summing adjacent channels in groups of `factor`."""
    if factor <= 0:
        raise ValueError("factor must be a positive integer")
    if len(counts) % factor != 0:
        raise ValueError(
            f"Counts length {len(counts)} is not divisible by rebin factor {factor}"
        )
    return counts.reshape(-1, factor).sum(axis=1)


def read_spe(filepath: str, rebin_factor: int = 1) -> SpeData:
    """
    Read counts and metadata from an ORTEC-style .Spe file.

    Args:
      rebin_factor:
        If >1, rebin by summing adjacent channels before returning counts.

    Returns:
      - counts (float array)
      - bins (0..N-1)
      - live_time_s and real_time_s from $MEAS_TIM when present
    """
    if int(rebin_factor) != rebin_factor or rebin_factor < 1:
        raise ValueError("rebin_factor must be an integer >= 1")

    path = Path(filepath)
    with path.open("r", encoding="latin-1") as f:
        lines = [line.strip() for line in f.readlines()]

    metadata: Dict[str, str] = {}
    live_time_s: Optional[float] = None
    real_time_s: Optional[float] = None
    data_start = None

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("$") and i + 1 < len(lines):
            tag = line
            value = lines[i + 1]
            metadata[tag] = value
            if tag == "$MEAS_TIM:":
                parts = value.split()
                if len(parts) >= 2:
                    try:
                        live_time_s = float(parts[0])
                        real_time_s = float(parts[1])
                    except ValueError:
                        live_time_s = None
                        real_time_s = None
            if tag.startswith("$DATA"):
                data_start = i
                break
            i += 2
        else:
            i += 1

    if data_start is None:
        raise ValueError(f"Could not find $DATA section in {filepath}")

    counts = []
    started_numbers = False
    for line in lines[data_start + 1 :]:
        if line.startswith("$") and started_numbers:
            break
        if not line:
            continue

        parts = line.split()

        # First line after $DATA is usually channel bounds ("0 2047")
        if len(parts) == 2 and not started_numbers:
            try:
                int(parts[0])
                int(parts[1])
                continue
            except ValueError:
                pass

        try:
            counts.append(float(parts[0]))
            started_numbers = True
        except Exception:
            continue

    if not counts:
        raise ValueError(f"No counts found in $DATA section for {filepath}")

    counts_arr_raw = np.array(counts, dtype=float)
    if rebin_factor > 1:
        counts_arr = rebin_counts(counts_arr_raw, factor=int(rebin_factor))
    else:
        counts_arr = counts_arr_raw.copy()

    bins = np.arange(len(counts_arr), dtype=float)
    return SpeData(
        path=str(path),
        counts=counts_arr,
        bins=bins,
        live_time_s=live_time_s,
        real_time_s=real_time_s,
        metadata=metadata,
        n_channels_raw=len(counts_arr_raw),
        rebin_factor=int(rebin_factor),
    )


def find_peak_candidates(
    bins: np.ndarray,
    counts: np.ndarray,
    *,
    min_bin: int = 0,
    prominence: Optional[float] = None,
    height: Optional[float] = None,
    distance: int = 20,
    max_peaks: int = 10,
) -> PeakCandidates:
    """
    Initial pass peak detector.

    Returns candidate bins/indices that you can manually choose from before
    local parametric fitting.
    """
    mask = bins >= min_bin
    bins_use = bins[mask]
    counts_use = counts[mask]
    if len(bins_use) == 0:
        raise ValueError(f"No bins remain after min_bin={min_bin}")

    if prominence is None:
        prominence = max(5.0, 0.03 * float(np.max(counts_use)))
    if height is None:
        height = max(5.0, 0.05 * float(np.max(counts_use)))

    idx_local, props = find_peaks(
        counts_use,
        prominence=prominence,
        height=height,
        distance=distance,
    )
    prominences_all = np.asarray(props["prominences"], dtype=float)

    if len(idx_local) == 0:
        return PeakCandidates(
            candidate_bins=np.array([], dtype=float),
            candidate_heights=np.array([], dtype=float),
            candidate_prominences=np.array([], dtype=float),
            indices=np.array([], dtype=int),
        )

    if len(idx_local) > max_peaks:
        order = np.argsort(prominences_all)[::-1][:max_peaks]
        idx_local = idx_local[order]
        prominences_all = prominences_all[order]
        idx_local = np.sort(idx_local)
        sort_order = np.argsort(idx_local)
        idx_local = idx_local[sort_order]
        prominences_all = prominences_all[sort_order]

    candidate_bins = bins_use[idx_local]
    candidate_heights = counts_use[idx_local]
    candidate_prominences = prominences_all

    original_indices = np.where(mask)[0][idx_local]
    return PeakCandidates(
        candidate_bins=np.asarray(candidate_bins, dtype=float),
        candidate_heights=np.asarray(candidate_heights, dtype=float),
        candidate_prominences=np.asarray(candidate_prominences, dtype=float),
        indices=np.asarray(original_indices, dtype=int),
    )


def fit_local_peak(
    bins: np.ndarray,
    counts: np.ndarray,
    candidate_bin: float,
    *,
    fit_half_width: int = 12,
    min_p_value: float = 0.01,
) -> LocalPeakFit:
    """
    Fit one selected candidate with Gaussian + linear background.

    Args:
      candidate_bin: center bin around which a local window is built.
    """
    p = int(np.argmin(np.abs(bins - candidate_bin)))
    left = max(0, p - fit_half_width)
    right = min(len(bins) - 1, p + fit_half_width)

    xfit = bins[left : right + 1]
    yfit = counts[left : right + 1]
    yerr = poisson_errors(yfit)

    A0 = float(max(yfit) - np.median(yfit))
    mu0 = float(bins[p])
    sigma0 = float(max(1.5, fit_half_width / 4))
    b00 = float(np.median(yfit))
    b10 = 0.0

    lower = [0.0, float(xfit.min()), 0.5, -np.inf, -np.inf]
    upper = [np.inf, float(xfit.max()), float(fit_half_width), np.inf, np.inf]

    success = True
    try:
        popt, pcov = curve_fit(
            gaussian_plus_linear,
            xfit,
            yfit,
            p0=[A0, mu0, sigma0, b00, b10],
            sigma=yerr,
            absolute_sigma=True,
            bounds=(lower, upper),
            maxfev=20000,
        )
        model = gaussian_plus_linear(xfit, *popt)
        chi2_val, ndof, p_value = compute_chi2(yfit, model, yerr, n_params=5)
        perr = np.sqrt(np.diag(pcov))
        if np.isfinite(p_value) and p_value < min_p_value:
            success = False
    except Exception:
        success = False
        popt = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=float)
        perr = np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=float)
        pcov = None
        chi2_val = np.nan
        ndof = len(xfit) - 5
        p_value = np.nan

    return LocalPeakFit(
        candidate_bin=float(candidate_bin),
        fit_center=float(popt[1]),
        fit_center_err=float(perr[1]),
        sigma=float(popt[2]),
        sigma_err=float(perr[2]),
        amplitude=float(popt[0]),
        amplitude_err=float(perr[0]),
        background_intercept=float(popt[3]),
        background_slope=float(popt[4]),
        chi2_val=float(chi2_val),
        ndof=int(ndof),
        p_value=float(p_value),
        fit_range=(int(xfit.min()), int(xfit.max())),
        success=bool(success),
        covariance=pcov,
    )


def peak_counts_from_fit(
    fit: LocalPeakFit,
    *,
    method: str = "gaussian_area",
    n_sigma: float = 2.5,
) -> Tuple[float, float]:
    """
    Estimate net counts in the fitted peak (background-subtracted).

    Methods:
      - "gaussian_area": net = A * sigma * sqrt(2*pi)
      - "windowed_model_subtraction": integrate model peak over mu +/- n_sigma
    """
    # A fit can be numerically valid but flagged unsuccessful by a strict p-value
    # threshold. For rate extraction, accept finite fitted parameters.
    if not np.isfinite(fit.amplitude) or not np.isfinite(fit.sigma):
        return np.nan, np.nan

    if method == "gaussian_area":
        area = fit.amplitude * fit.sigma * np.sqrt(2.0 * np.pi)

        area_err = np.nan
        if fit.covariance is not None and fit.covariance.shape == (5, 5):
            var_A = fit.covariance[0, 0]
            var_s = fit.covariance[2, 2]
            cov_As = fit.covariance[0, 2]
            coeff = 2.0 * np.pi
            var_area = coeff * (
                (fit.sigma**2) * var_A
                + (fit.amplitude**2) * var_s
                + 2.0 * fit.amplitude * fit.sigma * cov_As
            )
            area_err = float(np.sqrt(max(var_area, 0.0)))
        return float(area), float(area_err)

    if method == "windowed_model_subtraction":
        x0 = fit.fit_center - n_sigma * fit.sigma
        x1 = fit.fit_center + n_sigma * fit.sigma
        # Gaussian-only integral on [x0, x1]
        x = np.linspace(x0, x1, 400)
        y_peak = fit.amplitude * np.exp(-0.5 * ((x - fit.fit_center) / fit.sigma) ** 2)
        area = np.trapz(y_peak, x)
        return float(area), np.nan

    raise ValueError("method must be 'gaussian_area' or 'windowed_model_subtraction'")


def peak_rate_from_fit(
    fit: LocalPeakFit,
    live_time_s: float,
    *,
    method: str = "gaussian_area",
    n_sigma: float = 2.5,
) -> Tuple[float, float]:
    """
    Convert net peak counts to rate [counts/s].
    """
    if live_time_s <= 0:
        raise ValueError("live_time_s must be > 0")
    counts, counts_err = peak_counts_from_fit(fit, method=method, n_sigma=n_sigma)
    if not np.isfinite(counts):
        return np.nan, np.nan
    rate = counts / live_time_s
    rate_err = counts_err / live_time_s if np.isfinite(counts_err) else np.nan
    return float(rate), float(rate_err)
