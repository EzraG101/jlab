import os
import re
import math
import glob
import json
import copy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import chi2
# from scipy.odr import ODR, Model, RealData

# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 13,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "figure.figsize": (10, 6),
    "axes.grid": False,
})

# Colorblind-friendly palette
CBLUE = "#0072B2"
CORANGE = "#E69F00"
CGREEN = "#009E73"
CRED = "#D55E00"
CPURPLE = "#CC79A7"
CBLACK = "#000000"
CGRAY = "#7F7F7F"

# ============================================================
# Systematic-study configuration
# ============================================================

FACTOR = 16
UNIVERSAL_CUTOFF = 120 // FACTOR
HALF_WIDTH_FIT = 88 // FACTOR
PROMINENCE = 20 * FACTOR

SYSTEMATIC_CONFIG = {
    "enabled": False,
    "fit_half_width_values": [10, 12, 14],
    "p_value_cut_values": [0.05, 0],
    "low_bin_cut_shifts": [-10, 0, 10],
    "cs_peak_rules": ["closest_to_max", "highest_count"],
}

# ============================================================
# Physics constants
# ============================================================

CS137_ENERGY_KEV = 661.657
ELECTRON_REST_ENERGY_KEV = 510.99895

# ============================================================
# User-configurable known source energies (keV)
# ============================================================

KNOWN_PEAKS_KEV = {
    "Na22": np.array([511.0]),
    "Ba133": np.array([81.0, 356.01, 661.657]), # last is Cs137 but will use for Ba133 recoil
    "Cs137": np.array([661.657]),
}

# ============================================================
# Optional low-bin cutoffs to suppress low-energy noise
# Values are in REBINNED bins 
# ============================================================

LOW_BIN_CUTOFFS = {
    ("03-05", "Ba133", "scatter"): UNIVERSAL_CUTOFF,
    ("03-05", "Ba133", "recoil"): UNIVERSAL_CUTOFF,
    ("03-05", "Na22", "scatter"): UNIVERSAL_CUTOFF,
    ("03-05", "Na22", "recoil"): UNIVERSAL_CUTOFF,
    ("03-10", "Ba133", "scatter"): UNIVERSAL_CUTOFF,
    ("03-10", "Ba133", "recoil"): UNIVERSAL_CUTOFF,
    ("03-10", "Na22", "scatter"): UNIVERSAL_CUTOFF,
    ("03-10", "Na22", "recoil"): UNIVERSAL_CUTOFF,
    ("03-12", "Ba133", "scatter"): UNIVERSAL_CUTOFF,
    ("03-12", "Ba133", "recoil"): UNIVERSAL_CUTOFF,
    ("03-12", "Na22", "scatter"): UNIVERSAL_CUTOFF,
    ("03-12", "Na22", "recoil"): UNIVERSAL_CUTOFF,
}

# ============================================================
# Data containers
# ============================================================

@dataclass
class Spectrum:
    filepath: str
    filename: str
    date: str
    source: str
    spec_type: str
    angle: Optional[float]
    counts_raw: np.ndarray
    counts: np.ndarray
    bins_raw: np.ndarray
    bins: np.ndarray

@dataclass
class PeakFitResult:
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

@dataclass
class CalibrationResult:
    date: str
    spec_type: str
    slope: float
    intercept: float
    slope_err: float
    intercept_err: float
    cov: np.ndarray
    chi2_val: float
    ndof: int
    p_value: float
    used_bins: np.ndarray
    used_bin_errs: np.ndarray
    used_energies: np.ndarray

@dataclass
class CsPeakEnergy:
    date: str
    spec_type: str
    angle: float
    peak_bin: float
    peak_bin_err: float
    energy_keV: float
    energy_err_keV: float
    chi2_val: float
    p_value: float
    note: str = ""

# ============================================================
# Utility helpers
# ============================================================

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s.strip())

def poisson_errors(counts: np.ndarray):
    return np.sqrt(np.maximum(counts, 1.0))

def get_low_bin_cutoff(
    date: str,
    source: str,
    spec_type: str,
    angle: Optional[float] = None,
    low_bin_cutoffs: Optional[Dict[Tuple[str, str, str], int]] = None
) -> int:
    if low_bin_cutoffs is None:
        low_bin_cutoffs = LOW_BIN_CUTOFFS
    return low_bin_cutoffs.get((date, source, spec_type), 0)

def mean_with_propagated_uncertainty(values, errors):
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    N = len(values)
    if N == 0:
        return np.nan, np.nan
    return np.mean(values), np.sqrt(np.sum(errors**2)) / N

def sample_sem(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan
    return np.std(values, ddof=1) / np.sqrt(len(values))

def systematic_uncertainty_from_variations(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    return np.std(values, ddof=1)

def combine_stat_and_sys(stat_err, sys_err):
    if not np.isfinite(stat_err) and not np.isfinite(sys_err):
        return np.nan
    if not np.isfinite(stat_err):
        return sys_err
    if not np.isfinite(sys_err):
        return stat_err
    return np.sqrt(stat_err**2 + sys_err**2)

def inverse_with_error(x, xerr):
    y = 1.0 / x
    yerr = np.abs(xerr / (x**2))
    return y, yerr

def one_minus_cos_theta(theta_deg):
    return 1.0 - np.cos(np.deg2rad(theta_deg))

def inv_one_minus_cos_theta(theta_deg):
    x = one_minus_cos_theta(theta_deg)
    return 1.0 / x

# ============================================================
# File reading and parsing
# ============================================================

def parse_spe_filename(filename: str):
    base = os.path.basename(filename)
    pattern_no_angle = r"^(?P<date>\d{2}-\d{2})-(?P<source>Na22|Ba133|Cs137)-(?P<stype>scatter|recoil)\.Spe$"
    pattern_angle = r"^(?P<date>\d{2}-\d{2})-(?P<source>Na22|Ba133|Cs137)-(?P<stype>scatter|recoil)-(?P<angle>-?\d+(\.\d+)?)\.Spe$"

    m = re.match(pattern_angle, base, re.IGNORECASE)
    if m:
        return m.group("date"), m.group("source"), m.group("stype").lower(), float(m.group("angle"))

    m = re.match(pattern_no_angle, base, re.IGNORECASE)
    if m:
        return m.group("date"), m.group("source"), m.group("stype").lower(), None

    raise ValueError(f"Filename does not match expected format: {filename}")

def read_spe_file(filepath: str) -> np.ndarray:
    with open(filepath, "r", encoding="latin-1") as f:
        lines = [line.strip() for line in f.readlines()]

    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("$DATA"):
            data_start = i
            break

    if data_start is None:
        raise ValueError(f"Could not find $DATA section in {filepath}")

    counts = []
    started_numbers = False
    for line in lines[data_start + 1:]:
        if line.startswith("$") and started_numbers:
            break
        if not line:
            continue

        parts = line.split()
        if len(parts) == 2 and not started_numbers:
            try:
                int(parts[0]); int(parts[1])
                continue
            except ValueError:
                pass

        try:
            counts.append(float(parts[0]))
            started_numbers = True
        except Exception:
            continue

    if len(counts) == 0:
        raise ValueError(f"No counts found in $DATA section for {filepath}")

    return np.array(counts, dtype=float)

def rebin_counts(counts: np.ndarray, factor: int = 16) -> np.ndarray:
    if len(counts) % factor != 0:
        raise ValueError(f"Counts length {len(counts)} not divisible by rebin factor {factor}")
    return counts.reshape(-1, factor).sum(axis=1)

def load_all_spectra(data_dir: str, factor=16) -> List[Spectrum]:
    spectra = []
    for filepath in sorted(glob.glob(os.path.join(data_dir, "*.Spe"))):
        date, source, spec_type, angle = parse_spe_filename(os.path.basename(filepath))
        counts_raw = read_spe_file(filepath)
        bins_raw = np.arange(len(counts_raw), dtype=float)

        if len(counts_raw) == 2048:
            counts = rebin_counts(counts_raw, factor=factor)
        elif len(counts_raw) == 1024:
            counts = counts_raw.copy()
        else:
            raise ValueError(f"Unexpected number of channels in {filepath}: {len(counts_raw)}")

        bins = np.arange(len(counts), dtype=float)

        spectra.append(Spectrum(
            filepath=filepath,
            filename=os.path.basename(filepath),
            date=date,
            source=source,
            spec_type=spec_type,
            angle=angle,
            counts_raw=counts_raw,
            counts=counts,
            bins_raw=bins_raw,
            bins=bins
        ))
    return spectra

# ============================================================
# Models and fitting
# ============================================================

def quadratic(x, a, b, c):
    return a * x * x + b * x + c

def gaussian_plus_linear(x, A, mu, sigma, b0, b1):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + b0 + b1 * x

def compute_chi2(y, yfit, yerr, n_params):
    residuals = (y - yfit) / yerr
    chi2_val = np.sum(residuals**2)
    ndof = len(y) - n_params
    p_value = chi2.sf(chi2_val, ndof) if ndof > 0 else np.nan
    return chi2_val, ndof, p_value

def weighted_linear(x, m, b):
    return m * x + b

# def fit_linear_odr(x, y, sx):
    def f_odr(beta, x_):
        return beta[0] * x_ + beta[1]

    model = Model(f_odr)
    data = RealData(x, y, sx=sx)
    beta0 = np.polyfit(x, y, 1)

    odr = ODR(data, model, beta0=beta0)
    out = odr.run()

    popt = np.array(out.beta, dtype=float)
    pcov = np.array(out.cov_beta, dtype=float) * out.res_var

    yfit = weighted_linear(x, *popt)
    yerr_eff = np.maximum(np.abs(popt[0]) * sx, 1e-12)
    chi2_val, ndof, p_value = compute_chi2(y, yfit, yerr_eff, 2)
    return popt, pcov, chi2_val, ndof, p_value

# ============================================================
# Peak finding and fitting
# ============================================================

def find_and_fit_peaks(
    bins: np.ndarray,
    counts: np.ndarray,
    title_prefix: str = "",
    output_dir: str = "better-plots",
    prominence: Optional[float] = None,
    height: Optional[float] = None,
    distance: int = 20,
    fit_half_width: int = 12,
    max_peaks: int = 10,
    min_bin: int = 0,
    calibration: Optional[CalibrationResult] = None,
    min_p_value: float = 0.05,
    save_plots: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    mask = bins >= min_bin
    bins_use = bins[mask]
    counts_use = counts[mask]

    if len(bins_use) == 0:
        raise ValueError(f"No bins remain after applying min_bin={min_bin}")

    if calibration is not None:
        xplot_use = calibration.slope * bins_use + calibration.intercept
        xlabel = "Energy [keV]"
    else:
        xplot_use = bins_use
        xlabel = "Bin"

    if prominence is None:
        prominence = max(5.0, 0.03 * np.max(counts_use))
    if height is None:
        height = max(5.0, 0.05 * np.max(counts_use))

    peak_indices_local, properties = find_peaks(
        counts_use,
        prominence=prominence,
        height=height,
        distance=distance
    )

    if len(peak_indices_local) > max_peaks:
        order = np.argsort(properties["prominences"])[::-1][:max_peaks]
        peak_indices_local = np.sort(peak_indices_local[order])

    if save_plots:
        fig, ax = plt.subplots()
        bar_width = np.diff(xplot_use).mean() if len(xplot_use) > 1 else 1.0
        ax.bar(xplot_use, counts_use, width=bar_width, color=CBLUE, edgecolor=None, linewidth=0)

        if len(peak_indices_local) > 0:
            ax.plot(xplot_use[peak_indices_local], counts_use[peak_indices_local], 'o',
                    color=CRED, markersize=8, label="Peak candidates")
            for p in peak_indices_local:
                left = max(0, p - fit_half_width)
                right = min(len(bins_use) - 1, p + fit_half_width)
                ax.axvspan(xplot_use[left], xplot_use[right], color=CORANGE, alpha=0.15)

        if min_bin > 0:
            ax.text(
                0.02, 0.95, f"Low-bin cutoff applied: bin ≥ {min_bin}",
                transform=ax.transAxes, ha="left", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        ax.set_title(f"{title_prefix} Peak candidates")
        if len(peak_indices_local) > 0:
            ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{sanitize_filename(title_prefix)}_peak_candidates.png"), dpi=200)
        plt.close()

    fit_results = []

    for i, p in enumerate(peak_indices_local):
        left = max(0, p - fit_half_width)
        right = min(len(bins_use) - 1, p + fit_half_width)

        xfit = bins_use[left:right+1]
        yfit = counts_use[left:right+1]
        yerr = poisson_errors(yfit)

        xfit_plot = calibration.slope * xfit + calibration.intercept if calibration is not None else xfit

        A0 = max(yfit) - np.median(yfit)
        mu0 = bins_use[p]
        sigma0 = max(1.5, fit_half_width / 4)
        b00 = np.median(yfit)
        b10 = 0.0

        lower = [0, xfit.min(), 0.5, -np.inf, -np.inf]
        upper = [np.inf, xfit.max(), fit_half_width, np.inf, np.inf]

        success = True
        rejected_for_p = False
        try:
            popt, pcov = curve_fit(
                gaussian_plus_linear,
                xfit, yfit,
                p0=[A0, mu0, sigma0, b00, b10],
                sigma=yerr,
                absolute_sigma=True,
                bounds=(lower, upper),
                maxfev=20000
            )
            model = gaussian_plus_linear(xfit, *popt)
            chi2_val, ndof, p_value = compute_chi2(yfit, model, yerr, 5)
            perr = np.sqrt(np.diag(pcov))

            if np.isfinite(p_value) and p_value < min_p_value:
                success = False
                rejected_for_p = True

        except Exception:
            success = False
            popt = [np.nan] * 5
            perr = [np.nan] * 5
            pcov = None
            chi2_val = np.nan
            ndof = len(xfit) - 5
            p_value = np.nan

        result = PeakFitResult(
            candidate_bin=bins_use[p],
            fit_center=popt[1],
            fit_center_err=perr[1],
            sigma=popt[2],
            sigma_err=perr[2],
            amplitude=popt[0],
            amplitude_err=perr[0],
            background_intercept=popt[3],
            background_slope=popt[4],
            chi2_val=chi2_val,
            ndof=ndof,
            p_value=p_value,
            fit_range=(int(xfit.min()), int(xfit.max())),
            success=success,
            covariance=pcov
        )
        fit_results.append(result)

        if save_plots:
            fig, ax = plt.subplots()
            bar_width = np.diff(xfit_plot).mean() if len(xfit_plot) > 1 else 1.0
            ax.bar(xfit_plot, yfit, width=bar_width, color=CBLUE, edgecolor=None, linewidth=0, label="Data")

            if np.all(np.isfinite(popt)):
                xdense = np.linspace(xfit.min(), xfit.max(), 400)
                ydense = gaussian_plus_linear(xdense, *popt)

                if calibration is not None:
                    xdense_plot = calibration.slope * xdense + calibration.intercept
                    peak_plot = calibration.slope * popt[1] + calibration.intercept
                    peak_plot_err = abs(calibration.slope) * perr[1]
                    unit = "keV"
                else:
                    xdense_plot = xdense
                    peak_plot = popt[1]
                    peak_plot_err = perr[1]
                    unit = "bins"

                ax.plot(xdense_plot, ydense, color=CRED, lw=2.5, label="Gaussian + linear fit")
                ax.axvline(peak_plot, color=CGREEN, ls="--", lw=2,
                           label=f"Peak = {peak_plot:.2f} ± {peak_plot_err:.2f} {unit}")

                status = "accepted" if success else "rejected"
                if rejected_for_p:
                    status += f" ($p<{min_p_value}$)"

                textbox = (
                    f"$\\mu$ = {peak_plot:.2f} ± {peak_plot_err:.2f} {unit}\n"
                    f"$\\sigma$ = {popt[2]:.2f} ± {perr[2]:.2f} bins\n"
                    f"$\\chi^2$/ndof = {chi2_val:.2f}/{ndof}\n"
                    f"$p$ = {p_value:.3f}\n"
                    f"{status}"
                )
                ax.text(
                    0.98, 0.95, textbox, transform=ax.transAxes,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
                )
            else:
                ax.text(
                    0.98, 0.95, "Fit failed", transform=ax.transAxes,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
                )

            ax.set_xlabel("Energy [keV]" if calibration is not None else "Bin")
            ax.set_ylabel("Counts")
            ax.set_title(f"{title_prefix} candidate {i+1} local fit")
            ax.legend(loc="best")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{sanitize_filename(title_prefix)}_candidate_{i+1}_fit.png"), dpi=200)
            plt.close()

    if save_plots:
        fig, ax = plt.subplots()
        bar_width = np.diff(xplot_use).mean() if len(xplot_use) > 1 else 1.0
        ax.bar(xplot_use, counts_use, width=bar_width, color=CBLUE, edgecolor=None, linewidth=0, label="Histogram")

        valid_results = sorted([r for r in fit_results if r.success], key=lambda r: r.fit_center)
        for i, r in enumerate(valid_results):
            xpeak = calibration.slope * r.fit_center + calibration.intercept if calibration is not None else r.fit_center
            yval = np.interp(r.fit_center, bins_use, counts_use)
            ax.axvline(xpeak, color=CRED, lw=2)
            ax.plot(xpeak, yval, 'o', color=CRED, markersize=8)
            ax.text(
                xpeak, yval + 0.03 * np.max(counts_use), f"{i}",
                color=CBLACK, ha="center", va="bottom", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none")
            )

        if min_bin > 0:
            ax.text(
                0.02, 0.95, f"Low-bin cutoff applied: bin ≥ {min_bin}",
                transform=ax.transAxes, ha="left", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
            )

        ax.set_xlabel("Energy [keV]" if calibration is not None else "Bin")
        ax.set_ylabel("Counts")
        ax.set_title(f"{title_prefix} fitted peaks")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{sanitize_filename(title_prefix)}_fitted_peaks.png"), dpi=200)
        plt.close()

    return fit_results

# ============================================================
# Manual matching for calibration peaks
# ============================================================

def manual_match_calibration_peaks(source, fit_results, known_energies):
    valid = sorted([r for r in fit_results if r.success], key=lambda r: r.fit_center)

    if len(valid) == 0:
        raise ValueError(f"No successful fitted peaks available for {source}")

    print("\n" + "=" * 70)
    print(f"Manual peak matching for {source}")
    print("=" * 70)

    print("\nSuccessful fitted peaks:")
    for i, r in enumerate(valid):
        print(
            f"[{i}] bin = {r.fit_center:.2f} ± {r.fit_center_err:.2f}, "
            f"sigma = {r.sigma:.2f} ± {r.sigma_err:.2f}, "
            f"amplitude = {r.amplitude:.1f}, "
            f"chi2/ndof = {r.chi2_val:.2f}/{r.ndof}, p = {r.p_value:.3f}"
        )

    print("\nKnown energies for this source:")
    for i, E in enumerate(known_energies):
        print(f"[{i}] {E:.3f} keV")

    print("\nEnter matches as:")
    print("    peak_index energy_index")
    print("one per line. Press Enter on blank line when done.")

    selected_peak_bins = []
    selected_peak_bin_errs = []
    selected_energies = []
    used_peak_indices = set()
    used_energy_indices = set()

    while True:
        entry = input("match> ").strip()
        if entry == "":
            break

        parts = entry.split()
        if len(parts) != 2:
            print("Please enter exactly two integers: peak_index energy_index")
            continue

        try:
            pidx = int(parts[0])
            eidx = int(parts[1])
        except ValueError:
            print("Indices must be integers.")
            continue

        if pidx < 0 or pidx >= len(valid):
            print("Invalid peak index.")
            continue
        if eidx < 0 or eidx >= len(known_energies):
            print("Invalid energy index.")
            continue
        if pidx in used_peak_indices:
            print("That peak index has already been used.")
            continue
        if eidx in used_energy_indices:
            print("That energy index has already been used.")
            continue

        used_peak_indices.add(pidx)
        used_energy_indices.add(eidx)
        selected_peak_bins.append(valid[pidx].fit_center)
        selected_peak_bin_errs.append(valid[pidx].fit_center_err)
        selected_energies.append(float(known_energies[eidx]))

        print(
            f"Accepted: peak {pidx} -> {valid[pidx].fit_center:.2f} ± "
            f"{valid[pidx].fit_center_err:.2f} bins matched to {known_energies[eidx]:.3f} keV"
        )

    if len(selected_peak_bins) < 1:
        raise ValueError(f"Need at least 1 matched peak for {source}")

    peak_bins = np.array(selected_peak_bins, dtype=float)
    peak_bin_errs = np.array(selected_peak_bin_errs, dtype=float)
    energies = np.array(selected_energies, dtype=float)
    order = np.argsort(peak_bins)
    return peak_bins[order], peak_bin_errs[order], energies[order]

def save_manual_matches(match_file, key, peak_bins, peak_bin_errs, energies):
    if os.path.exists(match_file):
        with open(match_file, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data[key] = {
        "peak_bins": np.asarray(peak_bins, dtype=float).tolist(),
        "peak_bin_errs": np.asarray(peak_bin_errs, dtype=float).tolist(),
        "energies": np.asarray(energies, dtype=float).tolist(),
    }
    with open(match_file, "w") as f:
        json.dump(data, f, indent=2)

def load_manual_matches(match_file, key):
    if not os.path.exists(match_file):
        raise FileNotFoundError(f"Manual match file not found: {match_file}")
    with open(match_file, "r") as f:
        data = json.load(f)
    if key not in data:
        raise KeyError(f"No saved manual matches found for key: {key}")

    entry = data[key]
    peak_bins = np.array(entry["peak_bins"], dtype=float)
    peak_bin_errs = np.array(entry["peak_bin_errs"], dtype=float)
    energies = np.array(entry["energies"], dtype=float)

    if not (len(peak_bins) == len(peak_bin_errs) == len(energies)):
        raise ValueError(f"Saved match data for {key} has inconsistent array lengths")
    if len(peak_bins) < 1:
        raise ValueError(f"Saved match data for {key} has fewer than 1 matched peak")

    return peak_bins, peak_bin_errs, energies

def get_manual_matches(match_file, key, source, fit_results, known_energies, force_rematch=False):
    if (not force_rematch) and os.path.exists(match_file):
        try:
            peak_bins, peak_bin_errs, energies = load_manual_matches(match_file, key)
            print(f"[INFO] Loaded saved manual matches for {key}")
            return peak_bins, peak_bin_errs, energies
        except KeyError:
            pass
        except Exception as e:
            print(f"[WARN] Could not load saved matches for {key}: {e}")

    print(f"[INFO] No saved matches for {key}; entering interactive matching.")
    peak_bins, peak_bin_errs, energies = manual_match_calibration_peaks(source, fit_results, known_energies)
    save_manual_matches(match_file, key, peak_bins, peak_bin_errs, energies)
    print(f"[INFO] Saved manual matches for {key}")
    return peak_bins, peak_bin_errs, energies

# ============================================================
# Calibration
# ============================================================

def calibrate_day_type(
    date: str,
    spec_type: str,
    source_peak_results: Dict[str, List[PeakFitResult]],
    output_dir: str = "better-plots",
    known_peaks_dict: Dict[str, np.ndarray] = KNOWN_PEAKS_KEV,
    interactive: bool = True,
    match_file: str = "manual_matches.json",
    force_rematch: bool = False
) -> CalibrationResult:
    bins_list = []
    bin_errs_list = []
    energies_list = []

    print("\n" + "#" * 80)
    print(f"Calibration setup for date={date}, type={spec_type}")
    print("#" * 80)

    for source in ["Ba133", "Na22"]:
        if source not in source_peak_results:
            print(f"[WARN] No peak results found for {date} {spec_type} {source}")
            continue
        try:
            if interactive:
                key = f"{date}_{spec_type}_{source}"
                peak_bins, peak_bin_errs, energies = get_manual_matches(
                    match_file=match_file,
                    key=key,
                    source=source,
                    fit_results=source_peak_results[source],
                    known_energies=known_peaks_dict[source],
                    force_rematch=force_rematch
                )
            else:
                raise ValueError("This version expects interactive/manual matching.")

            bins_list.append(peak_bins)
            bin_errs_list.append(peak_bin_errs)
            energies_list.append(energies)
            print(f"[INFO] Using {len(peak_bins)} matched peaks from {source}")
        except Exception as e:
            print(f"[WARN] Skipping {source} for {date} {spec_type}: {e}")

    if len(bins_list) == 0:
        raise ValueError(f"No usable calibration source peaks found for {date} {spec_type}")

    bins_all = np.concatenate(bins_list)
    bin_errs_all = np.concatenate(bin_errs_list)
    energies_all = np.concatenate(energies_list)

    if len(bins_all) < 2:
        raise ValueError(f"Need at least two total matched calibration peaks for {date} {spec_type}")

    popt, pcov, chi2_val, ndof, p_value = fit_linear_odr(bins_all, energies_all, bin_errs_all)
    slope, intercept = popt
    perr = np.sqrt(np.diag(pcov))

    fig, ax = plt.subplots()
    ax.errorbar(
        bins_all, energies_all,
        xerr=bin_errs_all, yerr=np.abs(slope) * bin_errs_all,
        fmt='o', color=CBLUE, ecolor=CBLUE, capsize=3, label="Selected calibration peaks"
    )

    xdense = np.linspace(0, max(1050, 1.05 * np.max(bins_all)), 400)
    ax.plot(xdense, weighted_linear(xdense, slope, intercept), color=CRED, lw=2.5, label="Linear fit")

    textbox = (
        f"$E = m b + c$\n"
        f"$m$ = {slope:.4f} ± {perr[0]:.4f} keV/bin\n"
        f"$c$ = {intercept:.2f} ± {perr[1]:.2f} keV\n"
        f"$\\chi^2$/ndof = {chi2_val:.2f}/{ndof}\n"
        f"$p$ = {p_value:.3f}"
    )
    ax.text(
        0.98, 0.05, textbox,
        transform=ax.transAxes, ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
    )

    ax.set_xlabel("Peak location [bin]")
    ax.set_ylabel("Energy [keV]")
    ax.set_title(f"Calibration: {date} {spec_type} (Na22 + Ba133)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"calibration_{date}_{spec_type}.png"), dpi=200)
    plt.close()

    return CalibrationResult(
        date=date,
        spec_type=spec_type,
        slope=slope,
        intercept=intercept,
        slope_err=perr[0],
        intercept_err=perr[1],
        cov=pcov,
        chi2_val=chi2_val,
        ndof=ndof,
        p_value=p_value,
        used_bins=bins_all,
        used_bin_errs=bin_errs_all,
        used_energies=energies_all
    )

def bin_to_energy(bin_value: float, bin_err: float, cal: CalibrationResult):
    E = cal.slope * bin_value + cal.intercept
    varE = (
        (bin_value**2) * cal.cov[0, 0]
        + cal.cov[1, 1]
        + (cal.slope**2) * (bin_err**2)
        + 2 * bin_value * cal.cov[0, 1]
    )
    return E, math.sqrt(max(varE, 0.0))

def b_to_e(bin_value: float, bin_err: float, m: float, b: float, m_err: float, b_err: float, cross_cov: float):
    E = m * bin_value + b
    varE = (
        (bin_value**2) * m_err * m_err
        + b_err * b_err
        + (m**2) * (bin_err**2)
        + 2 * bin_value * cross_cov
    )
    return E, math.sqrt(max(varE, 0.0))

# ============================================================
# Cs137 peak selection and plots
# ============================================================

def choose_cs137_peak(
    fit_results: List[PeakFitResult],
    bins: np.ndarray,
    counts: np.ndarray,
    spec_type: str,
    min_bin: int = 0,
    rule: str = "closest_to_max"
):
    valid = [r for r in fit_results if r.success]
    if len(valid) == 0:
        raise ValueError("No successful peak fits for Cs137 spectrum")

    mask = bins >= min_bin
    bins_use = bins[mask]
    counts_use = counts[mask]
    if len(bins_use) == 0:
        raise ValueError("No bins remain after applying min_bin in choose_cs137_peak")

    max_bin = bins_use[np.argmax(counts_use)]

    if rule == "closest_to_max":
        desired_peak = min(valid, key=lambda r: abs(r.fit_center - max_bin))

    elif rule == "highest_count":
        # Choose the fitted peak whose center lies at the largest histogram count
        desired_peak = max(
            valid,
            key=lambda r: np.interp(r.fit_center, bins_use, counts_use)
        )

    else:
        raise ValueError(f"Unknown Cs137 peak rule: {rule}")

    sanity_peak = None
    if spec_type == "recoil" and len(valid) > 1:
        highest_energy_peak = max(valid, key=lambda r: r.fit_center)
        if highest_energy_peak is not desired_peak:
            sanity_peak = highest_energy_peak

    return desired_peak, sanity_peak

def make_selected_cs137_peak_plot(
    spectrum: Spectrum,
    calibration: CalibrationResult,
    peak_results: List[PeakFitResult],
    selected_peak: PeakFitResult,
    output_dir: str = "better-plots",
    sanity_peak: Optional[PeakFitResult] = None,
    min_bin: int = 0
):
    mask = spectrum.bins >= min_bin
    bins_use = spectrum.bins[mask]
    counts_use = spectrum.counts[mask]
    energies = calibration.slope * bins_use + calibration.intercept
    bar_width = np.diff(energies).mean() if len(energies) > 1 else 1.0

    fig, ax = plt.subplots()
    ax.bar(energies, counts_use, width=bar_width, color=CBLUE, edgecolor=None, linewidth=0, label="Histogram")

    valid = sorted([r for r in peak_results if r.success], key=lambda r: r.fit_center)
    for r in valid:
        xpk = calibration.slope * r.fit_center + calibration.intercept
        ypk = np.interp(r.fit_center, bins_use, counts_use)
        ax.plot(xpk, ypk, 'o', color=CGRAY, markersize=7)

    Esel, Esel_err = bin_to_energy(selected_peak.fit_center, selected_peak.fit_center_err, calibration)
    xsel = Esel
    ysel = np.interp(selected_peak.fit_center, bins_use, counts_use)
    ax.axvline(xsel, color=CRED, lw=2.5, label="Selected peak")
    ax.plot(xsel, ysel, 'o', color=CRED, markersize=9)

    textbox = (
        f"Selected peak = {Esel:.2f} ± {Esel_err:.2f} keV\n"
        f"$\\chi^2$/ndof = {selected_peak.chi2_val:.2f}/{selected_peak.ndof}\n"
        f"$p$ = {selected_peak.p_value:.3f}"
    )

    if sanity_peak is not None:
        Esan, Esan_err = bin_to_energy(sanity_peak.fit_center, sanity_peak.fit_center_err, calibration)
        xsan = Esan
        ysan = np.interp(sanity_peak.fit_center, bins_use, counts_use)
        ax.axvline(xsan, color=CPURPLE, lw=2, ls='--', label="Sanity peak")
        ax.plot(xsan, ysan, 'o', color=CPURPLE, markersize=8)
        textbox += f"\nSanity peak = {Esan:.2f} ± {Esan_err:.2f} keV"

    if min_bin > 0:
        ax.text(
            0.02, 0.95, f"Low-bin cutoff applied: bin ≥ {min_bin}",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
        )

    ax.text(
        0.98, 0.95, textbox,
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
    )

    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel("Counts")
    ax.set_title(f"{spectrum.date} {spectrum.spec_type} {spectrum.angle:.1f}° selected Cs137 peak")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f"{sanitize_filename(f'{spectrum.date}_Cs137_{spectrum.spec_type}_{spectrum.angle}_selected_peak')}.png"
        ),
        dpi=200
    )
    plt.close()

# ============================================================
# Daily calibration overview
# ============================================================

def make_daily_calibration_overview(date, spectra_map, peak_map, output_dir="better-plots", low_bin_cutoffs=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    layout = [
        ("scatter", "Na22"),
        ("scatter", "Ba133"),
        ("recoil", "Na22"),
        ("recoil", "Ba133"),
    ]

    for ax, (spec_type, source) in zip(axes.flat, layout):
        key = (spec_type, source)
        if key not in spectra_map:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{date} {source} {spec_type}")
            continue

        sp = spectra_map[key]
        min_bin = get_low_bin_cutoff(sp.date, sp.source, sp.spec_type, sp.angle, low_bin_cutoffs)
        mask = sp.bins >= min_bin
        bins_use = sp.bins[mask]
        counts_use = sp.counts[mask]

        ax.bar(bins_use, counts_use, width=1.0, color=CBLUE, edgecolor=None, linewidth=0)
        valid = sorted([r for r in peak_map.get(key, []) if r.success], key=lambda r: r.fit_center)

        for i, r in enumerate(valid):
            yval = np.interp(r.fit_center, bins_use, counts_use)
            ax.axvline(r.fit_center, color=CRED, lw=2)
            ax.plot(r.fit_center, yval, 'o', color=CRED, markersize=7)
            ax.text(
                r.fit_center,
                yval + 0.03 * np.max(counts_use) if np.max(counts_use) > 0 else yval + 1,
                f"{i}",
                color=CBLACK, ha="center", va="bottom", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none")
            )

        if min_bin > 0:
            ax.text(
                0.02, 0.95, f"bin ≥ {min_bin}",
                transform=ax.transAxes, ha="left", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
            )

        ax.set_title(f"{source} {spec_type}")
        ax.set_xlabel("Bin")
        ax.set_ylabel("Counts")

    fig.suptitle(f"{date} calibration spectra overview", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(output_dir, f"calibration_overview_{date}.png"), dpi=200)
    plt.close()

# ============================================================
# Energy sum and Compton plots
# ============================================================

def make_energy_sum_plot(
    sums,
    output_dir="better-plots",
    filename="energy_sum_vs_angle.png",
    systematic_summary=None
):
    if len(sums) == 0:
        return None

    angles = np.array([x[1] for x in sums], dtype=float)
    Etots = np.array([x[4] for x in sums], dtype=float)
    Eerrs_stat = np.array([x[5] for x in sums], dtype=float)

    if systematic_summary is not None:
        Eerrs_sys = np.array([
            get_angle_systematic(systematic_summary.get("sum_by_angle", None), ang)
            for ang in angles
        ], dtype=float)
        _, Eerrs_plot = combine_point_stat_and_sys(Etots, Eerrs_stat, Eerrs_sys)
    else:
        Eerrs_sys = np.full_like(Eerrs_stat, np.nan, dtype=float)
        Eerrs_plot = Eerrs_stat

    fig, ax = plt.subplots()
    ax.errorbar(
        angles, Etots, yerr=Eerrs_plot,
        fmt='o', color=CBLUE, ecolor=CBLUE, capsize=4, markersize=8,
        label="Measured sums"
    )

    expected_energy = 661.6
    mean_all, mean_all_err = mean_with_propagated_uncertainty(Etots, Eerrs_stat)

    mask_no_310 = ~np.isclose(angles, 310.0)
    if np.any(mask_no_310):
        mean_no_310, mean_no_310_err = mean_with_propagated_uncertainty(
            Etots[mask_no_310], Eerrs_stat[mask_no_310]
        )
    else:
        mean_no_310, mean_no_310_err = np.nan, np.nan

    ax.axhline(expected_energy, color=CRED, lw=2.5, ls='--',
               label=f"Expected: {expected_energy:.1f} keV")
    ax.axhline(mean_all, color=CGREEN, lw=2.5, ls='-.',
               label=f"Mean (all): {mean_all:.1f} keV")
    if np.isfinite(mean_no_310):
        ax.axhline(mean_no_310, color=CPURPLE, lw=2.5, ls=':',
                   label=f"Mean (excluding 310°): {mean_no_310:.1f} keV")

    ax.set_xlabel("Scattering angle [deg]")
    ax.set_ylabel(r"$E_{\mathrm{scatter}} + E_{\mathrm{recoil}}$ [keV]")
    ax.set_title("Sum of scatter and recoil energies vs angle")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()

    sys_all = np.nan
    sys_no_310 = np.nan
    tot_all = mean_all_err
    tot_no_310 = mean_no_310_err

    if systematic_summary is not None:
        sys_all = systematic_summary.get("mean_including_310_systematic_keV", np.nan)
        sys_no_310 = systematic_summary.get("mean_excluding_310_systematic_keV", np.nan)
        tot_all = combine_stat_and_sys(mean_all_err, sys_all)
        tot_no_310 = combine_stat_and_sys(mean_no_310_err, sys_no_310)

    return {
        "including_310": {
            "mean_keV": mean_all,
            "stat_err_keV": mean_all_err,
            "sys_err_keV": sys_all,
            "total_err_keV": tot_all,
            "sem_keV": sample_sem(Etots),
        },
        "excluding_310": {
            "mean_keV": mean_no_310,
            "stat_err_keV": mean_no_310_err,
            "sys_err_keV": sys_no_310,
            "total_err_keV": tot_no_310,
            "sem_keV": sample_sem(Etots[mask_no_310]) if np.any(mask_no_310) else np.nan,
        }
    }

def make_inverse_scatter_energy_plot(
    cs_energies,
    output_dir="better-plots",
    incident_energy_keV=CS137_ENERGY_KEV,
    electron_rest_energy_keV=ELECTRON_REST_ENERGY_KEV,
    systematic_summary=None,
    filename="inverse_scatter_energy_vs_one_minus_cos.png"
):
    scatter = sorted([c for c in cs_energies if c.spec_type == "scatter"], key=lambda x: x.angle)
    if len(scatter) == 0:
        print("[WARN] No scatter energies available for inverse scatter-energy plot.")
        return

    theta = np.array([c.angle for c in scatter], dtype=float)
    E = np.array([c.energy_keV for c in scatter], dtype=float)
    Eerr_stat = np.array([c.energy_err_keV for c in scatter], dtype=float)

    if systematic_summary is not None:
        Eerr_sys = np.array([
            get_angle_systematic(systematic_summary.get("scatter_by_angle", None), ang)
            for ang in theta
        ], dtype=float)
        _, Eerr = combine_point_stat_and_sys(E, Eerr_stat, Eerr_sys)
    else:
        Eerr = Eerr_stat

    x = one_minus_cos_theta(theta)
    y, yerr = inverse_with_error(E, Eerr)

    x_theory = np.linspace(0.0, max(1.05 * np.max(x), 2.05), 500)
    y_theory = (1.0 / incident_energy_keV) + (1.0 / electron_rest_energy_keV) * x_theory

    fig, ax = plt.subplots()
    ax.errorbar(
        x, y, yerr=yerr,
        fmt='o', color=CBLUE, ecolor=CBLUE, capsize=4, markersize=8,
        label="Measured data"
    )
    ax.plot(x_theory, y_theory, color=CRED, lw=2.5, label="Compton prediction")

    textbox = (
        r"$\frac{1}{E_\gamma'} = \frac{1}{E_0} + \frac{1}{m_ec^2}(1-\cos\theta)$" "\n"
        f"$E_0$ = {incident_energy_keV:.3f} keV\n"
        f"$m_ec^2$ = {electron_rest_energy_keV:.3f} keV"
    )
    ax.text(
        0.98, 0.05, textbox,
        transform=ax.transAxes, ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
    )

    ax.set_xlabel(r"$1-\cos\theta$ [dimensionless]")
    ax.set_ylabel(r"$1/E_{\mathrm{scatter}}$ [keV$^{-1}$]")
    ax.set_title(r"Inverse scattered-photon energy vs $1-\cos\theta$")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()

def make_inverse_recoil_energy_plot(
    cs_energies,
    output_dir="better-plots",
    incident_energy_keV=CS137_ENERGY_KEV,
    electron_rest_energy_keV=ELECTRON_REST_ENERGY_KEV,
    systematic_summary=None,
    filename="inverse_recoil_energy_vs_inv_one_minus_cos.png"
):
    recoil = sorted([c for c in cs_energies if c.spec_type == "recoil"], key=lambda x: x.angle)
    if len(recoil) == 0:
        print("[WARN] No recoil energies available for inverse recoil-energy plot.")
        return

    theta = np.array([c.angle for c in recoil], dtype=float)
    T = np.array([c.energy_keV for c in recoil], dtype=float)
    Terr_stat = np.array([c.energy_err_keV for c in recoil], dtype=float)

    if systematic_summary is not None:
        Terr_sys = np.array([
            get_angle_systematic(systematic_summary.get("recoil_by_angle", None), ang)
            for ang in theta
        ], dtype=float)
        _, Terr = combine_point_stat_and_sys(T, Terr_stat, Terr_sys)
    else:
        Terr = Terr_stat

    x_raw = one_minus_cos_theta(theta)
    valid = x_raw > 0

    theta = theta[valid]
    T = T[valid]
    Terr = Terr[valid]
    x = 1.0 / x_raw[valid]
    y, yerr = inverse_with_error(T, Terr)

    if len(x) == 0:
        print("[WARN] No valid recoil points for inverse recoil-energy plot.")
        return

    x_theory = np.linspace(0.0, 1.05 * np.max(x), 500)
    y_theory = (1.0 / incident_energy_keV) + (electron_rest_energy_keV / incident_energy_keV**2) * x_theory

    fig, ax = plt.subplots()
    ax.errorbar(
        x, y, yerr=yerr,
        fmt='o', color=CBLUE, ecolor=CBLUE, capsize=4, markersize=8,
        label="Measured data"
    )
    ax.plot(x_theory, y_theory, color=CRED, lw=2.5, label="Compton prediction")

    textbox = (
        r"$\frac{1}{T_e} = \frac{1}{E_0} + \frac{m_ec^2}{E_0^2}\frac{1}{1-\cos\theta}$" "\n"
        f"$E_0$ = {incident_energy_keV:.3f} keV\n"
        f"$m_ec^2$ = {electron_rest_energy_keV:.3f} keV"
    )
    ax.text(
        0.98, 0.05, textbox,
        transform=ax.transAxes, ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
    )

    ax.set_xlabel(r"$1/(1-\cos\theta)$ [dimensionless]")
    ax.set_ylabel(r"$1/E_{\mathrm{recoil}}$ [keV$^{-1}$]")
    ax.set_title(r"Inverse recoil-electron energy vs $1/(1-\cos\theta)$")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()

# ============================================================
# Main analysis
# ============================================================

def analyze_compton_data(
    data_dir: str,
    output_dir: str = "better-plots",
    known_peaks_dict: Dict[str, np.ndarray] = KNOWN_PEAKS_KEV,
    match_file: str = "manual_matches.json",
    force_rematch: bool = False,
    low_bin_cutoffs: Optional[Dict[Tuple[str, str, str], int]] = None,
    fit_half_width: int = 12,
    p_value_cut: float = 0.05,
    cs_peak_rule: str = "closest_to_max",
    save_plots: bool = True,
    interactive_calibration: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    if low_bin_cutoffs is None:
        low_bin_cutoffs = LOW_BIN_CUTOFFS

    spectra = load_all_spectra(data_dir)

    by_date_type_source = {}
    for sp in spectra:
        by_date_type_source.setdefault((sp.date, sp.spec_type, sp.source), []).append(sp)

    calibrations = {}

    all_dates = sorted(set(sp.date for sp in spectra))
    for date in all_dates:
        day_spectra_map = {}
        day_peak_map = {}

        for spec_type in ["scatter", "recoil"]:
            source_peak_results = {}

            for source in ["Na22", "Ba133"]:
                key = (date, spec_type, source)
                if key not in by_date_type_source:
                    continue

                sp = by_date_type_source[key][0]
                day_spectra_map[(spec_type, source)] = sp

                min_bin = get_low_bin_cutoff(sp.date, sp.source, sp.spec_type, sp.angle, low_bin_cutoffs)

                peak_results = find_and_fit_peaks(
                    sp.bins,
                    sp.counts,
                    title_prefix=f"{date}_{source}_{spec_type}",
                    output_dir=output_dir,
                    min_bin=min_bin,
                    calibration=None,
                    min_p_value=p_value_cut,
                    fit_half_width=fit_half_width,
                    save_plots=save_plots
                )
                source_peak_results[source] = peak_results
                day_peak_map[(spec_type, source)] = peak_results

            if len(source_peak_results) >= 1:
                try:
                    cal = calibrate_day_type(
                        date=date,
                        spec_type=spec_type,
                        source_peak_results=source_peak_results,
                        output_dir=output_dir,
                        known_peaks_dict=known_peaks_dict,
                        interactive=interactive_calibration,
                        match_file=match_file,
                        force_rematch=force_rematch
                    )
                    calibrations[(date, spec_type)] = cal
                    print(f"[OK] Calibration built for {date} {spec_type}")
                    print(
                        f"     E = ({cal.slope:.5f} ± {cal.slope_err:.5f}) * bin + "
                        f"({cal.intercept:.3f} ± {cal.intercept_err:.3f}) keV"
                    )
                except Exception as e:
                    print(f"[WARN] Could not calibrate {date} {spec_type}: {e}")

        if save_plots:
            make_daily_calibration_overview(
                date=date,
                spectra_map=day_spectra_map,
                peak_map=day_peak_map,
                output_dir=output_dir,
                low_bin_cutoffs=low_bin_cutoffs
            )

    cs_energies = []
    cs_sanity = []

    for sp in spectra:
        if sp.source != "Cs137":
            continue

        cal_key = (sp.date, sp.spec_type)
        if cal_key not in calibrations:
            print(f"[WARN] Missing calibration for {sp.filename}")
            continue

        cal = calibrations[cal_key]
        min_bin = get_low_bin_cutoff(sp.date, sp.source, sp.spec_type, sp.angle, low_bin_cutoffs)

        peak_results = find_and_fit_peaks(
            sp.bins,
            sp.counts,
            title_prefix=f"{sp.date}_{sp.source}_{sp.spec_type}_{sp.angle}",
            output_dir=output_dir,
            min_bin=min_bin,
            calibration=cal,
            min_p_value=p_value_cut,
            fit_half_width=fit_half_width,
            save_plots=save_plots
        )

        try:
            desired_peak, sanity_peak = choose_cs137_peak(
                fit_results=peak_results,
                bins=sp.bins,
                counts=sp.counts,
                spec_type=sp.spec_type,
                min_bin=min_bin,
                rule=cs_peak_rule
            )

            E, Eerr = bin_to_energy(desired_peak.fit_center, desired_peak.fit_center_err, cal)

            cs_energies.append(CsPeakEnergy(
                date=sp.date,
                spec_type=sp.spec_type,
                angle=sp.angle,
                peak_bin=desired_peak.fit_center,
                peak_bin_err=desired_peak.fit_center_err,
                energy_keV=E,
                energy_err_keV=Eerr,
                chi2_val=desired_peak.chi2_val,
                p_value=desired_peak.p_value,
                note="desired peak"
            ))

            if sanity_peak is not None and sanity_peak is not desired_peak:
                E2, E2err = bin_to_energy(sanity_peak.fit_center, sanity_peak.fit_center_err, cal)
                cs_sanity.append(CsPeakEnergy(
                    date=sp.date,
                    spec_type=sp.spec_type,
                    angle=sp.angle,
                    peak_bin=sanity_peak.fit_center,
                    peak_bin_err=sanity_peak.fit_center_err,
                    energy_keV=E2,
                    energy_err_keV=E2err,
                    chi2_val=sanity_peak.chi2_val,
                    p_value=sanity_peak.p_value,
                    note="high-energy sanity peak"
                ))

            if save_plots:
                make_selected_cs137_peak_plot(
                    spectrum=sp,
                    calibration=cal,
                    peak_results=peak_results,
                    selected_peak=desired_peak,
                    sanity_peak=sanity_peak,
                    output_dir=output_dir,
                    min_bin=min_bin
                )

        except Exception as e:
            print(f"[WARN] Could not determine Cs137 peak for {sp.filename}: {e}")

    scatter_dict = {(c.date, c.angle): c for c in cs_energies if c.spec_type == "scatter"}
    recoil_dict = {(c.date, c.angle): c for c in cs_energies if c.spec_type == "recoil"}

    sums = []
    for key in sorted(set(scatter_dict.keys()) & set(recoil_dict.keys()), key=lambda x: (x[0], x[1])):
        s = scatter_dict[key]
        r = recoil_dict[key]
        Etot = s.energy_keV + r.energy_keV
        Etot_err = math.sqrt(s.energy_err_keV**2 + r.energy_err_keV**2)
        sums.append((key[0], key[1], s, r, Etot, Etot_err))

    mean_results = None
    if len(sums) > 0 and save_plots:
        mean_results = make_energy_sum_plot(sums, output_dir=output_dir, filename="energy_sum_vs_angle.png")
    elif len(sums) > 0:
        # still compute means
        angles = np.array([x[1] for x in sums], dtype=float)
        Etots = np.array([x[4] for x in sums], dtype=float)
        Eerrs = np.array([x[5] for x in sums], dtype=float)
        mean_all, mean_all_err = mean_with_propagated_uncertainty(Etots, Eerrs)
        mask_no_310 = ~np.isclose(angles, 310.0)
        mean_no_310, mean_no_310_err = mean_with_propagated_uncertainty(Etots[mask_no_310], Eerrs[mask_no_310]) if np.any(mask_no_310) else (np.nan, np.nan)
        mean_results = {
            "including_310": {"mean_keV": mean_all, "stat_err_keV": mean_all_err, "sem_keV": sample_sem(Etots)},
            "excluding_310": {"mean_keV": mean_no_310, "stat_err_keV": mean_no_310_err, "sem_keV": sample_sem(Etots[mask_no_310]) if np.any(mask_no_310) else np.nan}
        }

    print("\n=== Calibrations ===")
    for key, cal in calibrations.items():
        print(
            f"{key}: slope = {cal.slope:.5f} ± {cal.slope_err:.5f} keV/bin, "
            f"intercept = {cal.intercept:.3f} ± {cal.intercept_err:.3f} keV, "
            f"chi2/ndof = {cal.chi2_val:.2f}/{cal.ndof}, p = {cal.p_value:.3f}"
        )

    print("\n=== Cs137 desired peak energies ===")
    for c in sorted(cs_energies, key=lambda x: (x.date, x.spec_type, x.angle)):
        print(
            f"{c.date} {c.spec_type:7s} angle={c.angle:6.1f} deg: "
            f"E = {c.energy_keV:.2f} ± {c.energy_err_keV:.2f} keV, "
            f"bin = {c.peak_bin:.2f} ± {c.peak_bin_err:.2f}, "
            f"p = {c.p_value:.3f}"
        )

    if len(cs_sanity) > 0:
        print("\n=== Cs137 recoil high-energy sanity peaks ===")
        for c in sorted(cs_sanity, key=lambda x: (x.date, x.angle)):
            print(
                f"{c.date} recoil angle={c.angle:6.1f} deg: "
                f"E = {c.energy_keV:.2f} ± {c.energy_err_keV:.2f} keV"
            )

    if len(cs_sanity) > 0:
        sanity_energies = np.array([c.energy_keV for c in cs_sanity], dtype=float)
        sanity_errors = np.array([c.energy_err_keV for c in cs_sanity], dtype=float)

        sanity_mean, sanity_mean_err = mean_with_propagated_uncertainty(
            sanity_energies, sanity_errors
        )
        sanity_sem = sample_sem(sanity_energies)

        print("\n=== Cs137 recoil sanity-peak mean ===")
        print(
            f"Mean sanity-peak energy = {sanity_mean:.2f} ± {sanity_mean_err:.2f} keV "
            f"(propagated)"
        )
        print(
            f"SEM from spread = {sanity_sem:.2f} keV"
        )

    print("\n=== Energy sums ===")
    for date, angle, s, r, Etot, Etot_err in sums:
        print(
            f"{date} angle={angle:6.1f} deg: "
            f"Escatter = {s.energy_keV:.2f} ± {s.energy_err_keV:.2f} keV, "
            f"Erecoil = {r.energy_keV:.2f} ± {r.energy_err_keV:.2f} keV, "
            f"sum = {Etot:.2f} ± {Etot_err:.2f} keV"
        )

    if mean_results is not None:
        print("\n=== Mean energy sums ===")
        print(
            f"Including 310°: mean = {mean_results['including_310']['mean_keV']:.2f} ± "
            f"{mean_results['including_310']['stat_err_keV']:.2f} keV (stat), "
            f"SEM = {mean_results['including_310']['sem_keV']:.2f} keV"
        )
        if np.isfinite(mean_results["excluding_310"]["mean_keV"]):
            print(
                f"Excluding 310°: mean = {mean_results['excluding_310']['mean_keV']:.2f} ± "
                f"{mean_results['excluding_310']['stat_err_keV']:.2f} keV (stat), "
                f"SEM = {mean_results['excluding_310']['sem_keV']:.2f} keV"
            )

    # Final plots should be last
    if save_plots:
        make_inverse_scatter_energy_plot(cs_energies=cs_energies, output_dir=output_dir)
        make_inverse_recoil_energy_plot(cs_energies=cs_energies, output_dir=output_dir)

    return {
        "spectra": spectra,
        "calibrations": calibrations,
        "cs_energies": cs_energies,
        "cs_sanity": cs_sanity,
        "energy_sums": sums,
        "mean_results": mean_results
    }

# ============================================================
# Systematic uncertainty machinery
# ============================================================

def build_systematic_variations(base_low_bin_cutoffs=None):
    if base_low_bin_cutoffs is None:
        base_low_bin_cutoffs = LOW_BIN_CUTOFFS

    variations = []
    for fit_half_width in SYSTEMATIC_CONFIG["fit_half_width_values"]:
        for p_cut in SYSTEMATIC_CONFIG["p_value_cut_values"]:
            for low_shift in SYSTEMATIC_CONFIG["low_bin_cut_shifts"]:
                for cs_rule in SYSTEMATIC_CONFIG["cs_peak_rules"]:
                    shifted = {}
                    for k, v in copy.deepcopy(base_low_bin_cutoffs).items():
                        shifted[k] = max(0, v + low_shift)
                    variations.append({
                        "fit_half_width": fit_half_width,
                        "p_value_cut": p_cut,
                        "low_bin_cutoffs": shifted,
                        "cs_peak_rule": cs_rule,
                        "label": f"fhw={fit_half_width}, p={p_cut}, cutshift={low_shift}, csr={cs_rule}"
                    })
    return variations

def run_systematic_study(
    data_dir: str,
    output_dir: str = "better-plots",
    known_peaks_dict: Dict[str, np.ndarray] = KNOWN_PEAKS_KEV,
    match_file: str = "manual_matches.json",
    base_low_bin_cutoffs=None
):
    variations = build_systematic_variations(base_low_bin_cutoffs)
    systematic_runs = []

    print("\n" + "=" * 80)
    print("Running systematic study")
    print("=" * 80)

    for i, var in enumerate(variations, start=1):
        print(f"[SYS {i}/{len(variations)}] {var['label']}")
        try:
            result = analyze_compton_data(
                data_dir=data_dir,
                output_dir=output_dir,
                known_peaks_dict=known_peaks_dict,
                match_file=match_file,
                force_rematch=False,
                low_bin_cutoffs=var["low_bin_cutoffs"],
                fit_half_width=var["fit_half_width"],
                p_value_cut=var["p_value_cut"],
                cs_peak_rule=var["cs_peak_rule"],
                save_plots=False,
                interactive_calibration=True
            )
            systematic_runs.append({"variation": var, "result": result})
        except Exception as e:
            print(f"[WARN] Systematic variation failed: {var['label']} :: {e}")

    return systematic_runs

def get_angle_systematic(sys_dict, angle):
    """
    Fetch systematic uncertainty for a given angle from a dictionary like:
        sys_summary["scatter_by_angle"]
        sys_summary["recoil_by_angle"]
        sys_summary["sum_by_angle"]
    """
    if sys_dict is None:
        return np.nan
    if angle not in sys_dict:
        return np.nan
    return sys_dict[angle]["systematic_err_keV"]

def combine_point_stat_and_sys(values, stat_errs, sys_errs):
    """
    Combine point-by-point statistical and systematic uncertainties in quadrature.
    """
    values = np.asarray(values, dtype=float)
    stat_errs = np.asarray(stat_errs, dtype=float)
    sys_errs = np.asarray(sys_errs, dtype=float)

    total_errs = np.full_like(stat_errs, np.nan, dtype=float)
    for i in range(len(stat_errs)):
        total_errs[i] = combine_stat_and_sys(stat_errs[i], sys_errs[i])

    return values, total_errs

def summarize_systematics(systematic_runs):
    """
    Estimate systematic uncertainty from the spread across analysis variations for:
      - scatter energy by angle
      - recoil energy by angle
      - sum energy by angle
      - mean including 310
      - mean excluding 310
    """
    if len(systematic_runs) == 0:
        return None

    scatter_by_angle = {}
    recoil_by_angle = {}
    sum_by_angle = {}

    for run in systematic_runs:
        result = run["result"]

        # Scatter / recoil energies
        for c in result["cs_energies"]:
            if c.spec_type == "scatter":
                scatter_by_angle.setdefault(c.angle, []).append(c.energy_keV)
            elif c.spec_type == "recoil":
                recoil_by_angle.setdefault(c.angle, []).append(c.energy_keV)

        # Sums
        for _, angle, _, _, Etot, _ in result["energy_sums"]:
            sum_by_angle.setdefault(angle, []).append(Etot)

    scatter_systematics = {
        angle: {
            "values_keV": vals,
            "systematic_err_keV": systematic_uncertainty_from_variations(vals)
        }
        for angle, vals in scatter_by_angle.items()
    }

    recoil_systematics = {
        angle: {
            "values_keV": vals,
            "systematic_err_keV": systematic_uncertainty_from_variations(vals)
        }
        for angle, vals in recoil_by_angle.items()
    }

    sum_systematics = {
        angle: {
            "values_keV": vals,
            "systematic_err_keV": systematic_uncertainty_from_variations(vals)
        }
        for angle, vals in sum_by_angle.items()
    }

    mean_all_vals = []
    mean_no_310_vals = []

    for run in systematic_runs:
        mr = run["result"]["mean_results"]
        if mr is None:
            continue
        if np.isfinite(mr["including_310"]["mean_keV"]):
            mean_all_vals.append(mr["including_310"]["mean_keV"])
        if np.isfinite(mr["excluding_310"]["mean_keV"]):
            mean_no_310_vals.append(mr["excluding_310"]["mean_keV"])

    return {
        "scatter_by_angle": scatter_systematics,
        "recoil_by_angle": recoil_systematics,
        "sum_by_angle": sum_systematics,
        "mean_including_310_systematic_keV": systematic_uncertainty_from_variations(mean_all_vals),
        "mean_excluding_310_systematic_keV": systematic_uncertainty_from_variations(mean_no_310_vals),
        "mean_including_310_values_keV": mean_all_vals,
        "mean_excluding_310_values_keV": mean_no_310_vals,
    }

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    DATA_DIR = "data"
    OUTPUT_DIR = "better-plots"
    DIR_TYPE = {
        "recoil": "\\recoil",
        "scatter": "\\scatter",   
    }
    DIR_SOURCE = {
        "Na22": "\\Na22",
        "Ba133": "\\Ba133",
        "Cs137": "\\Cs137",
    }
    DIR_ANALYSIS = {
        "raw": "\\raw",
        "local": "\\local-peaks",
    }
    MATCH_FILE = "better-manual_matches.json"

    ###################
    # FILE MANAGEMENT #
    ###################

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR+"\\calibrations", exist_ok=True)
    os.makedirs(OUTPUT_DIR+"\\final", exist_ok=True)
    for sp_type in DIR_TYPE:
        for source in DIR_SOURCE:
            for a in DIR_ANALYSIS:
                os.makedirs(OUTPUT_DIR + DIR_SOURCE[source] + DIR_TYPE[sp_type] + DIR_ANALYSIS[a], exist_ok=True)


    spectra = load_all_spectra(DATA_DIR, factor=FACTOR) # load all histograms

    by_date_type_source = {}
    for sp in spectra:
        by_date_type_source.setdefault((sp.date, sp.spec_type, sp.source), []).append(sp)

    #####################
    # RAW REBINNED DATA #
    #####################
    
    spectra_maps = {}

    all_dates = sorted(set(sp.date for sp in spectra))
    for date in all_dates:
        day_spectra_map = {}

        for spec_type in ["scatter", "recoil"]:
            source_peak_results = {}

            for source in ["Na22", "Ba133", "Cs137"]:
                key = (date, spec_type, source)
                if key not in by_date_type_source:
                    continue
                
  
                sp = by_date_type_source[key]
                day_spectra_map[(spec_type, source)] = sp

        spectra_maps[date] = day_spectra_map

        for sp_type in ("recoil", "scatter"):
            for source in ("Na22", "Ba133", "Cs137"):
                output_dir = OUTPUT_DIR + DIR_SOURCE[source] + DIR_TYPE[sp_type] + DIR_ANALYSIS["raw"]
                key = (sp_type, source)
                sp_list = day_spectra_map[key]
                for sp in sp_list:
                    min_bin = get_low_bin_cutoff(sp.date, sp.source, sp.spec_type, sp.angle, LOW_BIN_CUTOFFS)
                    mask = sp.bins >= min_bin
                    bins_use = sp.bins[mask]
                    counts_use = sp.counts[mask]

                    fig, ax = plt.subplots()
                    ax.bar(bins_use, counts_use, width=1.0, color=CBLUE, edgecolor=None, linewidth=0)
                    if source == "Cs137":
                        ax.set_title(f"{date}: {source} {sp.angle} ({sp_type})")
                    else:
                        ax.set_title(f"{date}: {source} ({sp_type})")
                    ax.set_xlabel("Bin")
                    ax.set_ylabel("Counts")
                    plt.tight_layout(rect=[0, 0, 1, 0.97])
                    if source == "Cs137":
                        plt.savefig(os.path.join(output_dir, f"{date}-{sp.angle}-rebinned{2048//FACTOR}.png"), dpi=200)
                    else:
                        plt.savefig(os.path.join(output_dir, f"{date}-rebinned{2048//FACTOR}.png"), dpi=200)
                    plt.close()
    
    ####################
    # SINGLE NA22 PEAK #
    ####################
    Na22_peak_locations = {}
    Na22_peak_errors = {}
    for date in all_dates:
        for sp_type in ("recoil", "scatter"):
            output_dir = OUTPUT_DIR + DIR_SOURCE["Na22"] + DIR_TYPE[sp_type] + DIR_ANALYSIS["local"]
            sp_Na22 = spectra_maps[date][(sp_type, "Na22")][0]

            min_bin = get_low_bin_cutoff(sp_Na22.date, sp_Na22.source, sp_Na22.spec_type, sp_Na22.angle, LOW_BIN_CUTOFFS)
            mask = sp_Na22.bins >= min_bin
            
            counts = sp_Na22.counts
            bins = sp_Na22.bins

            max_bin_index = int(np.argmax(counts[mask])) + min_bin

            bins_use = bins[max_bin_index-HALF_WIDTH_FIT:max_bin_index+HALF_WIDTH_FIT+1]
            counts_use = counts[max_bin_index-HALF_WIDTH_FIT:max_bin_index+HALF_WIDTH_FIT+1]

            # print(max_bin_index)

            left = max(0, max_bin_index - HALF_WIDTH_FIT)
            right = min(len(bins_use) - 1, max_bin_index + HALF_WIDTH_FIT)

            xfit = bins_use
            yfit = counts_use
            yerr = poisson_errors(yfit)

            A0 = max(yfit) - np.median(yfit)
            mu0 = max_bin_index
            sigma0 = max(1.5, HALF_WIDTH_FIT / 4)
            b00 = np.median(yfit)
            b10 = 0.0

            lower = [0, xfit.min(), 0.5, -np.inf, -np.inf]
            upper = [np.inf, xfit.max(), HALF_WIDTH_FIT, np.inf, np.inf]

            popt, pcov = curve_fit(
                gaussian_plus_linear,
                xfit, yfit,
                p0=[A0, mu0, sigma0, b00, b10],
                sigma=yerr,
                absolute_sigma=True,
                bounds=(lower, upper),
                maxfev=20000
            )
            model = gaussian_plus_linear(xfit, *popt)
            chi2_val, ndof, p_value = compute_chi2(yfit, model, yerr, 5)
            perr = np.sqrt(np.diag(pcov))

            xdense = np.linspace(xfit.min(), xfit.max(), 400)
            ydense = gaussian_plus_linear(xdense, *popt)
            peak_plot = popt[1]
            peak_plot_err = perr[1]

            Na22_peak_locations[f"{date}-{sp_type}", 511] = peak_plot
            Na22_peak_errors[f"{date}-{sp_type}", 511] = peak_plot_err

            fig, ax = plt.subplots()
            ax.bar(bins_use, counts_use, width=1.0, color=CBLUE, edgecolor=None, linewidth=0)
            ax.set_title(f"{date}: Na22 ({sp_type})")
            ax.set_xlabel("Bin")
            ax.set_ylabel("Counts")
            ax.plot(xdense, ydense, color=CRED, lw=2.5, label="Gaussian + linear fit")
            ax.axvline(peak_plot, color=CGREEN, ls="--", lw=2,
                        label=f"Peak = {peak_plot:.2f} ± {peak_plot_err:.2f}")

            textbox = (
                f"$\\mu$ = {peak_plot:.2f} ± {peak_plot_err:.2f}\n"
                f"$\\sigma$ = {popt[2]:.2f} ± {perr[2]:.2f} bins\n"
                f"$\\chi^2$/ndof = {chi2_val:.2f}/{ndof}\n"
                f"$p$ = {p_value:.3f}\n"
            )
            ax.text(
                0.98, 0.95, textbox, transform=ax.transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
            )
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(os.path.join(output_dir, f"{date}-local_peak-{2048//FACTOR}.png"), dpi=200)
            plt.close()

    ###############
    # Ba133 Peaks #
    ###############
    Ba133_peak_locations = {}
    Ba133_peak_errors = {}
    for date in all_dates:
        for sp_type in ("recoil", "scatter"):
            output_dir = OUTPUT_DIR + DIR_SOURCE["Ba133"] + DIR_TYPE[sp_type] + DIR_ANALYSIS["local"]
            sp_Ba133 = spectra_maps[date][(sp_type, "Ba133")][0]

            min_bin = get_low_bin_cutoff(sp_Ba133.date, sp_Ba133.source, sp_Ba133.spec_type, sp_Ba133.angle, LOW_BIN_CUTOFFS)
            mask = sp_Ba133.bins >= min_bin

            counts = sp_Ba133.counts
            counts_use = counts[mask]
            bins = sp_Ba133.bins
            bins_use = bins[mask]
            
            # Candidates from find_peaks
            peaks, _ = find_peaks(counts_use, distance=2*HALF_WIDTH_FIT, prominence=PROMINENCE)

            fig, ax = plt.subplots()
            ax.bar(bins_use, counts_use, width=1.0, color=CBLUE, edgecolor=None, linewidth=0)
            ax.vlines(bins_use[peaks], 0, counts_use[peaks], colors=CORANGE)
            ax.set_title(f"{date}: Ba133 ({sp_type})")
            ax.set_xlabel("Bin")
            ax.set_ylabel("Counts")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(os.path.join(output_dir, f"{date}-candidates-{2048//FACTOR}.png"), dpi=200)
            plt.close()

            # Refinement
            indices = [(0, 81), (1, 160), (-1, 356)] if sp_type == "scatter" else [(0, 81), (1, 160), (-2, 356)]

            for index in indices:
                max_bin_index = peaks[index[0]] + min_bin

                bins_use = bins[max_bin_index-HALF_WIDTH_FIT:max_bin_index+HALF_WIDTH_FIT+1]
                counts_use = counts[max_bin_index-HALF_WIDTH_FIT:max_bin_index+HALF_WIDTH_FIT+1]

                left = max(0, max_bin_index - HALF_WIDTH_FIT)
                right = min(len(bins_use) - 1, max_bin_index + HALF_WIDTH_FIT)

                xfit = bins_use
                yfit = counts_use
                yerr = poisson_errors(yfit)

                A0 = max(yfit) - np.median(yfit)
                mu0 = max_bin_index
                sigma0 = max(1.5, HALF_WIDTH_FIT / 4)
                b00 = np.median(yfit)
                b10 = 0.0

                lower = [0, xfit.min(), 0.5, -np.inf, -np.inf]
                upper = [np.inf, xfit.max(), HALF_WIDTH_FIT, np.inf, np.inf]

                popt, pcov = curve_fit(
                    gaussian_plus_linear,
                    xfit, yfit,
                    p0=[A0, mu0, sigma0, b00, b10],
                    sigma=yerr,
                    absolute_sigma=True,
                    bounds=(lower, upper),
                    maxfev=20000
                )
                model = gaussian_plus_linear(xfit, *popt)
                chi2_val, ndof, p_value = compute_chi2(yfit, model, yerr, 5)
                perr = np.sqrt(np.diag(pcov))

                xdense = np.linspace(xfit.min(), xfit.max(), 400)
                ydense = gaussian_plus_linear(xdense, *popt)
                peak_plot = popt[1]
                peak_plot_err = perr[1]

                Ba133_peak_locations[f"{date}-{sp_type}", index[1]] = peak_plot
                Ba133_peak_errors[f"{date}-{sp_type}", index[1]] = peak_plot_err

                fig, ax = plt.subplots()
                ax.bar(bins_use, counts_use, width=1.0, color=CBLUE, edgecolor=None, linewidth=0)
                ax.set_title(f"{date}: Ba133 ({sp_type})")
                ax.set_xlabel("Bin")
                ax.set_ylabel("Counts")
                ax.plot(xdense, ydense, color=CRED, lw=2.5, label="Gaussian + linear fit")
                ax.axvline(peak_plot, color=CGREEN, ls="--", lw=2,
                            label=f"Peak = {peak_plot:.2f} ± {peak_plot_err:.2f}")

                textbox = (
                    f"$\\mu$ = {peak_plot:.2f} ± {peak_plot_err:.2f}\n"
                    f"$\\sigma$ = {popt[2]:.2f} ± {perr[2]:.2f} bins\n"
                    f"$\\chi^2$/ndof = {chi2_val:.2f}/{ndof}\n"
                    f"$p$ = {p_value:.3f}\n"
                )
                ax.text(
                    0.98, 0.95, textbox, transform=ax.transAxes,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
                )
                plt.tight_layout(rect=[0, 0, 1, 0.97])
                plt.savefig(os.path.join(output_dir, f"{date}-local_peak{index[1]}-{2048//FACTOR}.png"), dpi=200)
                plt.close()
    
    #############
    # CALIBRATE #
    #############
    calibrations = {}
    calibration_errors = {}
    calibration_cross_covs = {}
    for date in all_dates:
        for sp_type in ["scatter", "recoil"]:
            output_dir = OUTPUT_DIR + "\\calibrations"

            xfit = np.array([511, 356, 160, 81])
            yfit = np.array([Na22_peak_locations[f"{date}-{sp_type}", 511], Ba133_peak_locations[f"{date}-{sp_type}", 356], Ba133_peak_locations[f"{date}-{sp_type}", 160], Ba133_peak_locations[f"{date}-{sp_type}", 81]])
            yerr = np.array([Na22_peak_errors[f"{date}-{sp_type}", 511], Ba133_peak_errors[f"{date}-{sp_type}", 356], Ba133_peak_errors[f"{date}-{sp_type}", 160], Ba133_peak_errors[f"{date}-{sp_type}", 81]])
            
            popt, pcov = curve_fit(
                weighted_linear,
                xfit, yfit,
                sigma=yerr,
                absolute_sigma=True,
                bounds=(0, 2048//FACTOR),
                maxfev=20000
            )
            # Swap x and y !!! BE CAREFUL
            m = 1 / popt[0]
            b = - popt[1] / popt[0]
            model = weighted_linear(yfit, m, b)
            chi2_val, ndof, p_value = compute_chi2(xfit, model, m*yerr, 2)

            perr = np.sqrt(np.diag(pcov))
            m_err = m * m * perr[0]
            b_err = m * perr[1] - m * b * perr[0] 

            calibrations[f"{date}-{sp_type}"] = m, b
            calibration_errors[f"{date}-{sp_type}"] = m_err, b_err
            calibration_cross_covs[f"{date}-{sp_type}"] = pcov[0,1]

            xdense = np.linspace(yfit.min(), yfit.max(), 400)
            ydense = weighted_linear(xdense, m, b)

            fig, ax = plt.subplots()
            ax.scatter(yfit, xfit, color=CPURPLE)
            ax.set_title(f"{date}: Calibration")
            ax.set_xlabel("Bin")
            ax.set_ylabel("Energy [keV]")
            ax.plot(xdense, ydense, color=CRED, lw=2.5, label="Linear Fit")

            textbox = (
                f"$m$ = {m:.2f} ± {m_err:.2f} keV/bin\n"
                f"$b$ = {b:.2f} ± {b_err:.2f} bins\n"
                f"$\\chi^2$/ndof = {chi2_val:.2f}/{ndof}\n"
                f"$p$ = {p_value:.3f}\n"
            )
            ax.text(
                0.02, 0.95, textbox, transform=ax.transAxes,
                ha="left", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
            )
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(os.path.join(output_dir, f"{date}-{sp_type}-{2048//FACTOR}.png"), dpi=200)
            plt.close()

    ###############################
    # Cs137 Raw After Calibration #
    ###############################

    for sp_type in ("recoil", "scatter"):
        for date in all_dates:
            output_dir = OUTPUT_DIR + DIR_SOURCE["Cs137"] + DIR_TYPE[sp_type] + DIR_ANALYSIS["raw"]
            key = (sp_type, "Cs137")
            sp_list = spectra_maps[date][key]
            for sp in sp_list:
                min_bin = get_low_bin_cutoff(sp.date, sp.source, sp.spec_type, sp.angle, LOW_BIN_CUTOFFS)
                mask = sp.bins >= min_bin

                m, b = calibrations[f"{date}-{sp_type}"]
                energies_use = m * sp.bins[mask] + b
                counts_use = sp.counts[mask]

                fig, ax = plt.subplots()
                ax.bar(energies_use, counts_use, width=m*1.0, color=CBLUE, edgecolor=None, linewidth=0)
                ax.set_title(f"{date}: Cs137 {sp.angle} ({sp_type})")
                ax.set_xlabel("Energy [keV]")
                ax.set_ylabel("Counts")
                plt.tight_layout(rect=[0, 0, 1, 0.97])
                plt.savefig(os.path.join(output_dir, f"{date}-{sp.angle}-calibrated-rebinned{2048//FACTOR}.png"), dpi=200)

    #####################
    # Cs137 Single Peak #
    #####################
    Cs137_peak_locations = {}
    Cs137_peak_errors = {}
    for date in all_dates:
        for sp_type in ("recoil", "scatter"):
            output_dir = OUTPUT_DIR + DIR_SOURCE["Cs137"] + DIR_TYPE[sp_type] + DIR_ANALYSIS["local"]
            sp_list = spectra_maps[date][(sp_type, "Cs137")]

            for sp_Cs137 in sp_list:
                min_bin = get_low_bin_cutoff(sp_Cs137.date, sp_Cs137.source, sp_Cs137.spec_type, sp_Cs137.angle, LOW_BIN_CUTOFFS)
                mask = sp_Cs137.bins >= min_bin
                
                counts = sp_Cs137.counts
                bins = sp_Cs137.bins

                max_bin_index = int(np.argmax(counts[mask])) + min_bin

                bins_use = bins[max_bin_index-HALF_WIDTH_FIT:max_bin_index+HALF_WIDTH_FIT+1]
                counts_use = counts[max_bin_index-HALF_WIDTH_FIT:max_bin_index+HALF_WIDTH_FIT+1]

                left = max(0, max_bin_index - HALF_WIDTH_FIT)
                right = min(len(bins_use) - 1, max_bin_index + HALF_WIDTH_FIT)

                xfit = bins_use
                yfit = counts_use
                yerr = poisson_errors(yfit)

                A0 = max(yfit) - np.median(yfit)
                mu0 = max_bin_index
                sigma0 = max(1.5, HALF_WIDTH_FIT / 4)
                b00 = np.median(yfit)
                b10 = 0.0

                lower = [0, xfit.min(), 0.5, -np.inf, -np.inf]
                upper = [np.inf, xfit.max(), HALF_WIDTH_FIT, np.inf, np.inf]

                popt, pcov = curve_fit(
                    gaussian_plus_linear,
                    xfit, yfit,
                    p0=[A0, mu0, sigma0, b00, b10],
                    sigma=yerr,
                    absolute_sigma=True,
                    bounds=(lower, upper),
                    maxfev=20000
                )
                model = gaussian_plus_linear(xfit, *popt)
                chi2_val, ndof, p_value = compute_chi2(yfit, model, yerr, 5)
                perr = np.sqrt(np.diag(pcov))

                xdense = np.linspace(xfit.min(), xfit.max(), 400)
                ydense = gaussian_plus_linear(xdense, *popt)
                peak_plot = popt[1]
                peak_plot_err = perr[1]

                m, b = calibrations[f"{date}-{sp_type}"]
                m_err, b_err = calibration_errors[f"{date}-{sp_type}"]
                cross_cov = calibration_cross_covs[f"{date}-{sp_type}"]

                peak_energy, peak_energy_err = b_to_e(peak_plot, peak_plot_err, m, b, m_err, b_err, cross_cov)
                sigma, sigma_err = b_to_e(popt[2], perr[2], m, b, m_err, b_err, cross_cov)

                Cs137_peak_locations[sp_Cs137.angle, sp_type] = peak_energy
                Cs137_peak_errors[sp_Cs137.angle, sp_type] = peak_energy_err

                fig, ax = plt.subplots()
                ax.bar(m*bins_use+b, counts_use, width=m*1.0, color=CBLUE, edgecolor=None, linewidth=0)
                ax.set_title(f"{date}: Cs137 {sp_Cs137.angle} ({sp_type})")
                ax.set_xlabel("Energy [keV]")
                ax.set_ylabel("Counts")
                ax.plot(m*xdense+b, ydense, color=CRED, lw=2.5, label="Gaussian + linear fit")
                ax.axvline(peak_energy, color=CGREEN, ls="--", lw=2,
                            label=f"Peak = {peak_energy:.2f} ± {peak_energy_err:.2f}")

                textbox = (
                    f"$\\mu$ = {peak_energy:.2f} ± {peak_energy_err:.2f} keV\n"
                    f"$\\sigma$ = {sigma:.2f} ± {sigma_err:.2f} keV\n"
                    f"$\\chi^2$/ndof = {chi2_val:.2f}/{ndof}\n"
                    f"$p$ = {p_value:.3f}\n"
                )
                ax.text(
                    0.98, 0.95, textbox, transform=ax.transAxes,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
                )
                plt.tight_layout(rect=[0, 0, 1, 0.97])
                plt.savefig(os.path.join(output_dir, f"{date}-{sp_Cs137.angle}-local_peak-{2048//FACTOR}.png"), dpi=200)
                plt.close()
    
    ####################
    # Experiment Goals #
    ####################

    # Energy Sum

    peaks_by_angle = {}
    output_dir = OUTPUT_DIR + "\\final"
    for (theta, sp_type) in Cs137_peak_locations:
        print(theta)
        if theta in peaks_by_angle:
            peaks_by_angle[theta].append((sp_type, Cs137_peak_locations[theta, sp_type], Cs137_peak_errors[theta, sp_type]))
        else:
            peaks_by_angle[theta] = [(sp_type, Cs137_peak_locations[theta, sp_type], Cs137_peak_errors[theta, sp_type])]

    angles = []
    Erecoils = []
    Erecoilerrs = []
    Escatters = []
    Escattererrs = []
    Etots = []
    Eerrs = []
    for theta in peaks_by_angle:
        angles.append(theta)
        Erecoils.append(peaks_by_angle[theta][0][1] if peaks_by_angle[theta][0][0] == "recoil" else peaks_by_angle[theta][1][1])
        Erecoilerrs.append(peaks_by_angle[theta][0][2] if peaks_by_angle[theta][0][0] == "recoil" else peaks_by_angle[theta][1][2])
        Escatters.append(peaks_by_angle[theta][0][1] if peaks_by_angle[theta][0][0] == "scatter" else peaks_by_angle[theta][1][1])
        Escattererrs.append(peaks_by_angle[theta][0][2] if peaks_by_angle[theta][0][0] == "scatter" else peaks_by_angle[theta][1][2])
        Etots.append(peaks_by_angle[theta][0][1]+peaks_by_angle[theta][1][1])
        Eerrs.append(np.sqrt(peaks_by_angle[theta][0][2]**2 + peaks_by_angle[theta][1][2]**2))
    
    angles = np.array(angles)
    Erecoils = np.array(Erecoils)
    Erecoilerrs = np.array(Erecoilerrs)
    Escatters = np.array(Escatters)
    Escattererrs = np.array(Escattererrs)
    Etots = np.array(Etots)
    Eerrs = np.array(Eerrs)

    mask = ~np.isclose(angles, 310)
    mean_energy, mean_energy_err = mean_with_propagated_uncertainty(Etots, Eerrs)
    mean_energy_no_310, mean_energy_err_no_310 = mean_with_propagated_uncertainty(Etots[mask], Eerrs[mask])

    fig, ax = plt.subplots()
    ax.errorbar(
        angles, Etots, yerr=Eerrs,
        fmt='o', color=CBLUE, ecolor=CBLUE, capsize=4, markersize=8,
        label="Measured sums"
    )
    ax.axhline(mean_energy, color=CRED, lw=2.5, ls='--',
               label=f"Mean: {mean_energy:.1f} ± {mean_energy_err:.1f} keV")
    ax.axhline(mean_energy_no_310, color=CGREEN, lw=2.5, ls='--',
               label=f"Mean w/o 310: {mean_energy_no_310:.1f} ± {mean_energy_err_no_310:.1f} keV")
    ax.axhline(661.567, color=CPURPLE, lw=2.5, ls='--',
               label=f"Expected: {661.657:.1f} keV")
    ax.set_xlabel("Scattering angle [deg]")
    ax.set_ylabel(r"$E_{\gamma} + E_{e}$ [keV]")
    ax.set_title("Sum of scatter and recoil energies vs angle")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"energy_sum_vs_angle-{2048//FACTOR}.png"), dpi=200)
    plt.close()

    # Scatter Energy Plot

    x = one_minus_cos_theta(angles)
    y, yerr = inverse_with_error(Escatters, Escattererrs)

    x_theory = np.linspace(0.0, max(1.05 * np.max(x), 2.05), 500)
    y_theory = (1.0 / CS137_ENERGY_KEV) + (1.0 / ELECTRON_REST_ENERGY_KEV) * x_theory

    popt, pcov = curve_fit(
        weighted_linear,
        x, y,
        sigma=yerr,
        absolute_sigma=True,
        bounds=(0, 2048//FACTOR),
        maxfev=20000
    )
    perr = np.sqrt(np.diag(pcov))

    measured_Cs137_energy, measured_Cs137_energy_err = inverse_with_error(popt[1], perr[1])
    measured_electron_energy, measured_electron_energy_err = inverse_with_error(popt[0], perr[0])

    yfit = weighted_linear(x, *popt)
    
    chi2_val, ndof, p_value = compute_chi2(y, yfit, yerr, 2)

    fig, ax = plt.subplots()
    ax.errorbar(
        x, y, yerr=yerr,
        fmt='o', color=CBLUE, ecolor=CBLUE, capsize=4, markersize=8,
        label="Measured data"
    )
    ax.plot(x_theory, y_theory, color=CRED, lw=2.5, label="Compton prediction")
    ax.plot(x_theory, weighted_linear(x_theory, *popt), color=CGREEN, lw=2.5, label="Linear fit")

    textbox = (
        f"$\\chi^2$/ndof = {chi2_val:.2f}/{ndof}\n"
        f"$p$ = {p_value:.3f}"
    )
    ax.text(
        0.98, 0.95, textbox,
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
    )

    ax.set_xlabel(r"$1-\cos\theta$")
    ax.set_ylabel(r"$1/E_{\gamma}$ [keV$^{-1}$]")
    ax.set_title(r"Inverse scattered-photon energy vs $1-\cos\theta$")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_energy_vs_angle-{2048//FACTOR}.png"), dpi=200)
    plt.close()

    # Recoil Energy Plot

    x = 1 / one_minus_cos_theta(angles)
    y, yerr = inverse_with_error(Erecoils, Erecoilerrs)

    x_theory = np.linspace(0.0, max(1.05 * np.max(x), 2.05), 500)
    y_theory = (1.0 / CS137_ENERGY_KEV) + (ELECTRON_REST_ENERGY_KEV / CS137_ENERGY_KEV**2) * x_theory
    yfit = (1.0 / CS137_ENERGY_KEV) + (ELECTRON_REST_ENERGY_KEV / CS137_ENERGY_KEV**2) * x

    popt, pcov = curve_fit(
        weighted_linear,
        x, y,
        sigma=yerr,
        absolute_sigma=True,
        bounds=(0, 2048//FACTOR),
        maxfev=20000
    )
    perr = np.sqrt(np.diag(pcov))

    measured_Cs137_energy, measured_Cs137_energy_err = inverse_with_error(popt[1], perr[1])
    measured_electron_energy, measured_electron_energy_err = popt[0] * popt[1]**2, perr[0] * popt[1]**2

    yfit = weighted_linear(x, *popt)
    
    chi2_val, ndof, p_value = compute_chi2(y, yfit, yerr, 2)


    fig, ax = plt.subplots()
    ax.errorbar(
        x, y, yerr=yerr,
        fmt='o', color=CBLUE, ecolor=CBLUE, capsize=4, markersize=8,
        label="Measured data"
    )
    ax.plot(x_theory, y_theory, color=CRED, lw=2.5, label="Compton prediction")
    ax.plot(x_theory, weighted_linear(x_theory, *popt), color=CGREEN, lw=2.5, label="Linear Fit")

    # textbox = (
    #     r"$\frac{1}{E_\gamma'} = \frac{1}{E_0} + \frac{1}{m_ec^2}(1-\cos\theta)$" "\n"
    #     f"$E_0$ = {CS137_ENERGY_KEV:.3f} keV\n"
    #     f"$m_ec^2$ = {ELECTRON_REST_ENERGY_KEV:.3f} keV"
    # )
    textbox = (
        f"$\\chi^2$/ndof = {chi2_val:.2f}/{ndof}\n"
        f"$p$ = {p_value:.3f}"
    )
    ax.text(
        0.98, 0.05, textbox,
        transform=ax.transAxes, ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
    )

    ax.set_xlabel(r"$(1-\cos\theta)^{-1}$")
    ax.set_ylabel(r"$1/E_{e}$ [keV$^{-1}$]")
    ax.set_title(r"Inverse recoil-electron energy vs $(1-\cos\theta)^{-1}$")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"recoil_energy_vs_angle-{2048//FACTOR}.png"), dpi=200)
    plt.close()