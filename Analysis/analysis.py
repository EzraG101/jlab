import os
import re
import math
import glob
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import chi2

# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 14,
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
# User-configurable known source energies (keV)
# Keep only peaks you realistically expect to use.
# ============================================================

KNOWN_PEAKS_KEV = {
    "Na22": np.array([511.0, 1274.537]),
    "Ba133": np.array([81.0, 276.4, 302.85, 356.01, 383.85]),
    "Cs137": np.array([661.657]),
}

# ============================================================
# Optional low-bin cutoffs to suppress low-energy noise
# Values are in REBINNED bins (1024-bin spectra).
# Add/edit entries as needed.
# ============================================================

LOW_BIN_CUTOFFS = {
    ("03-10", "Na22", "scatter"): 80,
    # Add more if needed, e.g.
    # ("03-10", "Ba133", "scatter"): 40,
    # ("03-11", "Cs137", "recoil"): 20,
}

def get_low_bin_cutoff(
    date: str,
    source: str,
    spec_type: str,
    angle: Optional[float] = None
) -> int:
    return LOW_BIN_CUTOFFS.get((date, source, spec_type), 0)

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

# ============================================================
# File reading and parsing
# ============================================================

def parse_spe_filename(filename: str):
    """
    Expected formats:
      mm-dd-Na22-scatter.Spe
      mm-dd-Ba133-recoil.Spe
      mm-dd-Cs137-scatter-30.Spe
    """
    base = os.path.basename(filename)
    pattern_no_angle = r"^(?P<date>\d{2}-\d{2})-(?P<source>Na22|Ba133|Cs137)-(?P<stype>scatter|recoil)\.Spe$"
    pattern_angle = r"^(?P<date>\d{2}-\d{2})-(?P<source>Na22|Ba133|Cs137)-(?P<stype>scatter|recoil)-(?P<angle>-?\d+(\.\d+)?)\.Spe$"

    m = re.match(pattern_angle, base, re.IGNORECASE)
    if m:
        return (
            m.group("date"),
            m.group("source"),
            m.group("stype").lower(),
            float(m.group("angle"))
        )

    m = re.match(pattern_no_angle, base, re.IGNORECASE)
    if m:
        return (
            m.group("date"),
            m.group("source"),
            m.group("stype").lower(),
            None
        )

    raise ValueError(f"Filename does not match expected format: {filename}")

def read_spe_file(filepath: str) -> np.ndarray:
    """
    Reads counts from a common ORTEC-style .Spe file.
    Assumes counts live between '$DATA:' and the next section beginning with '$'.
    """
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
            val = float(parts[0])
            counts.append(val)
            started_numbers = True
        except Exception:
            continue

    if len(counts) == 0:
        raise ValueError(f"No counts found in $DATA section for {filepath}")

    return np.array(counts, dtype=float)

def rebin_counts(counts: np.ndarray, factor: int = 2) -> np.ndarray:
    if len(counts) % factor != 0:
        raise ValueError(f"Counts length {len(counts)} not divisible by rebin factor {factor}")
    return counts.reshape(-1, factor).sum(axis=1)

def load_all_spectra(data_dir: str) -> List[Spectrum]:
    spectra = []
    for filepath in sorted(glob.glob(os.path.join(data_dir, "*.Spe"))):
        date, source, spec_type, angle = parse_spe_filename(os.path.basename(filepath))
        counts_raw = read_spe_file(filepath)
        bins_raw = np.arange(len(counts_raw), dtype=float)

        if len(counts_raw) == 2048:
            counts = rebin_counts(counts_raw, factor=2)
        elif len(counts_raw) == 1024:
            counts = counts_raw.copy()
        else:
            raise ValueError(
                f"Unexpected number of channels in {filepath}: {len(counts_raw)}. "
                "Expected 2048 or 1024."
            )

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
# Models and fit utilities
# ============================================================

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

def fit_weighted_linear(x, y, yerr):
    popt, pcov = curve_fit(weighted_linear, x, y, sigma=yerr, absolute_sigma=True)
    yfit = weighted_linear(x, *popt)
    chi2_val, ndof, p_value = compute_chi2(y, yfit, yerr, 2)
    return popt, pcov, chi2_val, ndof, p_value

# ============================================================
# Peak finding and fitting
# ============================================================

def find_and_fit_peaks(
    bins: np.ndarray,
    counts: np.ndarray,
    title_prefix: str = "",
    output_dir: str = "plots",
    prominence: Optional[float] = None,
    height: Optional[float] = None,
    distance: int = 20,
    fit_half_width: int = 12,
    max_peaks: int = 10,
    source_hint: Optional[str] = None,
    min_bin: int = 0,
):
    """
    Find candidate peaks, fit each with Gaussian + linear background,
    and generate diagnostic plots.

    min_bin trims off low-bin noise when searching/fitting peaks.
    """
    os.makedirs(output_dir, exist_ok=True)

    mask = bins >= min_bin
    bins_use = bins[mask]
    counts_use = counts[mask]

    if len(bins_use) == 0:
        raise ValueError(f"No bins remain after applying min_bin={min_bin}")

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
        prominences = properties["prominences"]
        order = np.argsort(prominences)[::-1][:max_peaks]
        peak_indices_local = peak_indices_local[order]
        peak_indices_local = np.sort(peak_indices_local)

    # Candidate overview plot
    fig, ax = plt.subplots()
    ax.bar(bins_use, counts_use, width=1.0, color=CBLUE, edgecolor=None, linewidth=0)

    if len(peak_indices_local) > 0:
        ax.plot(
            bins_use[peak_indices_local],
            counts_use[peak_indices_local],
            'o',
            color=CRED,
            markersize=8,
            label="Peak candidates"
        )

        for p in peak_indices_local:
            left = max(0, p - fit_half_width)
            right = min(len(bins_use) - 1, p + fit_half_width)
            ax.axvspan(bins_use[left], bins_use[right], color=CORANGE, alpha=0.15)

    ax.set_xlabel("Bin")
    ax.set_ylabel("Counts")
    ax.set_title(f"{title_prefix} Peak candidates")
    if min_bin > 0:
        ax.text(
            0.02, 0.95,
            f"Low-bin cutoff applied: bin ≥ {min_bin}",
            transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
        )
    if len(peak_indices_local) > 0:
        ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{sanitize_filename(title_prefix)}_peak_candidates.png"), dpi=200)
    plt.close(fig)

    fit_results = []

    for i, p in enumerate(peak_indices_local):
        left = max(0, p - fit_half_width)
        right = min(len(bins_use) - 1, p + fit_half_width)

        xfit = bins_use[left:right+1]
        yfit = counts_use[left:right+1]
        yerr = poisson_errors(yfit)

        A0 = max(yfit) - np.median(yfit)
        mu0 = bins_use[p]
        sigma0 = max(1.5, fit_half_width / 4)
        b00 = np.median(yfit)
        b10 = 0.0

        lower = [0, xfit.min(), 0.5, -np.inf, -np.inf]
        upper = [np.inf, xfit.max(), fit_half_width, np.inf, np.inf]

        success = True
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

        # Local fit plot
        fig, ax = plt.subplots()
        ax.bar(xfit, yfit, width=1.0, color=CBLUE, edgecolor=None, linewidth=0, label="Data")

        if success:
            xdense = np.linspace(xfit.min(), xfit.max(), 400)
            ax.plot(xdense, gaussian_plus_linear(xdense, *popt), color=CRED, lw=2.5, label="Gaussian + linear fit")
            ax.axvline(popt[1], color=CGREEN, ls="--", lw=2, label=f"Peak = {popt[1]:.2f} ± {perr[1]:.2f} bins")

            textbox = (
                f"$\\mu$ = {popt[1]:.2f} ± {perr[1]:.2f} bins\n"
                f"$\\sigma$ = {popt[2]:.2f} ± {perr[2]:.2f} bins\n"
                f"$\\chi^2$/ndof = {chi2_val:.2f}/{ndof}\n"
                f"$p$ = {p_value:.3f}"
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

        ax.set_xlabel("Bin")
        ax.set_ylabel("Counts")
        ax.set_title(f"{title_prefix} candidate {i+1} local fit")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{sanitize_filename(title_prefix)}_candidate_{i+1}_fit.png"), dpi=200)
        plt.close(fig)

    # Final fitted-peaks plot
    fig, ax = plt.subplots()
    ax.bar(bins_use, counts_use, width=1.0, color=CBLUE, edgecolor=None, linewidth=0, label="Histogram")

    valid_results = [r for r in fit_results if r.success]
    valid_results = sorted(valid_results, key=lambda r: r.fit_center)
    for i, r in enumerate(valid_results):
        yval = np.interp(r.fit_center, bins_use, counts_use)
        ax.axvline(r.fit_center, color=CRED, lw=2)
        ax.plot(r.fit_center, yval, 'o', color=CRED, markersize=8)
        ax.text(
            r.fit_center,
            yval + 0.03 * np.max(counts_use),
            f"{i}",
            color=CBLACK,
            ha="center",
            va="bottom",
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none")
        )

    if min_bin > 0:
        ax.text(
            0.02, 0.95,
            f"Low-bin cutoff applied: bin ≥ {min_bin}",
            transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
        )

    ax.set_xlabel("Bin")
    ax.set_ylabel("Counts")
    ax.set_title(f"{title_prefix} fitted peaks")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{sanitize_filename(title_prefix)}_fitted_peaks.png"), dpi=200)
    plt.close(fig)

    return fit_results

# ============================================================
# Manual matching for calibration peaks
# ============================================================

def manual_match_calibration_peaks(
    source: str,
    fit_results: List[PeakFitResult],
    known_energies: np.ndarray
):
    """
    Interactively match fitted peaks to known source energies.
    Allows one or more matched peaks from a source.
    """
    valid = [r for r in fit_results if r.success]
    valid = sorted(valid, key=lambda r: r.fit_center)

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
    print("one per line. Press Enter on a blank line when done.")
    print("Example:")
    print("    0 1")
    print("    2 3")

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
            f"Accepted: peak {pidx} -> "
            f"{valid[pidx].fit_center:.2f} ± {valid[pidx].fit_center_err:.2f} bins "
            f"matched to {known_energies[eidx]:.3f} keV"
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

def get_manual_matches(
    match_file,
    key,
    source,
    fit_results,
    known_energies,
    force_rematch=False
):
    if (not force_rematch) and os.path.exists(match_file):
        try:
            peak_bins, peak_bin_errs, energies = load_manual_matches(match_file, key)
            print(f"[INFO] Loaded saved manual matches for {key}")
            return peak_bins, peak_bin_errs, energies
        except KeyError:
            pass
        except ValueError as e:
            print(f"[WARN] Saved matches for {key} are invalid: {e}")
        except Exception as e:
            print(f"[WARN] Could not load saved matches for {key}: {e}")

    print(f"[INFO] No saved matches for {key}; entering interactive matching.")
    peak_bins, peak_bin_errs, energies = manual_match_calibration_peaks(
        source=source,
        fit_results=fit_results,
        known_energies=known_energies
    )

    save_manual_matches(match_file, key, peak_bins, peak_bin_errs, energies)
    print(f"[INFO] Saved manual matches for {key}")

    return peak_bins, peak_bin_errs, energies

# ============================================================
# Calibration helpers
# ============================================================

def calibrate_day_type(
    date: str,
    spec_type: str,
    source_peak_results: Dict[str, List[PeakFitResult]],
    output_dir: str = "plots",
    known_peaks_dict: Dict[str, np.ndarray] = KNOWN_PEAKS_KEV,
    interactive: bool = True,
    match_file: str = "manual_matches.json",
    force_rematch: bool = False
) -> CalibrationResult:
    """
    Build one combined calibration using Na22 + Ba133 for one day and one type.
    Each source may contribute one or more matched peaks.
    Need at least two total matched points across both sources.
    """
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

        except ValueError as e:
            print(f"[WARN] Skipping {source} for {date} {spec_type}: {e}")
        except Exception as e:
            print(f"[WARN] Unexpected issue with {source} for {date} {spec_type}: {e}")

    if len(bins_list) == 0:
        raise ValueError(f"No usable calibration source peaks found for {date} {spec_type}")

    bins_all = np.concatenate(bins_list)
    bin_errs_all = np.concatenate(bin_errs_list)
    energies_all = np.concatenate(energies_list)

    if len(bins_all) < 2:
        raise ValueError(
            f"Need at least two total matched calibration peaks for {date} {spec_type}; "
            f"got only {len(bins_all)}"
        )

    p0 = np.polyfit(bins_all, energies_all, 1)
    slope0, intercept0 = p0[0], p0[1]

    yerr_eff = np.maximum(np.abs(slope0) * bin_errs_all, 0.5)

    popt, pcov, chi2_val, ndof, p_value = fit_weighted_linear(
        bins_all, energies_all, yerr_eff
    )
    slope, intercept = popt
    perr = np.sqrt(np.diag(pcov))

    fig, ax = plt.subplots()
    ax.errorbar(
        bins_all,
        energies_all,
        xerr=bin_errs_all,
        yerr=np.abs(slope) * bin_errs_all,
        fmt='o',
        color=CBLUE,
        ecolor=CBLUE,
        capsize=3,
        label="Selected calibration peaks"
    )

    xdense = np.linspace(0, max(1050, 1.05 * np.max(bins_all)), 400)
    ax.plot(
        xdense,
        weighted_linear(xdense, slope, intercept),
        color=CRED,
        lw=2.5,
        label="Linear fit"
    )

    textbox = (
        f"$E = m b + c$\n"
        f"$m$ = {slope:.4f} ± {perr[0]:.4f} keV/bin\n"
        f"$c$ = {intercept:.2f} ± {perr[1]:.2f} keV\n"
        f"$\\chi^2$/ndof = {chi2_val:.2f}/{ndof}\n"
        f"$p$ = {p_value:.3f}"
    )
    ax.text(
        0.98,
        0.05,
        textbox,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
    )

    ax.set_xlabel("Peak location [bin]")
    ax.set_ylabel("Energy [keV]")
    ax.set_title(f"Calibration: {date} {spec_type} (Na22 + Ba133)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"calibration_{date}_{spec_type}.png"), dpi=200)
    plt.close(fig)

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
    """
    E = m b + c
    var(E) = b^2 var(m) + var(c) + m^2 var(b) + 2 b cov(m,c)
    """
    E = cal.slope * bin_value + cal.intercept
    varE = (
        (bin_value**2) * cal.cov[0, 0]
        + cal.cov[1, 1]
        + (cal.slope**2) * (bin_err**2)
        + 2 * bin_value * cal.cov[0, 1]
    )
    return E, math.sqrt(max(varE, 0.0))

# ============================================================
# Cs137 peak selection logic
# ============================================================

def choose_cs137_peak(
    fit_results: List[PeakFitResult],
    spec_type: str
) -> Tuple[PeakFitResult, Optional[PeakFitResult]]:
    """
    For scatter: choose strongest valid peak.
    For recoil: choose lower-bin peak as desired, highest-bin peak as sanity check.
    """
    valid = [r for r in fit_results if r.success]
    if len(valid) == 0:
        raise ValueError("No successful peak fits for Cs137 spectrum")

    valid_sorted_bin = sorted(valid, key=lambda r: r.fit_center)
    valid_sorted_amp = sorted(valid, key=lambda r: r.amplitude, reverse=True)

    if spec_type == "scatter":
        return valid_sorted_amp[0], None
    elif spec_type == "recoil":
        if len(valid_sorted_bin) == 1:
            return valid_sorted_bin[0], None
        return valid_sorted_bin[0], valid_sorted_bin[-1]
    else:
        raise ValueError(f"Unknown spec_type: {spec_type}")

# ============================================================
# Main analysis workflow
# ============================================================

def analyze_compton_data(
    data_dir: str,
    output_dir: str = "plots",
    known_peaks_dict: Dict[str, np.ndarray] = KNOWN_PEAKS_KEV,
    match_file: str = "manual_matches.json",
    force_rematch: bool = False
):
    os.makedirs(output_dir, exist_ok=True)

    spectra = load_all_spectra(data_dir)

    by_date_type_source: Dict[Tuple[str, str, str], List[Spectrum]] = {}
    for sp in spectra:
        key = (sp.date, sp.spec_type, sp.source)
        by_date_type_source.setdefault(key, []).append(sp)

    calibrations: Dict[Tuple[str, str], CalibrationResult] = {}

    all_dates = sorted(set(sp.date for sp in spectra))
    for date in all_dates:
        for spec_type in ["scatter", "recoil"]:
            source_peak_results = {}

            for source in ["Na22", "Ba133"]:
                key = (date, spec_type, source)
                if key not in by_date_type_source:
                    continue

                sp = by_date_type_source[key][0]

                min_bin = get_low_bin_cutoff(
                    date=sp.date,
                    source=sp.source,
                    spec_type=sp.spec_type,
                    angle=sp.angle
                )

                peak_results = find_and_fit_peaks(
                    sp.bins,
                    sp.counts,
                    title_prefix=f"{date}_{source}_{spec_type}",
                    output_dir=output_dir,
                    source_hint=source,
                    min_bin=min_bin
                )
                source_peak_results[source] = peak_results

            if len(source_peak_results) >= 1:
                try:
                    cal = calibrate_day_type(
                        date=date,
                        spec_type=spec_type,
                        source_peak_results=source_peak_results,
                        output_dir=output_dir,
                        known_peaks_dict=known_peaks_dict,
                        interactive=True,
                        match_file=match_file,
                        force_rematch=force_rematch
                    )
                    calibrations[(date, spec_type)] = cal
                    print(f"[OK] Calibration built for {date} {spec_type}")
                    print(f"     E = ({cal.slope:.5f} ± {cal.slope_err:.5f}) * bin + ({cal.intercept:.3f} ± {cal.intercept_err:.3f}) keV")
                except Exception as e:
                    print(f"[WARN] Could not calibrate {date} {spec_type}: {e}")

    cs_energies: List[CsPeakEnergy] = []
    cs_sanity: List[CsPeakEnergy] = []

    for sp in spectra:
        if sp.source != "Cs137":
            continue

        cal_key = (sp.date, sp.spec_type)
        if cal_key not in calibrations:
            print(f"[WARN] Missing calibration for {sp.filename}")
            continue

        cal = calibrations[cal_key]

        min_bin = get_low_bin_cutoff(
            date=sp.date,
            source=sp.source,
            spec_type=sp.spec_type,
            angle=sp.angle
        )

        peak_results = find_and_fit_peaks(
            sp.bins,
            sp.counts,
            title_prefix=f"{sp.date}_{sp.source}_{sp.spec_type}_{sp.angle}",
            output_dir=output_dir,
            source_hint="Cs137",
            min_bin=min_bin
        )

        try:
            desired_peak, sanity_peak = choose_cs137_peak(peak_results, sp.spec_type)
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

    if len(sums) > 0:
        fig, ax = plt.subplots()
        angles = np.array([x[1] for x in sums], dtype=float)
        Etots = np.array([x[4] for x in sums], dtype=float)
        Eerrs = np.array([x[5] for x in sums], dtype=float)

        ax.errorbar(
            angles, Etots, yerr=Eerrs,
            fmt='o', color=CBLUE, ecolor=CBLUE, capsize=4, markersize=8
        )

        ax.set_xlabel("Scattering angle [deg]")
        ax.set_ylabel(r"$E_{\mathrm{scatter}} + E_{\mathrm{recoil}}$ [keV]")
        ax.set_title("Sum of scatter and recoil energies vs angle")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "energy_sum_vs_angle.png"), dpi=200)
        plt.close(fig)

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
            f"bin = {c.peak_bin:.2f} ± {c.peak_bin_err:.2f}, "
            f"E = {c.energy_keV:.2f} ± {c.energy_err_keV:.2f} keV, "
            f"chi2 p = {c.p_value:.3f}"
        )

    if len(cs_sanity) > 0:
        print("\n=== Cs137 recoil high-energy sanity peaks ===")
        for c in sorted(cs_sanity, key=lambda x: (x.date, x.angle)):
            print(
                f"{c.date} recoil  angle={c.angle:6.1f} deg: "
                f"bin = {c.peak_bin:.2f} ± {c.peak_bin_err:.2f}, "
                f"E = {c.energy_keV:.2f} ± {c.energy_err_keV:.2f} keV"
            )

    print("\n=== Energy sums ===")
    for date, angle, s, r, Etot, Etot_err in sums:
        print(
            f"{date} angle={angle:6.1f} deg: "
            f"Escatter = {s.energy_keV:.2f} ± {s.energy_err_keV:.2f} keV, "
            f"Erecoil = {r.energy_keV:.2f} ± {r.energy_err_keV:.2f} keV, "
            f"sum = {Etot:.2f} ± {Etot_err:.2f} keV"
        )

    return {
        "spectra": spectra,
        "calibrations": calibrations,
        "cs_energies": cs_energies,
        "cs_sanity": cs_sanity,
        "energy_sums": sums
    }

# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    DATA_DIR = "data"                 # folder containing .Spe files
    OUTPUT_DIR = "plots"
    MATCH_FILE = "manual_matches.json"

    results = analyze_compton_data(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        known_peaks_dict=KNOWN_PEAKS_KEV,
        match_file=MATCH_FILE,
        force_rematch=False
    )