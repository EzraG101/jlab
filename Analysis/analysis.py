import os
import re
import math
import glob
from dataclasses import dataclass, field
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
# Adjust as needed depending on which peaks you expect to use.
# ============================================================

KNOWN_PEAKS_KEV = {
    "Na22": np.array([511.0, 1274.537]),  # common visible peaks
    # Common Ba133 gamma/X-ray lines; use subset depending on your detector/spectrum
    "Ba133": np.array([81.0, 276.4, 302.85, 356.01, 383.85]),
    "Cs137": np.array([661.657]),  # not for calibration, but useful sanity check
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
    spec_type: str  # scatter or recoil
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
    slope: float             # keV / bin
    intercept: float         # keV
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

    # In many .Spe files, next line is channel start/end like "0 2047"
    counts = []
    started_numbers = False
    for line in lines[data_start + 1:]:
        if line.startswith("$") and started_numbers:
            break
        if not line:
            continue

        parts = line.split()
        # Detect "0 2047" line or count lines
        # Once we begin reading numbers, interpret them as counts
        if len(parts) == 2 and not started_numbers:
            # likely channel bounds, skip
            try:
                int(parts[0]); int(parts[1])
                continue
            except ValueError:
                pass

        try:
            # ORTEC usually has one integer count per line
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

        # Rebin 2048 -> 1024 if applicable
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
    """
    Fit y = m x + b with weighting by yerr.
    Returns m, b, covariance, chi2, ndof, p.
    """
    popt, pcov = curve_fit(weighted_linear, x, y, sigma=yerr, absolute_sigma=True)
    yfit = weighted_linear(x, *popt)
    chi2_val, ndof, p_value = compute_chi2(y, yfit, yerr, 2)
    return popt, pcov, chi2_val, ndof, p_value

# ============================================================
# Peak finding and fitting
# ============================================================

def poisson_errors(counts: np.ndarray):
    # Conservative nonzero error
    return np.sqrt(np.maximum(counts, 1.0))

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
):
    """
    1) Find candidate peaks.
    2) Fit each candidate with Gaussian + linear background in a local window.
    3) Return fit results.

    Generates:
      - candidate overview plot
      - zoom plots for each fit
      - final peak plot
    """
    os.makedirs(output_dir, exist_ok=True)

    if prominence is None:
        prominence = max(5.0, 0.03 * np.max(counts))
    if height is None:
        height = max(5.0, 0.05 * np.max(counts))

    peak_indices, properties = find_peaks(
        counts,
        prominence=prominence,
        height=height,
        distance=distance
    )

    # If too many, keep strongest by prominence
    if len(peak_indices) > max_peaks:
        prominences = properties["prominences"]
        order = np.argsort(prominences)[::-1][:max_peaks]
        peak_indices = peak_indices[order]
        peak_indices = np.sort(peak_indices)

    # --- Plot candidate peaks and fit windows
    fig, ax = plt.subplots()
    ax.bar(bins, counts, width=1.0, color=CBLUE, edgecolor=None, linewidth=0)
    ax.plot(bins[peak_indices], counts[peak_indices], 'o', color=CRED, markersize=8, label="Peak candidates")

    for p in peak_indices:
        left = max(0, p - fit_half_width)
        right = min(len(bins) - 1, p + fit_half_width)
        ax.axvspan(left, right, color=CORANGE, alpha=0.15)

    ax.set_xlabel("Bin")
    ax.set_ylabel("Counts")
    ax.set_title(f"{title_prefix} Peak candidates")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{sanitize_filename(title_prefix)}_peak_candidates.png"), dpi=200)
    plt.close(fig)

    fit_results = []

    for i, p in enumerate(peak_indices):
        left = max(0, p - fit_half_width)
        right = min(len(bins) - 1, p + fit_half_width)

        xfit = bins[left:right+1]
        yfit = counts[left:right+1]
        yerr = poisson_errors(yfit)

        # Initial guesses
        A0 = max(yfit) - np.median(yfit)
        mu0 = bins[p]
        sigma0 = max(1.5, fit_half_width / 4)
        b00 = np.median(yfit)
        b10 = 0.0

        # Bounds
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
            model = np.full_like(xfit, np.nan, dtype=float)

        result = PeakFitResult(
            candidate_bin=bins[p],
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
            fit_range=(left, right),
            success=success,
            covariance=pcov
        )
        fit_results.append(result)

        # --- Plot local fit
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

    # --- Final peak plot
    fig, ax = plt.subplots()
    ax.bar(bins, counts, width=1.0, color=CBLUE, edgecolor=None, linewidth=0, label="Histogram")

    valid_results = [r for r in fit_results if r.success]
    for r in valid_results:
        ax.axvline(r.fit_center, color=CRED, lw=2)
        ax.plot(r.fit_center, np.interp(r.fit_center, bins, counts), 'o', color=CRED, markersize=8)

    ax.set_xlabel("Bin")
    ax.set_ylabel("Counts")
    ax.set_title(f"{title_prefix} fitted peaks")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{sanitize_filename(title_prefix)}_fitted_peaks.png"), dpi=200)
    plt.close(fig)

    return fit_results

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s.strip())

# ============================================================
# Calibration helpers
# ============================================================

def select_calibration_peaks(
    source: str,
    fit_results: List[PeakFitResult],
    known_energies: np.ndarray
):
    """
    Match found peaks to known calibration energies.
    Since your spectra have well-separated peaks and not too many peaks,
    this uses a simple rule:
      - sort found peaks by bin position
      - sort known energies by energy
      - take the same number from each, truncated to min length

    If you want, later we can replace this with a manual peak assignment step.
    """
    valid = [r for r in fit_results if r.success]
    valid = sorted(valid, key=lambda r: r.fit_center)
    known = np.sort(np.array(known_energies, dtype=float))

    n = min(len(valid), len(known))
    if n < 2:
        raise ValueError(f"Need at least 2 matched peaks to calibrate {source}, but got {n}")

    peak_bins = np.array([r.fit_center for r in valid[:n]])
    peak_bin_errs = np.array([r.fit_center_err for r in valid[:n]])
    energies = known[:n]

    return peak_bins, peak_bin_errs, energies

def calibrate_day_type(
    date: str,
    spec_type: str,
    source_peak_results: Dict[str, List[PeakFitResult]],
    output_dir: str = "plots",
    known_peaks_dict: Dict[str, np.ndarray] = KNOWN_PEAKS_KEV
) -> CalibrationResult:
    """
    Build calibration using Na22 + Ba133 for one day and one type.
    Fits energy = slope * bin + intercept.
    """
    bins_all = []
    bin_errs_all = []
    energies_all = []

    for source in ["Ba133", "Na22"]:
        if source not in source_peak_results:
            continue
        peak_bins, peak_bin_errs, energies = select_calibration_peaks(
            source,
            source_peak_results[source],
            known_peaks_dict[source]
        )
        bins_all.append(peak_bins)
        bin_errs_all.append(peak_bin_errs)
        energies_all.append(energies)

    if len(bins_all) == 0:
        raise ValueError(f"No calibration source peaks found for {date} {spec_type}")

    bins_all = np.concatenate(bins_all)
    bin_errs_all = np.concatenate(bin_errs_all)
    energies_all = np.concatenate(energies_all)

    # Since x-errors exist (bin errors), a rigorous treatment would use ODR.
    # For now, propagate x-errors into y-errors using an iterative slope estimate.
    # First rough linear fit with uniform uncertainties:
    p0 = np.polyfit(bins_all, energies_all, 1)
    slope0, intercept0 = p0[0], p0[1]

    yerr_eff = np.maximum(np.abs(slope0) * bin_errs_all, 0.5)  # keV
    popt, pcov, chi2_val, ndof, p_value = fit_weighted_linear(bins_all, energies_all, yerr_eff)
    slope, intercept = popt
    perr = np.sqrt(np.diag(pcov))

    # Plot calibration
    fig, ax = plt.subplots()
    ax.errorbar(
        bins_all, energies_all,
        xerr=bin_errs_all, yerr=np.abs(slope) * bin_errs_all,
        fmt='o', color=CBLUE, ecolor=CBLUE, capsize=3, label="Calibration peaks"
    )

    xdense = np.linspace(0, max(1050, 1.05 * np.max(bins_all)), 400)
    ax.plot(xdense, weighted_linear(xdense, slope, intercept), color=CRED, lw=2.5, label="Linear calibration fit")

    textbox = (
        f"$E = m b + c$\n"
        f"$m$ = {slope:.4f} ± {perr[0]:.4f} keV/bin\n"
        f"$c$ = {intercept:.2f} ± {perr[1]:.2f} keV\n"
        f"$\\chi^2$/ndof = {chi2_val:.2f}/{ndof}\n"
        f"$p$ = {p_value:.3f}"
    )
    ax.text(
        0.98, 0.05, textbox, transform=ax.transAxes,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
    )

    ax.set_xlabel("Peak location [bin]")
    ax.set_ylabel("Energy [keV]")
    ax.set_title(f"Calibration: {date} {spec_type}")
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
    var(E) = (b^2 var(m) + var(c) + m^2 var(b) + 2 b cov(m,c))
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
    For scatter:
      choose the strongest/most obvious valid peak, typically only one.

    For recoil:
      often two peaks, and the higher-energy one is the unwanted direct-photon peak.
      We return:
         desired_peak = lower-bin peak
         sanity_peak  = highest-bin peak (if present)
    """
    valid = [r for r in fit_results if r.success]
    if len(valid) == 0:
        raise ValueError("No successful peak fits for Cs137 spectrum")

    valid_sorted_bin = sorted(valid, key=lambda r: r.fit_center)
    valid_sorted_amp = sorted(valid, key=lambda r: r.amplitude, reverse=True)

    if spec_type == "scatter":
        # choose strongest amplitude peak
        return valid_sorted_amp[0], None

    elif spec_type == "recoil":
        if len(valid_sorted_bin) == 1:
            return valid_sorted_bin[0], None
        # desired = lower-energy/lower-bin peak, sanity = highest-bin peak
        return valid_sorted_bin[0], valid_sorted_bin[-1]

    else:
        raise ValueError(f"Unknown spec_type: {spec_type}")

# ============================================================
# Main analysis workflow
# ============================================================

def analyze_compton_data(
    data_dir: str,
    output_dir: str = "plots",
    known_peaks_dict: Dict[str, np.ndarray] = KNOWN_PEAKS_KEV
):
    os.makedirs(output_dir, exist_ok=True)

    spectra = load_all_spectra(data_dir)

    # Organize by date and type
    by_date_type_source: Dict[Tuple[str, str, str], List[Spectrum]] = {}
    for sp in spectra:
        key = (sp.date, sp.spec_type, sp.source)
        by_date_type_source.setdefault(key, []).append(sp)

    # --------------------------------------------------------
    # 1) Fit peaks in calibration spectra and build calibrations
    # --------------------------------------------------------
    calibrations: Dict[Tuple[str, str], CalibrationResult] = {}

    all_dates = sorted(set(sp.date for sp in spectra))
    for date in all_dates:
        for spec_type in ["scatter", "recoil"]:
            source_peak_results = {}

            for source in ["Na22", "Ba133"]:
                key = (date, spec_type, source)
                if key not in by_date_type_source:
                    continue

                # Usually there should be one calibration file per date/type/source
                # If there are multiple, analyze first for now.
                sp = by_date_type_source[key][0]
                peak_results = find_and_fit_peaks(
                    sp.bins,
                    sp.counts,
                    title_prefix=f"{date}_{source}_{spec_type}",
                    output_dir=output_dir,
                    source_hint=source
                )
                source_peak_results[source] = peak_results

            if len(source_peak_results) >= 1:
                try:
                    cal = calibrate_day_type(
                        date=date,
                        spec_type=spec_type,
                        source_peak_results=source_peak_results,
                        output_dir=output_dir,
                        known_peaks_dict=known_peaks_dict
                    )
                    calibrations[(date, spec_type)] = cal
                    print(f"[OK] Calibration built for {date} {spec_type}")
                    print(f"     E = ({cal.slope:.5f} ± {cal.slope_err:.5f}) * bin + ({cal.intercept:.3f} ± {cal.intercept_err:.3f}) keV")
                except Exception as e:
                    print(f"[WARN] Could not calibrate {date} {spec_type}: {e}")

    # --------------------------------------------------------
    # 2) Analyze Cs137 spectra
    # --------------------------------------------------------
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

        peak_results = find_and_fit_peaks(
            sp.bins,
            sp.counts,
            title_prefix=f"{sp.date}_{sp.source}_{sp.spec_type}_{sp.angle}",
            output_dir=output_dir,
            source_hint="Cs137"
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

    # --------------------------------------------------------
    # 3) Combine scatter and recoil by angle and plot energy sum
    # --------------------------------------------------------
    # Match by (date, angle)
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

    # --------------------------------------------------------
    # 4) Print summary tables
    # --------------------------------------------------------
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
    DATA_DIR = "data"       # <-- folder containing .Spe files
    OUTPUT_DIR = "plots"

    results = analyze_compton_data(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        known_peaks_dict=KNOWN_PEAKS_KEV
    )