"""
theory_comparison.py
--------------------
Computes Klein-Nishina and Thomson differential cross sections at E0 = 661.6 keV
and overlays them on the measured angular scattering rates from evan_angular.ipynb.

Usage:
    python theory_comparison.py

Outputs two figures (saved as PNG):
    angular_comparison_scatter.png  — scatter detector data vs theory
    angular_comparison_recoil.png   — recoil detector data vs theory
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
sys.path.insert(0, str(HERE))
from analysis_evan import read_spe, find_peak_candidates, fit_local_peak, peak_counts_from_fit

# ── constants ─────────────────────────────────────────────────────────────────
E0_KEV    = 661.6          # incident photon energy [keV]
ME_KEV    = 511.0          # electron rest energy [keV]
EPSILON   = E0_KEV / ME_KEV   # reduced photon energy ε = E0 / m_e c^2
R_E_CM    = 2.8179e-13     # classical electron radius [cm]
BARN      = 1e-24          # 1 barn in cm^2

# ── cross-section functions ───────────────────────────────────────────────────

def scattered_energy(theta_rad: np.ndarray) -> np.ndarray:
    """Compton formula: scattered photon energy E' at angle θ [keV]."""
    return E0_KEV / (1.0 + EPSILON * (1.0 - np.cos(theta_rad)))


def dsigma_thomson(theta_rad: np.ndarray) -> np.ndarray:
    """
    Thomson differential cross section dσ/dΩ [cm^2/sr].
    dσ/dΩ = (r_e^2 / 2)(1 + cos^2 θ)
    """
    return 0.5 * R_E_CM**2 * (1.0 + np.cos(theta_rad)**2)


def dsigma_klein_nishina(theta_rad: np.ndarray) -> np.ndarray:
    """
    Klein-Nishina differential cross section dσ/dΩ [cm^2/sr].
    dσ/dΩ = (r_e^2 / 2)(E'/E0)^2 [E'/E0 + E0/E' - sin^2 θ]
    """
    ratio = scattered_energy(theta_rad) / E0_KEV  # E'/E0
    return (
        0.5 * R_E_CM**2 * ratio**2
        * (ratio + 1.0 / ratio - np.sin(theta_rad)**2)
    )


def sigma_total_thomson() -> float:
    """Total Thomson cross section [cm^2]: σ_T = (8π/3) r_e^2."""
    return (8.0 * np.pi / 3.0) * R_E_CM**2


def sigma_total_kn_analytic() -> float:
    """
    Klein-Nishina total cross section [cm^2] via the analytic formula
    (from Heitler, The Quantum Theory of Radiation):

    σ_KN = 2π r_e^2 {
        (1+ε)/ε^2 [ 2(1+ε)/(1+2ε) - ln(1+2ε)/ε ]
        + ln(1+2ε)/(2ε)
        - (1+3ε)/(1+2ε)^2
    }
    """
    e = EPSILON
    ln_term = np.log(1.0 + 2.0 * e)
    term1 = (1.0 + e) / e**2 * (2.0 * (1.0 + e) / (1.0 + 2.0 * e) - ln_term / e)
    term2 = ln_term / (2.0 * e)
    term3 = (1.0 + 3.0 * e) / (1.0 + 2.0 * e)**2
    return 2.0 * np.pi * R_E_CM**2 * (term1 + term2 - term3)


def sigma_total_kn_numerical() -> float:
    """Total KN cross section by numerical integration over solid angle [cm^2]."""
    def integrand(theta):
        return dsigma_klein_nishina(np.array([theta]))[0] * 2.0 * np.pi * np.sin(theta)
    result, _ = quad(integrand, 0.0, np.pi)
    return result


# ── angular data loading (mirrors evan_angular.ipynb) ─────────────────────────

# Files skipped in the original notebook (ambiguous peak selection)
SKIP_FILES = {"03-10-Cs137-Recoil-310.Spe", "03-10-Cs137-Scatter-310.Spe"}

# Manually confirmed peak indices per file (same as evan_angular.ipynb)
PEAK_INDEX_MAP: dict[str, int] = {
    "03-05-Cs137-recoil-280": 2,
    "03-05-Cs137-scatter-280": 1,
    "03-10-Cs137-Recoil-205": 3,
    "03-10-Cs137-Recoil-210": 4,
    "03-10-Cs137-Recoil-215": 4,
    "03-10-Cs137-Recoil-220": 4,
    "03-10-Cs137-Recoil-225": 4,
    "03-10-Cs137-Recoil-250": 3,
    "03-10-Cs137-Scatter-205": 1,
    "03-10-Cs137-Scatter-210": 0,
    "03-10-Cs137-Scatter-215": 0,
    "03-10-Cs137-Scatter-220": 0,
    "03-10-Cs137-Scatter-225": 1,
    "03-10-Cs137-Scatter-250": 1,
    "03-12-Cs137-recoil-50": 0,
    "03-12-Cs137-recoil-80": 2,
    "03-12-Cs137-scatter-50": 4,
    "03-12-Cs137-scatter-80": 2,
}

_FNAME_RE = re.compile(
    r"(\d{2}-\d{2})-Cs137[-_](scatter|recoil|Scatter|Recoil)[-_](\d+)",
    re.IGNORECASE,
)


def parse_cs137_angular_filename(name: str):
    m = _FNAME_RE.search(name)
    if m is None:
        return None
    date = m.group(1)
    det  = m.group(2).lower()
    angle = int(m.group(3))
    return date, det, angle


def effective_angle(dial_deg: float) -> float:
    """Convert dial angle to physical scattering angle (both in degrees)."""
    return 180.0 - abs(dial_deg - 180.0)


def load_angular_data() -> list[dict]:
    """Load and fit all angular SPE files, returning list of result dicts."""
    results = []
    for spe_path in sorted(DATA_DIR.glob("*Cs137*.Spe")) + sorted(DATA_DIR.glob("*Cs137*.spe")):
        if spe_path.name in SKIP_FILES:
            continue
        parsed = parse_cs137_angular_filename(spe_path.stem)
        if parsed is None:
            continue
        date, det, dial_deg = parsed
        stem_key = spe_path.stem

        idx = PEAK_INDEX_MAP.get(stem_key)
        if idx is None:
            continue

        spe = read_spe(str(spe_path), rebin_factor=8)
        cands = find_peak_candidates(
            spe.bins, spe.counts,
            min_bin=40, distance=20, max_peaks=10,
        )
        if idx >= len(cands.candidate_bins):
            print(f"  WARNING: {spe_path.name}: index {idx} out of range, skipping")
            continue

        cb = cands.candidate_bins[idx]
        fit = fit_local_peak(spe.bins, spe.counts, cb, fit_half_width=15)
        net, net_err = peak_counts_from_fit(fit)
        live = spe.live_time_s or 1.0
        rate = net / live
        rate_err = net_err / live if np.isfinite(net_err) else np.nan

        theta_eff = effective_angle(float(dial_deg))
        results.append({
            "filename":      spe_path.name,
            "date":          date,
            "detector":      det,
            "dial_deg":      dial_deg,
            "theta_eff_deg": theta_eff,
            "rate_cps":      rate,
            "rate_err_cps":  rate_err,
            "chi2":          fit.chi2_val,
            "ndof":          fit.ndof,
            "p_value":       fit.p_value,
        })
    return results


# ── normalisation helper ──────────────────────────────────────────────────────

def best_fit_scale(theta_data_deg, rates, rate_errs, theory_func):
    """
    Fit a single overall normalisation constant A such that
    A * theory_func(θ) best matches the data (weighted by rate_errs).
    Returns A.
    """
    theta_rad = np.deg2rad(theta_data_deg)
    theory_vals = theory_func(theta_rad)
    weights = 1.0 / rate_errs**2

    # Analytic weighted least-squares for a single multiplicative factor:
    # A = Σ(w_i * R_i * T_i) / Σ(w_i * T_i^2)
    A = np.sum(weights * rates * theory_vals) / np.sum(weights * theory_vals**2)
    return A


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(detector: str, all_results: list[dict], out_path: Path):
    data = [r for r in all_results if r["detector"] == detector]
    if not data:
        print(f"No data for detector '{detector}'")
        return

    thetas = np.array([r["theta_eff_deg"] for r in data])
    rates  = np.array([r["rate_cps"]      for r in data])
    errs   = np.array([r["rate_err_cps"]  for r in data])

    # Sort by angle for clean lines
    order = np.argsort(thetas)
    thetas, rates, errs = thetas[order], rates[order], errs[order]

    # Theory curves over full range
    theta_fine = np.linspace(0.01, np.pi, 500)
    kn_fine   = dsigma_klein_nishina(theta_fine)
    th_fine   = dsigma_thomson(theta_fine)

    # Fit normalisation to KN and to Thomson separately
    mask = np.isfinite(errs) & (errs > 0)
    A_kn = best_fit_scale(thetas[mask], rates[mask], errs[mask], dsigma_klein_nishina)
    A_th = best_fit_scale(thetas[mask], rates[mask], errs[mask], dsigma_thomson)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        thetas, rates, yerr=errs,
        fmt="o", color="C0", markersize=5, capsize=3,
        label="Data",
    )
    ax.plot(
        np.rad2deg(theta_fine), A_kn * kn_fine,
        color="C2", lw=2, label="Klein–Nishina (normalised)",
    )
    ax.plot(
        np.rad2deg(theta_fine), A_th * th_fine,
        color="C1", lw=2, linestyle="--", label="Thomson (normalised)",
    )

    ax.set_xlabel(r"Effective scattering angle $\theta_\mathrm{eff}$ (degrees)", fontsize=12)
    ax.set_ylabel("Photopeak rate (counts/s)", fontsize=12)
    ax.set_title(f"{'Scatter' if detector == 'scatter' else 'Recoil'} detector — "
                 r"$^{137}$Cs at 661.6 keV")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 180)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── theoretical dσ/dΩ diagram ────────────────────────────────────────────────

def plot_theory_dsigma(out_path: Path):
    """
    Plot dσ/dΩ / r_e^2  vs  θ  for Thomson and Klein-Nishina side by side.
    """
    theta_fine = np.linspace(0.01, np.pi, 500)
    kn = dsigma_klein_nishina(theta_fine) / R_E_CM**2
    th = dsigma_thomson(theta_fine)       / R_E_CM**2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, vals, label, color in zip(
        axes,
        [th, kn],
        ["Thomson", "Klein–Nishina"],
        ["C1", "C2"],
    ):
        ax.plot(np.rad2deg(theta_fine), vals, color=color, lw=2.5)
        ax.set_xlabel(r"Scattering angle $\theta$ (degrees)", fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.set_xlim(0, 180)
        ax.set_ylim(bottom=0)
        ax.set_xticks([0, 45, 90, 135, 180])
    axes[0].set_ylabel(r"$(d\sigma/d\Omega)\,/\,r_e^2$", fontsize=12)
    fig.suptitle(
        r"Differential cross section at $E_0 = 661.6\,\mathrm{keV}$  ($\varepsilon = 1.295$)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # ── print theoretical totals ─────────────────────────────────────────────
    sig_T  = sigma_total_thomson()
    sig_KN = sigma_total_kn_analytic()
    sig_KN_num = sigma_total_kn_numerical()

    print(f"ε = E0/mec^2 = {EPSILON:.4f}")
    print(f"σ_Thomson         = {sig_T/BARN:.4f} barn  ({sig_T:.4e} cm^2)")
    print(f"σ_KN (analytic)   = {sig_KN/BARN:.4f} barn  ({sig_KN:.4e} cm^2)")
    print(f"σ_KN (numerical)  = {sig_KN_num/BARN:.4f} barn  ({sig_KN_num:.4e} cm^2)")
    print(f"σ_KN / σ_T        = {sig_KN/sig_T:.4f}")
    print()

    # ── load angular data and plot ────────────────────────────────────────────
    print("Loading angular SPE files …")
    results = load_angular_data()
    print(f"Loaded {len(results)} files")
    print()

    # Print summary table
    print(f"{'File':<40} {'θ_eff':>6} {'rate':>10} {'±':>8} {'χ²/ν':>10} {'p':>6}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: (x["detector"], x["theta_eff_deg"])):
        chi2_str = f"{r['chi2']:.1f}/{r['ndof']}"
        print(f"{r['filename']:<40} {r['theta_eff_deg']:>6.0f}°"
              f" {r['rate_cps']:>10.2f} {r['rate_err_cps']:>8.3f}"
              f" {chi2_str:>10} {r['p_value']:>6.3f}")
    print()

    # Save all plots to presentation folder
    out_dir = HERE / "presentation"
    out_dir.mkdir(exist_ok=True)
    plot_theory_dsigma(out_dir / "theory_dsigma.png")
    plot_comparison("scatter", results, out_dir / "angular_comparison_scatter.png")
    plot_comparison("recoil",  results, out_dir / "angular_comparison_recoil.png")


if __name__ == "__main__":
    main()
