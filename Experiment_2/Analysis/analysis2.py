# johnson_noise_analysis.py

import os
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------
# Constants / defaults
# -------------------------
KB_TRUE = 1.380649e-23  # CODATA exact [J/K]
ROOM_TEMPERATURE = 293.0  # K

DATA_DIR = ".\\data"
PLOT_DIR = ".\\plots"

# -------------------------
# I/O helper (kept in your style)
# -------------------------
def get_data(filename: str) -> dict:
    """
    Reads CSV and returns {column_name: np.array}.
    """
    df = pd.read_csv(filename)
    column_names = list(df.columns)
    values = df.to_numpy()

    data_dict = {}
    for i, key in enumerate(column_names):
        data_dict[key] = values[:, i]
    return data_dict


# -------------------------
# Statistics helpers
# -------------------------
def repeats_to_mean_sem(values: np.ndarray, n_rep: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert block repeats -> mean and SEM (std/sqrt(n), with ddof=1).
    Assumes repeats are stored contiguously in blocks of length n_rep.
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size % n_rep != 0:
        raise ValueError(
            f"Length {arr.size} not divisible by n_rep={n_rep}. "
            "Check file ordering/repetition count."
        )

    blocks = arr.reshape(-1, n_rep)
    means = np.mean(blocks, axis=1)

    if n_rep > 1:
        sem = np.std(blocks, axis=1, ddof=1) / np.sqrt(n_rep)
    else:
        sem = np.zeros_like(means)

    return means, sem


def mean_sem(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float).ravel()
    m = float(np.mean(arr))
    if arr.size > 1:
        s = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    else:
        s = 0.0
    return m, s


def _trapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    # numpy version compatibility
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapezoid(y, x=x, axis=axis)


# -------------------------
# Physics model
# -------------------------
def G_of_R(f_hz: np.ndarray, g2: np.ndarray, C_F: float, R_ohm: np.ndarray) -> np.ndarray:
    """
    G(R) = ∫ g2(f) / [1 + (2π f C R)^2] df
    Vectorized over R.
    """
    f = np.asarray(f_hz, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    R = np.asarray(R_ohm, dtype=float)

    denom = 1.0 + (2 * np.pi * f[:, None] * C_F * R[None, :]) ** 2
    integrand = g2[:, None] / denom
    return _trapz(integrand, x=f, axis=0)


def build_model(
    f_hz: np.ndarray,
    g2: np.ndarray,
    T: float,
    fit_C: bool,
    fit_V0: bool,
    C_fixed_F: Optional[float],
):
    """
    Returns (model_function, p0, bounds, param_names)
    """
    if fit_C:
        if fit_V0:
            # params: k, C, V0
            def model(R_ohm, k, C_F, V0_V2):
                return 4.0 * k * T * R_ohm * G_of_R(f_hz, g2, C_F, R_ohm) + V0_V2

            p0 = [1.3e-23, 50e-12, 0.0]
            bounds = ([0.0, 1e-12, 0.0], [1e-21, 2e-9, np.inf])  # C from 1 pF to 2 nF
            names = ["k", "C", "V0"]

        else:
            # params: k, C
            def model(R_ohm, k, C_F):
                return 4.0 * k * T * R_ohm * G_of_R(f_hz, g2, C_F, R_ohm)

            p0 = [1.3e-23, 50e-12]
            bounds = ([0.0, 1e-12], [1e-21, 2e-9])
            names = ["k", "C"]

    else:
        if C_fixed_F is None:
            raise ValueError("C_fixed_F must be provided if fit_C=False")

        if fit_V0:
            # params: k, V0
            def model(R_ohm, k, V0_V2):
                return 4.0 * k * T * R_ohm * G_of_R(f_hz, g2, C_fixed_F, R_ohm) + V0_V2

            p0 = [1.3e-23, 0.0]
            bounds = ([0.0, 0.0], [1e-21, np.inf])
            names = ["k", "V0"]

        else:
            # params: k
            def model(R_ohm, k):
                return 4.0 * k * T * R_ohm * G_of_R(f_hz, g2, C_fixed_F, R_ohm)

            p0 = [1.3e-23]
            bounds = ([0.0], [1e-21])
            names = ["k"]

    return model, p0, bounds, names


def fit_once(
    R_ohm: np.ndarray,
    V2_V2: np.ndarray,
    V2err_V2: np.ndarray,
    f_hz: np.ndarray,
    g2: np.ndarray,
    T: float,
    fit_C: bool,
    fit_V0: bool,
    C_fixed_F: Optional[float],
):
    # sort frequency (critical for integration)
    idx = np.argsort(f_hz)
    f_sorted = np.asarray(f_hz)[idx]
    g2_sorted = np.asarray(g2)[idx]

    model, p0, bounds, names = build_model(
        f_sorted, g2_sorted, T, fit_C=fit_C, fit_V0=fit_V0, C_fixed_F=C_fixed_F
    )

    sigma = np.maximum(np.asarray(V2err_V2, dtype=float), 1e-30)

    popt, pcov = curve_fit(
        model,
        np.asarray(R_ohm, dtype=float),
        np.asarray(V2_V2, dtype=float),
        p0=p0,
        bounds=bounds,
        sigma=sigma,
        absolute_sigma=True,
        maxfev=50000,
    )

    yfit = model(R_ohm, *popt)
    chi2 = float(np.sum(((V2_V2 - yfit) / sigma) ** 2))
    dof = int(len(V2_V2) - len(popt))
    chi2_red = chi2 / dof if dof > 0 else np.nan

    perr = np.sqrt(np.diag(pcov))
    return {
        "param_names": names,
        "popt": popt,
        "perr_stat": perr,
        "pcov": pcov,
        "model": model,
        "f_sorted": f_sorted,
        "g2_sorted": g2_sorted,
        "yfit": yfit,
        "chi2": chi2,
        "dof": dof,
        "chi2_red": chi2_red,
    }


def monte_carlo_uncertainty(
    R_ohm: np.ndarray,
    Rerr_ohm: np.ndarray,
    V2_V2: np.ndarray,
    V2err_V2: np.ndarray,
    f_hz: np.ndarray,
    ferr_hz: np.ndarray,
    g2: np.ndarray,
    g2err: np.ndarray,
    T: float,
    fit_C: bool,
    fit_V0: bool,
    C_fixed_F: Optional[float],
    C_fixed_err_F: float = 0.0,
    n_mc: int = 1000,
    seed: Optional[int] = 0,
):
    rng = np.random.default_rng(seed)
    samples = []
    fail = 0

    for _ in range(n_mc):
        # draw noisy realization
        R_s = rng.normal(R_ohm, Rerr_ohm)
        V2_s = rng.normal(V2_V2, V2err_V2)
        f_s = rng.normal(f_hz, ferr_hz)
        g2_s = rng.normal(g2, g2err)

        # physical clipping
        R_s = np.clip(R_s, 1e-3, None)
        V2_s = np.clip(V2_s, 0.0, None)
        f_s = np.clip(f_s, 1.0, None)
        g2_s = np.clip(g2_s, 1e-12, None)

        C_use = C_fixed_F
        if (not fit_C) and (C_fixed_F is not None) and (C_fixed_err_F > 0):
            C_use = float(np.clip(rng.normal(C_fixed_F, C_fixed_err_F), 1e-13, None))

        try:
            fit = fit_once(
                R_s, V2_s, np.maximum(V2err_V2, 1e-30),
                f_s, g2_s, T,
                fit_C=fit_C, fit_V0=fit_V0, C_fixed_F=C_use
            )
            samples.append(fit["popt"])
        except Exception:
            fail += 1

    samples = np.array(samples)
    if len(samples) < 5:
        raise RuntimeError(f"Too few successful MC fits ({len(samples)}). Failures={fail}")

    median = np.median(samples, axis=0)
    lo = np.percentile(samples, 16, axis=0)
    hi = np.percentile(samples, 84, axis=0)
    mc_sigma = 0.5 * (hi - lo)

    return {
        "samples": samples,
        "median": median,
        "lo16": lo,
        "hi84": hi,
        "sigma_mc": mc_sigma,
        "n_success": len(samples),
        "n_fail": fail,
    }


# -------------------------
# Plotting
# -------------------------
def plot_v2_fit(
    R_kohm: np.ndarray,
    Rerr_kohm: np.ndarray,
    V2_mV2: np.ndarray,
    V2err_mV2: np.ndarray,
    Rgrid_kohm: np.ndarray,
    V2fit_mV2: np.ndarray,
    outpath: str,
):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure()
    plt.errorbar(R_kohm, V2_mV2, xerr=Rerr_kohm, yerr=V2err_mV2, fmt="o", label="Data")
    plt.plot(Rgrid_kohm, V2fit_mV2, "-", label="Best fit")
    plt.xlabel("Resistance [kΩ]")
    plt.ylabel("Measured noise power $V^2$ [mV$^2$]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_k_by_R(
    R_kohm: np.ndarray,
    k_by_R: np.ndarray,
    kerr_by_R: np.ndarray,
    outpath: str,
):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure()
    plt.errorbar(R_kohm, k_by_R, yerr=kerr_by_R, fmt="o", label="Pointwise $k(R)$")
    plt.axhline(KB_TRUE, color="C1", label="CODATA $k_B$")
    plt.xlabel("Resistance [kΩ]")
    plt.ylabel("k [J/K]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# -------------------------
# End-to-end pipeline
# -------------------------
def run_johnson_analysis(
    calibration_csv: str,
    data_csv: str,
    nrep_cal: int = 3,
    nrep_data: int = 5,
    C_measurements_pF: Optional[np.ndarray] = None,
    T: float = ROOM_TEMPERATURE,
    fit_C: bool = False,      # Recommended: False first
    fit_V0: bool = True,
    n_mc: int = 1000,
    seed: int = 0,
    plot_dir: str = PLOT_DIR,
):
    # 1) read
    cal = get_data(calibration_csv)
    dat = get_data(data_csv)

    # 2) reduce repeats
    f_khz, ferr_khz = repeats_to_mean_sem(cal["f"], nrep_cal)
    g2, g2err = repeats_to_mean_sem(cal["gain^2"], nrep_cal)

    R_kohm, Rerr_kohm = repeats_to_mean_sem(dat["R"], nrep_data)
    V2_mV2, V2err_mV2 = repeats_to_mean_sem(dat["V^2"], nrep_data)

    # 3) capacitance
    if C_measurements_pF is None:
        C_measurements_pF = np.array([44.06, 45.72, 46.97, 46.80, 48.33, 50.60], dtype=float)

    C_mean_pF, C_sem_pF = mean_sem(C_measurements_pF)
    C_mean_F = C_mean_pF * 1e-12
    C_sem_F = C_sem_pF * 1e-12

    # 4) to SI
    f_hz = f_khz * 1e3
    ferr_hz = ferr_khz * 1e3
    R_ohm = R_kohm * 1e3
    Rerr_ohm = Rerr_kohm * 1e3
    V2_V2 = V2_mV2 * 1e-6
    V2err_V2 = V2err_mV2 * 1e-6

    # 5) nominal fit
    fit_nom = fit_once(
        R_ohm, V2_V2, V2err_V2,
        f_hz, g2, T,
        fit_C=fit_C, fit_V0=fit_V0,
        C_fixed_F=None if fit_C else C_mean_F
    )

    # 6) MC uncertainty
    mc = monte_carlo_uncertainty(
        R_ohm, Rerr_ohm, V2_V2, V2err_V2,
        f_hz, ferr_hz, g2, g2err,
        T=T,
        fit_C=fit_C, fit_V0=fit_V0,
        C_fixed_F=None if fit_C else C_mean_F,
        C_fixed_err_F=0.0 if fit_C else C_sem_F,
        n_mc=n_mc, seed=seed
    )

    names = fit_nom["param_names"]
    popt = fit_nom["popt"]
    perr_stat = fit_nom["perr_stat"]
    perr_mc = mc["sigma_mc"]
    perr_total = np.sqrt(perr_stat**2 + perr_mc**2)

    # pack readable summary
    out = {
        "param_names": names,
        "popt_SI": popt,
        "perr_stat_SI": perr_stat,
        "perr_mc_SI": perr_mc,
        "perr_total_SI": perr_total,
        "chi2_red": fit_nom["chi2_red"],
        "n_mc_success": mc["n_success"],
    }

    # named parameters
    p = {n: popt[i] for i, n in enumerate(names)}
    e = {n: perr_total[i] for i, n in enumerate(names)}

    k_val = p["k"]
    k_err = e["k"]

    C_val_pF = (p["C"] * 1e12) if ("C" in p) else C_mean_pF
    C_err_pF = (e["C"] * 1e12) if ("C" in p) else C_sem_pF

    V0_val_mV2 = (p["V0"] * 1e6) if ("V0" in p) else 0.0
    V0_err_mV2 = (e["V0"] * 1e6) if ("V0" in p) else 0.0

    out["fit_readable"] = {
        "k_J_per_K": (k_val, k_err),
        "Ceff_pF": (C_val_pF, C_err_pF),
        "V0_mV2": (V0_val_mV2, V0_err_mV2),
        "chi2_red": fit_nom["chi2_red"],
        "n_mc_success": mc["n_success"],
    }

    # 7) plots
    os.makedirs(plot_dir, exist_ok=True)

    # fit curve plot
    Rgrid_kohm = np.linspace(min(R_kohm), max(R_kohm), 300)
    Rgrid_ohm = Rgrid_kohm * 1e3

    # rebuild model with nominal calibration
    idx = np.argsort(f_hz)
    f_sorted = f_hz[idx]
    g2_sorted = g2[idx]

    if "C" in p:
        C_use = p["C"]
    else:
        C_use = C_mean_F
    if "V0" in p:
        V0_use = p["V0"]
    else:
        V0_use = 0.0

    V2fit_grid = 4 * p["k"] * T * Rgrid_ohm * G_of_R(f_sorted, g2_sorted, C_use, Rgrid_ohm) + V0_use
    plot_v2_fit(
        R_kohm, Rerr_kohm, V2_mV2, V2err_mV2,
        Rgrid_kohm, V2fit_grid * 1e6,
        os.path.join(plot_dir, "johnson_V2_fit.png")
    )

    # pointwise k(R) diagnostic
    G_data = G_of_R(f_sorted, g2_sorted, C_use, R_ohm)
    k_by_R = (V2_V2 - V0_use) / (4 * T * R_ohm * G_data)
    kerr_by_R = V2err_V2 / (4 * T * R_ohm * G_data)
    plot_k_by_R(
        R_kohm, k_by_R, kerr_by_R,
        os.path.join(plot_dir, "johnson_k_by_R.png")
    )

    return out


if __name__ == "__main__":
    calibration_file = os.path.join(DATA_DIR, "calibration-3_31.csv")
    data_file = os.path.join(DATA_DIR, "data-3_31-fix.csv")

    result = run_johnson_analysis(
        calibration_csv=calibration_file,
        data_csv=data_file,
        nrep_cal=3,
        nrep_data=5,
        C_measurements_pF=np.array([44.06, 45.72, 46.97, 46.80, 48.33, 50.60]),
        T=ROOM_TEMPERATURE,
        fit_C=True,    # start with fixed measured C (more stable)
        fit_V0=True,
        n_mc=1000,
        seed=42,
        plot_dir=PLOT_DIR,
    )

    fr = result["fit_readable"]
    k, dk = fr["k_J_per_K"]
    C, dC = fr["Ceff_pF"]
    V0, dV0 = fr["V0_mV2"]

    print("=== Johnson Noise Fit Summary ===")
    print(f"k      = {k:.6e} ± {dk:.2e} J/K")
    print(f"C_eff  = {C:.3f} ± {dC:.3f} pF")
    print(f"V0^2   = {V0:.4f} ± {dV0:.4f} mV^2")
    print(f"chi2_r = {fr['chi2_red']:.3f}")
    print(f"MC ok  = {fr['n_mc_success']}")
    print(f"bias vs CODATA = {(k / KB_TRUE - 1) * 100:.1f}%")