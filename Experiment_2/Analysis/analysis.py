import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Global Variables

DATA_DIR = ".\\data"
PLOT_DIR = ".\\plots"

ROOM_TEMPERATURE = 293 # (K)

# Colors

# Helper Functions

def get_data(filename:str) -> dict:
    """
    Reads the csv file at filename and outputs its content as a dictionary
    whose keys represent headers and values are np.arrays of data 
    """
    # Load the data
    df = pd.read_csv(filename)
    column_names = list(df.columns)
    values = df.to_numpy()
    
    # Create the dictionary
    data_dict = {}
    index = 0
    for key in column_names:
        data_dict[key] = values[:, index]
        index += 1
    
    return data_dict

def measurements_to_mean_with_err(measurements:np.ndarray, N:int|None=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes an np.array of measurements and the number of repetition per
    measurement, N, and outputs an np.array (of length len(measurements)/N) 
    of the means of each measurement and an np.array of the standard errors
    of each measurement
    """
    # Default Behavior
    if N is None:
        N = len(measurements)

    # Check N and measurements consistency
    L = len(measurements)
    if len(measurements) % N != 0:
        raise ValueError("Length of Measurements array is not divisible by number of repetitions.")
    l = L // N

    # Create the arrays
    means = []
    errs = []
    for i in range(l):
        trials = measurements[N * i:N * (i + 1)]
        means.append(np.mean(trials))
        errs.append(np.std(trials, ddof=1) / np.sqrt(N))
    
    return np.array(means), np.array(errs)

def plot_y_vs_x(
        x:np.ndarray, 
        xerr:np.ndarray, 
        y:np.ndarray, 
        yerr:np.ndarray, 
        xlabel:str, 
        ylabel:str, 
        output_dir:str, 
        filename:str,
        close:bool=True,
        ) -> None:
    """
    Makes a plot of y versus x with errors yerr and xerr, and saves it.
    """
    ### THINK ABOUT UPDATE FOR 2D y
    # Required directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Make scatter plot with error bars
    fig, ax = plt.subplots()
    ax.errorbar(
        x=x, 
        y=y, 
        xerr=xerr, 
        yerr=yerr,
        fmt="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if close:
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close('all')
    else:
        return fig, ax

def numerical_integral(
        x:np.ndarray, 
        xerr:np.ndarray, 
        y:np.ndarray, 
        yerr:np.ndarray
        ) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Takes in np.arrays of x and y values, along with their errors and computes the
    corresponding integral and the error on it
    """
    # Check consistency
    L = len(x)
    if np.size(x) != np.size(y, axis=0):
        raise ValueError("x and y input arrays should have compatible shapes.")
    elif np.shape(x) != np.shape(xerr):
        raise ValueError("x and xerr input arrays should have compatible shapes.")
    elif np.shape(y) != np.shape(yerr):
        raise ValueError("y and yerr input arrays should have compatible shapes.")
    
    # Sort arrays
    idx = np.argsort(x)
    x, xerr = x[idx], xerr[idx]
    y, yerr = y[idx], yerr[idx]
    
    # Compute integral with trapezoidal rule, keeping track of errors
    integral = 0
    err2 = 0
    
    for i in range(L-1):
        integral += (x[i+1] - x[i]) * (y[i+1] + y[i]) / 2 # trapezoid rule

        if i == 0: # left edge
            err2 += ((y[1] + y[0]) * xerr[0] / 2) ** 2 # x err contribution
            err2 += ((x[1] - x[0]) * yerr[0] / 2) ** 2 # y err contribution
        else: # middle
            err2 += ((y[i-1] - y[i+1]) * xerr[i] / 2) ** 2 # x err contribution
            err2 += ((x[i+1] - x[i-1]) * yerr[i] / 2) ** 2 # y err contribution
    
    # Right edge
    err2 += ((y[L-1] + y[L-2]) * xerr[L-1] / 2) ** 2 # x err contribution
    err2 += ((x[L-1] - x[L-2]) * yerr[L-1] / 2) ** 2 # y err contribution

    # Convert error squared to just error
    err = np.sqrt(err2)

    return integral, err

def compute_k(
        f:np.ndarray, 
        ferr:np.ndarray, 
        g2:np.ndarray, 
        g2err:np.ndarray, 
        V2:np.ndarray, 
        V2err:np.ndarray, 
        R:np.ndarray, 
        Rerr:np.ndarray, 
        C:float=0, 
        Cerr:float=0, 
        T:float=ROOM_TEMPERATURE
        ) -> tuple[float, float]:
    """
    From measured frequencies, gain^2, V^2, R, C, and T, computes Boltzmann constant
    according to Johnson noise theory and keeps track of errors.
    """
    # Convert units
    f = f * 10 ** 3 # kHz -> Hz
    R = R * 10 ** 3 # kOhm -> Ohm
    V2 = V2 * 10 ** -6 # mV^2 -> V^2
    C = C * 10 ** -12 # pF -> F
    ferr = ferr * 10 ** 3 # kHz -> Hz
    Rerr = Rerr * 10 ** 3 # kOhm -> Ohm
    V2err = V2err * 10 ** -6 # mV^2 -> V^2
    Cerr = Cerr * 10 ** -12 # pF -> F

    # Compute G
    n = len(f)
    m = len(R)
    y = []
    yerr2 = []
    for i in range(n):
        row_y = []
        row_err = []
        for j in range(m):
            row_y.append(g2[i] / (1 + (2 * np.pi * f[i] * C * R[j]) ** 2))

            err2 = (g2err[i] / (1 + (2 * np.pi * f[i] * C * R[j]) ** 2)) ** 2 # g2 err contribution

            err2 += (g2[i] / (1 + (2 * np.pi * f[i] * C * R[j]) ** 2) ** 2 * 2 * f[i] * (2 * np.pi * C * R[j]) ** 2 * ferr[i]) ** 2 # f err contribution
            err2 += (g2[i] / (1 + (2 * np.pi * f[i] * C * R[j]) ** 2) ** 2 * 2 * C * (2 * np.pi * f[i] * R[j]) ** 2 * Cerr) ** 2 # C err contribution
            err2 += (g2[i] / (1 + (2 * np.pi * f[i] * C * R[j]) ** 2) ** 2 * 2 * R[j] * (2 * np.pi * C * f[i]) ** 2 * Rerr[j]) ** 2 # R err contribution
            row_err.append(err2)
        y.append(row_y)
        yerr2.append(row_err)
    y = np.array(y)
    yerr2 = np.array(yerr2)
    yerr = np.sqrt(yerr2)

    Gval, Gerr = numerical_integral(f, ferr, y, yerr)

    # Compute k
    k = V2 / (4 * R * T * Gval)
    kerr2 = k * 0
    kerr2 += (V2err / (4 * R * T * Gval)) ** 2 # V2 err contribution
    kerr2 += (V2 / (4 * R ** 2 * T * Gval) * Rerr) ** 2 # R err contribution
    kerr2 += (V2 / (4 * R * T * Gval ** 2) * Gerr) ** 2 # G err contribution
    kerr = np.sqrt(kerr2)

    ### DOUBLE CHECK ALL CALCULATIONS

    return k, kerr

def _G_of_R(f_hz: np.ndarray, g2: np.ndarray, R_ohm: np.ndarray, C_F: float) -> np.ndarray:
    """
    G(R) = ∫ g2(f) / (1 + (2π f C R)^2) df
    Vectorized over R.
    """
    y = g2[:, None] / (1.0 + (2*np.pi*f_hz[:, None]*C_F*R_ohm[None, :])**2)
    return np.trapezoid(y, x=f_hz, axis=0)

def fit_johnson_full_model(
    f_khz: np.ndarray, ferr_khz: np.ndarray,
    g2: np.ndarray, g2err: np.ndarray,
    R_kohm: np.ndarray, Rerr_kohm: np.ndarray,
    V2_mV2: np.ndarray, V2err_mV2: np.ndarray,
    T: float = 293.0,
    fit_C: bool = True,
    fit_V0: bool = True,
    C_fixed_pF: float | None = None,
    V0_fixed_mV2: float = 0.0,
    n_mc: int = 500,
    seed: int | None = 0,
):
    """
    Fits full model and propagates errors.

    Inputs use your current units:
      f [kHz], R [kΩ], V2 [mV^2], C [pF].
    Returns best-fit params in SI + convenient lab units.
    """

    # --- unit conversion to SI ---
    f = np.asarray(f_khz, dtype=float) * 1e3
    ferr = np.asarray(ferr_khz, dtype=float) * 1e3
    g2 = np.asarray(g2, dtype=float)
    g2err = np.asarray(g2err, dtype=float)
    R = np.asarray(R_kohm, dtype=float) * 1e3
    Rerr = np.asarray(Rerr_kohm, dtype=float) * 1e3
    V2 = np.asarray(V2_mV2, dtype=float) * 1e-6
    V2err = np.asarray(V2err_mV2, dtype=float) * 1e-6

    # sort frequency arrays before integration
    idx = np.argsort(f)
    f, ferr, g2, g2err = f[idx], ferr[idx], g2[idx], g2err[idx]

    if (not fit_C) and (C_fixed_pF is None):
        raise ValueError("If fit_C=False, provide C_fixed_pF.")
    C_fixed = None if C_fixed_pF is None else C_fixed_pF * 1e-12
    V0_fixed = V0_fixed_mV2 * 1e-6

    # --- model factory for current (f, g2) calibration ---
    def build_model(f_use, g2_use):
        if fit_C and fit_V0:
            names = ["k", "C", "V0"]
            def model(R_ohm, k, C, V0):
                return 4*k*T*R_ohm*_G_of_R(f_use, g2_use, R_ohm, C) + V0
            p0 = [1.38e-23, 45e-12, max(0.0, 0.1*np.min(V2))]
            bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf])

        elif fit_C and (not fit_V0):
            names = ["k", "C"]
            def model(R_ohm, k, C):
                return 4*k*T*R_ohm*_G_of_R(f_use, g2_use, R_ohm, C) + V0_fixed
            p0 = [1.38e-23, 45e-12]
            bounds = ([0.0, 0.0], [np.inf, np.inf])

        elif (not fit_C) and fit_V0:
            names = ["k", "V0"]
            def model(R_ohm, k, V0):
                return 4*k*T*R_ohm*_G_of_R(f_use, g2_use, R_ohm, C_fixed) + V0
            p0 = [1.38e-23, max(0.0, 0.1*np.min(V2))]
            bounds = ([0.0, 0.0], [np.inf, np.inf])

        else:  # fit neither C nor V0
            names = ["k"]
            def model(R_ohm, k):
                return 4*k*T*R_ohm*_G_of_R(f_use, g2_use, R_ohm, C_fixed) + V0_fixed
            p0 = [1.38e-23]
            bounds = ([0.0], [np.inf])

        return model, names, p0, bounds

    def fit_once(model, p0, bounds):
        # first fit with V2err only
        sigma = np.maximum(V2err, 1e-30)
        popt, pcov = curve_fit(
            model, R, V2, p0=p0, sigma=sigma, absolute_sigma=True,
            bounds=bounds, maxfev=20000
        )

        # update sigma to include R uncertainty: sigma_eff^2 = sigma_V2^2 + (dV2/dR * sigma_R)^2
        dR = np.maximum(1e-9*np.maximum(R, 1.0), 1e-3*Rerr + 1e-12)
        y_plus = model(R + dR, *popt)
        y_minus = model(np.maximum(R - dR, 1e-15), *popt)
        dVdR = (y_plus - y_minus) / (2*dR)
        sigma_eff = np.sqrt(np.maximum(V2err, 0)**2 + (dVdR * np.maximum(Rerr, 0))**2)
        sigma_eff = np.maximum(sigma_eff, 1e-30)

        # second fit with sigma_eff
        popt, pcov = curve_fit(
            model, R, V2, p0=popt, sigma=sigma_eff, absolute_sigma=True,
            bounds=bounds, maxfev=20000
        )

        resid = V2 - model(R, *popt)
        chi2 = np.sum((resid / sigma_eff)**2)
        dof = len(V2) - len(popt)
        return popt, pcov, chi2, dof, sigma_eff

    # nominal fit
    model_nom, names, p0, bounds = build_model(f, g2)
    popt, pcov, chi2, dof, sigma_eff = fit_once(model_nom, p0, bounds)
    stat_err = np.sqrt(np.diag(pcov))

    # Monte Carlo propagation for calibration (f, g2) uncertainties
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_mc):
        f_s = rng.normal(f, ferr)
        g2_s = rng.normal(g2, g2err)

        # keep physical
        f_s = np.clip(f_s, 1e-12, None)
        g2_s = np.clip(g2_s, 1e-12, None)

        # sort by frequency
        sidx = np.argsort(f_s)
        f_s, g2_s = f_s[sidx], g2_s[sidx]

        model_s, _, _, _ = build_model(f_s, g2_s)
        try:
            p_s, _, _, _, _ = fit_once(model_s, popt, bounds)
            samples.append(p_s)
        except Exception:
            pass

    samples = np.array(samples) if len(samples) else np.empty((0, len(popt)))
    if len(samples) >= 2:
        mc_err = np.std(samples, axis=0, ddof=1)
    else:
        mc_err = np.full_like(popt, np.nan)

    total_err = np.sqrt(stat_err**2 + np.nan_to_num(mc_err, nan=0.0)**2)

    # pack results
    out = {
        "param_names": names,
        "popt_SI": popt,
        "stat_err_SI": stat_err,
        "mc_err_SI": mc_err,
        "total_err_SI": total_err,
        "pcov_stat_SI": pcov,
        "chi2": chi2,
        "dof": dof,
        "chi2_red": chi2 / dof if dof > 0 else np.nan,
        "sigma_eff_V2_SI": sigma_eff,
        "mc_samples_SI": samples,
    }

    # convenience lab units
    named = dict(zip(names, popt))
    named_err = dict(zip(names, total_err))
    out["fit_readable"] = {
        "k_J_per_K": (named.get("k", np.nan), named_err.get("k", np.nan)),
        "Ceff_pF": (
            (named["C"] * 1e12) if "C" in named else (C_fixed * 1e12 if C_fixed is not None else np.nan),
            (named_err.get("C", np.nan) * 1e12) if "C" in named else 0.0
        ),
        "V0_mV2": (
            (named["V0"] * 1e6) if "V0" in named else (V0_fixed * 1e6),
            (named_err.get("V0", np.nan) * 1e6) if "V0" in named else 0.0
        ),
        "chi2_red": out["chi2_red"],
        "n_mc_ok": len(samples),
    }
    return out

if __name__ == "__main__":
    # Testing
    calibration = get_data(DATA_DIR + "\\calibration-3_31.csv")
    freqs = measurements_to_mean_with_err(calibration["f"], 3)
    gain2s = measurements_to_mean_with_err(calibration["gain^2"], 3)
    data = get_data(DATA_DIR + "\\data-3_31.csv")
    Rs = measurements_to_mean_with_err(data["R"], 5)
    V2s = measurements_to_mean_with_err(data["V^2"], 5)

    Cval, Cerr = measurements_to_mean_with_err(np.array([44.06, 45.72, 46.97, 46.80, 48.33, 50.60]))
    Rs = Rs[0], Rs[1] + Rs[0] * 0.01

    # plot_y_vs_x(*freqs, *gain2s, output_dir=PLOT_DIR, filename="test.png", xlabel="Frequency [kHz]", ylabel="Gain^2")
    # plot_y_vs_x(*Rs, *V2s, output_dir=PLOT_DIR, filename="test.png", xlabel="Resistance [kOhm]", ylabel="V^2 [mV^2]")
    # print(numerical_integral(*freqs, *gain2s))
    # k = compute_k(*freqs, *gain2s, *V2s, *Rs, Cval[0], Cerr[0])
    # fig, ax = plot_y_vs_x(*Rs, *k, output_dir=PLOT_DIR, filename="test.png", xlabel="Resistance [kOhm]", ylabel="k [J/K]", close=False)
    # ax.hlines(1.38*10**-23, xmin=min(Rs[0]), xmax=max(Rs[0]))
    # plt.tight_layout()
    # plt.savefig(os.path.join(PLOT_DIR, "test.png"), dpi=300)
    # plt.close('all')

    fit = fit_johnson_full_model(
        freqs[0], freqs[1],
        gain2s[0], gain2s[1],
        Rs[0], Rs[1],
        V2s[0], V2s[1],
        T=ROOM_TEMPERATURE,
        fit_C=True,
        fit_V0=False,
        n_mc=800,
        seed=42,
        V0_fixed_mV2=0
    )

    print(fit["fit_readable"])
        