import math

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from scipy.stats import chi2


def power_law(f_hz, amplitude, exponent):
    return amplitude * np.power(f_hz, -exponent)


def _linear_zero_tail(f_hz, gain2, n_terminal=2):
    x = np.asarray(f_hz[-n_terminal:], dtype=float)
    y = np.asarray(gain2[-n_terminal:], dtype=float)
    if n_terminal == 2:
        slope = (y[-1] - y[0]) / (x[-1] - x[0])
        intercept = y[-1] - slope * x[-1]
    else:
        slope, intercept = np.polyfit(x, y, deg=1)

    if slope >= 0:
        f_zero = float(x[-1])
        extra = 0.0
    else:
        f_zero = float(-intercept / slope)
        extra = 0.0 if f_zero <= x[-1] else float(0.5 * y[-1] * (f_zero - x[-1]))

    return {
        "model": "linear_zero",
        "n_terminal": int(n_terminal),
        "slope_gain2_per_hz": float(slope),
        "intercept_gain2": float(intercept),
        "f_zero_hz": f_zero,
        "integral": extra,
    }


def fit_right_tail(f_hz, gain2, gain2_err, tail_start_hz):
    f_hz = np.asarray(f_hz, dtype=float)
    gain2 = np.asarray(gain2, dtype=float)
    gain2_err = np.asarray(gain2_err, dtype=float)

    mask = f_hz >= tail_start_hz
    if np.count_nonzero(mask) < 3:
        raise ValueError("Need at least three points for the right-tail fit")

    x = f_hz[mask]
    y = gain2[mask]
    sigma = np.maximum(gain2_err[mask], np.maximum(0.01 * y, 1.0))

    log_slope = -np.polyfit(np.log(x), np.log(y), 1)[0]
    p0 = [float(y[0] * x[0] ** log_slope), max(1.1, float(log_slope))]
    popt, pcov = curve_fit(
        power_law,
        x,
        y,
        sigma=sigma,
        absolute_sigma=True,
        p0=p0,
        bounds=([0.0, 1.000001], [np.inf, 25.0]),
        maxfev=50000,
    )

    residuals = y - power_law(x, *popt)
    chi2_value = float(np.sum((residuals / sigma) ** 2))
    dof = int(len(x) - len(popt))
    return {
        "amplitude": float(popt[0]),
        "exponent": float(popt[1]),
        "amplitude_err": float(math.sqrt(pcov[0, 0])),
        "exponent_err": float(math.sqrt(pcov[1, 1])),
        "cov": pcov,
        "mask": mask,
        "sigma": sigma,
        "residuals": residuals,
        "chi2": chi2_value,
        "dof": dof,
        "chi2_red": chi2_value / dof if dof > 0 else math.nan,
        "p_value": float(chi2.sf(chi2_value, dof)) if dof > 0 else math.nan,
    }


def integrate_gain(f_hz, gain2, gain2_err, tail_start_hz, n_mc=20000, seed=20260407, tail_mode="power_law"):
    f_hz = np.asarray(f_hz, dtype=float)
    gain2 = np.asarray(gain2, dtype=float)
    gain2_err = np.asarray(gain2_err, dtype=float)
    order = np.argsort(f_hz)
    f_hz = f_hz[order]
    gain2 = gain2[order]
    gain2_err = gain2_err[order]

    tail = fit_right_tail(f_hz, gain2, gain2_err, tail_start_hz)
    if not np.any(np.isclose(f_hz, tail_start_hz)):
        raise ValueError("tail_start_hz must be one of the measured calibration frequencies")

    trapezoid = float(np.trapezoid(gain2, x=f_hz))
    simpson_integral = float(simpson(gain2, x=f_hz))
    discretization_err = abs(simpson_integral - trapezoid)
    tail_lower_hz = float(np.max(f_hz))
    power_law_tail_integral = float(tail["amplitude"] * tail_lower_hz ** (1.0 - tail["exponent"]) / (tail["exponent"] - 1.0))
    if tail_mode == "power_law":
        tail_integral = power_law_tail_integral
        nominal_tail = {"model": "power_law", "f_zero_hz": math.inf, "integral": power_law_tail_integral}
    elif tail_mode == "linear_zero":
        linear_tail = _linear_zero_tail(f_hz, gain2, n_terminal=2)
        tail_integral = linear_tail["integral"]
        nominal_tail = linear_tail
    else:
        raise ValueError(f"Unknown tail_mode: {tail_mode}")

    tail_model_err = 0.0
    total = trapezoid + tail_integral

    rng = np.random.default_rng(seed)
    samples = []
    if n_mc > 0:
        tail_mean = np.asarray([tail["amplitude"], tail["exponent"]], dtype=float)
        for _ in range(n_mc):
            sample_gain = rng.normal(gain2, np.maximum(gain2_err, 0.0))
            sample_gain = np.clip(sample_gain, 0.0, None)
            sample_trap = float(np.trapezoid(sample_gain, x=f_hz))
            if tail_mode == "power_law":
                sample_amp, sample_exp = rng.multivariate_normal(tail_mean, tail["cov"])
                if sample_amp <= 0.0 or sample_exp <= 1.0:
                    continue
                sample_tail_integral = float(
                    sample_amp * tail_lower_hz ** (1.0 - sample_exp) / (sample_exp - 1.0)
                )
            else:
                sample_tail_integral = _linear_zero_tail(f_hz, sample_gain, n_terminal=2)["integral"]
            samples.append(sample_trap + sample_tail_integral)

    samples = np.asarray(samples, dtype=float)
    mc_err = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
    measured_only = float(np.trapezoid(gain2, x=f_hz))
    total_err = float(math.sqrt(mc_err**2 + discretization_err**2 + tail_model_err**2))

    return {
        "G": total,
        "G_err_stat": mc_err,
        "G_err_discretization": discretization_err,
        "G_err_tail_model": tail_model_err,
        "G_err_total": total_err,
        "G_measured_only": measured_only,
        "G_trapezoid_measured": trapezoid,
        "G_simpson_measured": simpson_integral,
        "G_tail_lower_hz": tail_lower_hz,
        "G_tail": tail_integral,
        "G_tail_power_law": power_law_tail_integral,
        "nominal_tail": nominal_tail,
        "tail": tail,
        "mc_samples_used": int(len(samples)),
    }


def gain_integral_for_tail_start(f_hz, gain2, gain2_err, tail_start_hz):
    return integrate_gain(f_hz, gain2, gain2_err, tail_start_hz, n_mc=0)
