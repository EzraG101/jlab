import math

import numpy as np
from scipy.optimize import brentq, curve_fit
from scipy.stats import chi2


def linear_model(x, slope, intercept):
    return slope * x + intercept


def weighted_linear_fit(x, y, yerr, xerr=None, max_iter=8):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    if xerr is None:
        xerr = np.zeros_like(x)
    else:
        xerr = np.asarray(xerr, dtype=float)

    sigma = np.maximum(yerr, 1e-30)
    popt = None
    pcov = None
    for _ in range(max_iter):
        popt, pcov = curve_fit(
            linear_model,
            x,
            y,
            sigma=sigma,
            absolute_sigma=True,
            maxfev=20000,
        )
        sigma_next = np.sqrt(np.maximum(yerr, 0.0) ** 2 + (popt[0] * np.maximum(xerr, 0.0)) ** 2)
        sigma_next = np.maximum(sigma_next, 1e-30)
        if np.allclose(sigma_next, sigma, rtol=1e-4, atol=0.0):
            sigma = sigma_next
            break
        sigma = sigma_next

    residuals = y - linear_model(x, *popt)
    chi2_value = float(np.sum((residuals / sigma) ** 2))
    dof = int(len(x) - len(popt))
    p_value = float(chi2.sf(chi2_value, dof)) if dof > 0 else math.nan
    weights = 1.0 / sigma**2
    y_mean_weighted = float(np.sum(weights * y) / np.sum(weights))
    ss_res = float(np.sum(weights * residuals**2))
    ss_tot = float(np.sum(weights * (y - y_mean_weighted) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return {
        "slope": float(popt[0]),
        "intercept": float(popt[1]),
        "slope_err": float(math.sqrt(pcov[0, 0])),
        "intercept_err": float(math.sqrt(pcov[1, 1])),
        "cov": pcov,
        "sigma_eff": sigma,
        "residuals": residuals,
        "standardized_residuals": residuals / sigma,
        "chi2": chi2_value,
        "dof": dof,
        "chi2_red": chi2_value / dof if dof > 0 else math.nan,
        "p_value": p_value,
        "r_squared": float(r_squared),
    }


def fit_with_extra_scatter(x, y, yerr, xerr=None):
    base = weighted_linear_fit(x, y, yerr, xerr=xerr)
    if not np.isfinite(base["chi2_red"]) or base["chi2_red"] <= 1.0:
        base["extra_y_scatter"] = 0.0
        return base

    def objective(extra):
        trial = weighted_linear_fit(x, y, np.sqrt(np.asarray(yerr) ** 2 + extra**2), xerr=xerr)
        return trial["chi2_red"] - 1.0

    high = float(np.nanmax(np.abs(base["residuals"])))
    if high <= 0:
        high = float(np.nanmax(yerr))
    high = max(high, 1e-12)
    while objective(high) > 0:
        high *= 2.0

    extra = brentq(objective, 0.0, high, xtol=1e-18, rtol=1e-10)
    adjusted = weighted_linear_fit(x, y, np.sqrt(np.asarray(yerr) ** 2 + extra**2), xerr=xerr)
    adjusted["extra_y_scatter"] = float(extra)
    return adjusted


def quadrature(*terms):
    return float(math.sqrt(sum(float(term) ** 2 for term in terms)))
