import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

# Colorblind-friendly palette
CBLUE = "#0072B2"
CORANGE = "#E69F00"
CGREEN = "#009E73"
CRED = "#D55E00"
CPURPLE = "#CC79A7"
CBLACK = "#000000"
CGRAY = "#7F7F7F"

N = 8
CS137_ENERGY_KEV = 661.657
ELECTRON_REST_ENERGY_KEV = 510.99895

# Helpers

def mean_with_propagated_uncertainty(values, errors):
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    N = len(values)
    if N == 0:
        return np.nan, np.nan
    return np.mean(values), np.sqrt(np.sum(errors**2)) / N

def one_minus_cos_theta(theta_deg):
    return 1.0 - np.cos(np.deg2rad(theta_deg))

def weighted_linear(x, m, b):
    return m * x + b

def inverse_with_error(x, xerr):
    y = 1.0 / x
    yerr = np.abs(xerr / (x**2))
    return y, yerr

def compute_chi2(y, yfit, yerr, n_params):
    residuals = (y - yfit) / yerr
    chi2_val = np.sum(residuals**2)
    ndof = len(y) - n_params
    p_value = chi2.sf(chi2_val, ndof) if ndof > 0 else np.nan
    return chi2_val, ndof, p_value

# Open file
with open("analysis.txt") as f:
    lines = f.readlines()
f.close()

# Load values
angles = np.array([list(map(float, lines[i].strip("[] \n").split())) for i in range(1,8*N,8)])
Erecoils = np.array([list(map(float, lines[i].strip("[] \n").split())) for i in range(2,8*N,8)])
Erecoilerrs = np.array([list(map(float, lines[i].strip("[] \n").split())) for i in range(3,8*N,8)])
Escatters = np.array([list(map(float, lines[i].strip("[] \n").split())) for i in range(4,8*N,8)])
Escattererrs = np.array([list(map(float, lines[i].strip("[] \n").split())) for i in range(5,8*N,8)])
Etots = np.array([list(map(float, lines[i].strip("[] \n").split())) for i in range(6,8*N,8)])
Eerrs = np.array([list(map(float, lines[i].strip("[] \n").split())) for i in range(7,8*N,8)])

mask = ~np.isclose(angles, 310)

angles = np.mean(angles, axis=0)
recoils = np.mean(Erecoils, axis=0)
recoil_errs = np.sqrt(np.sum(Erecoilerrs**2, axis=0))/N
sys_recoil_err = np.sqrt(np.var(Erecoils)/N)
scatters = np.mean(Escatters, axis=0)
scatter_errs = np.sqrt(np.sum(Escattererrs**2, axis=0))/N
sys_scatter_err = np.sqrt(np.var(Escatters)/N)
tots = np.mean(Etots, axis=0)
errs = np.sqrt(np.sum(Eerrs**2, axis=0))/N 
sys_err = np.sqrt(np.var(Etots)/N)

mean_energy, mean_energy_err = np.mean(Etots, axis=1), np.sqrt(np.sum(Eerrs**2)) / 10
mean_energy_no_310, mean_energy_err_no_310 = np.mean(np.resize(Etots[mask],(N,9)), axis=1), np.sqrt(np.sum(np.resize(Eerrs[mask],(N,9))**2)) / 9

sys_mean_energy_err = np.sqrt(np.var(mean_energy)/N)
mean_energy, mean_energy_err = mean_with_propagated_uncertainty(mean_energy, mean_energy_err)

sys_mean_energy_err_no_310 = np.sqrt(np.var(mean_energy_no_310)/N)
mean_energy_no_310, mean_energy_err_no_310 = mean_with_propagated_uncertainty(mean_energy_no_310, mean_energy_err_no_310)

fig, ax = plt.subplots()
ax.errorbar(
    angles, tots, yerr=np.sqrt(errs**2+sys_err**2),
    fmt='o', color=CBLUE, ecolor=CBLUE, capsize=4, markersize=8,
    label="Measured sums"
)
ax.axhline(mean_energy, color=CRED, lw=2.5, ls='--',
    label=f"Mean: {mean_energy:.1f} ± {mean_energy_err:.1f}(stat) ± {sys_mean_energy_err:.1f}(sys) keV")
ax.axhline(mean_energy_no_310, color=CGREEN, lw=2.5, ls='--',
    label=f"Mean w/o 310: {mean_energy_no_310:.1f} ± {mean_energy_err_no_310:.1f}(stat) ± {sys_mean_energy_err_no_310:.1f}(sys) keV")
ax.axhline(661.567, color=CPURPLE, lw=2.5, ls='--',
    label=f"Expected: {661.657:.1f} keV")
ax.set_xlabel("Scattering angle [deg]")
ax.set_ylabel(r"$E_{\gamma} + E_{e}$ [keV]")
ax.set_title("Sum of scatter and recoil energies vs angle")
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join("better-plots\\final", f"energy_sum_vs_angle-systematic.png"), dpi=200)
plt.close()

# Scatter Energy Plot

x = one_minus_cos_theta(angles)
y, yerr = inverse_with_error(scatters, np.sqrt(scatter_errs**2+sys_scatter_err**2))

x_theory = np.linspace(0.0, max(1.05 * np.max(x), 2.05), 500)
y_theory = (1.0 / CS137_ENERGY_KEV) + (1.0 / ELECTRON_REST_ENERGY_KEV) * x_theory

# popt, pcov = curve_fit(
#     weighted_linear,
#     x, y,
#     sigma=yerr,
#     absolute_sigma=True,
#     bounds=(0, 256),
#     maxfev=20000
# )
# perr = np.sqrt(np.diag(pcov))

# measured_Cs137_energy, measured_Cs137_energy_err = inverse_with_error(popt[1], perr[1])
# measured_electron_energy, measured_electron_energy_err = inverse_with_error(popt[0], perr[0])

# yfit = weighted_linear(x, *popt)
yfit = weighted_linear(x, 1.0 / ELECTRON_REST_ENERGY_KEV, 1.0 / CS137_ENERGY_KEV)

chi2_val, ndof, p_value = compute_chi2(y, yfit, yerr, 2)

fig, ax = plt.subplots()
ax.errorbar(
    x, y, yerr=yerr,
    fmt='o', color=CBLUE, ecolor=CBLUE, capsize=4, markersize=8,
    label="Measured data"
)
ax.plot(x_theory, y_theory, color=CRED, lw=2.5, label="Compton prediction")
# ax.plot(x_theory, weighted_linear(x_theory, *popt), color=CGREEN, lw=2.5, label="Linear fit")

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
plt.savefig(os.path.join("better-plots\\final", f"scatter_energy_vs_angle-systematic.png"), dpi=200)
plt.close()

# Recoil Energy Plot

x = 1.0 / one_minus_cos_theta(angles)
y, yerr = inverse_with_error(recoils, np.sqrt(recoil_errs**2 + sys_recoil_err**2))

x_theory = np.linspace(0.0, max(1.05 * np.max(x), 2.05), 500)
y_theory = (1.0 / CS137_ENERGY_KEV) + (ELECTRON_REST_ENERGY_KEV / CS137_ENERGY_KEV**2) * x_theory
yfit = (1.0 / CS137_ENERGY_KEV) + (ELECTRON_REST_ENERGY_KEV / CS137_ENERGY_KEV**2) * x

# popt, pcov = curve_fit(
#     weighted_linear,
#     x, y,
#     sigma=yerr,
#     absolute_sigma=True,
#     bounds=(0, 2048//FACTOR),
#     maxfev=20000
# )
# perr = np.sqrt(np.diag(pcov))

# measured_Cs137_energy, measured_Cs137_energy_err = inverse_with_error(popt[1], perr[1])
# measured_electron_energy, measured_electron_energy_err = popt[0] * popt[1]**2, perr[0] * popt[1]**2

# yfit = weighted_linear(x, *popt)

chi2_val, ndof, p_value = compute_chi2(y, yfit, yerr, 2)


fig, ax = plt.subplots()
ax.errorbar(
    x, y, yerr=yerr,
    fmt='o', color=CBLUE, ecolor=CBLUE, capsize=4, markersize=8,
    label="Measured data"
)
ax.plot(x_theory, y_theory, color=CRED, lw=2.5, label="Compton prediction")
# ax.plot(x_theory, weighted_linear(x_theory, *popt), color=CGREEN, lw=2.5, label="Linear Fit")

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
plt.savefig(os.path.join("better-plots\\final", f"recoil_energy_vs_angle-systematic.png"), dpi=200)
plt.close()