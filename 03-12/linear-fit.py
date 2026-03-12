import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, linregress

# =========================
# Data
# =========================
x_recoil = np.array([719, 124, 224, 430, 506])
x_scatter = np.array([667, 115, 215, 396, 465])
y = np.array([511, 81, 160, 302, 356])

# Poisson uncertainties (standard for count data)
sigma_y = np.sqrt(y)
weights = 1 / sigma_y**2
dof = len(y) - 2  # degrees of freedom

# =========================
# Function to Perform Fit
# =========================
def weighted_linear_fit(x, y, sigma_y):
    weights = 1 / sigma_y**2
    res = np.polyfit(x, y, 1, w=np.sqrt(weights), cov=True)
    (m, b), cov = res
    slope_err = np.sqrt(cov[0,0])
    intercept_err = np.sqrt(cov[1,1])

    y_fit = m * x + b
    chi_squared = np.sum(((y - y_fit) / sigma_y) ** 2)
    reduced_chi_squared = chi_squared / dof
    p_value = 1 - chi2.cdf(chi_squared, dof)

    return m, b, slope_err, intercept_err, y_fit, chi_squared, reduced_chi_squared, p_value

# =========================
# Perform Fits
# =========================
m_recoil, b_recoil, slope_err_recoil, intercept_err_recoil, yfit_recoil, chi2_recoil, redchi2_recoil, p_recoil = weighted_linear_fit(x_recoil, y, sigma_y)

m_scatter, b_scatter, slope_err_scatter, intercept_err_scatter, yfit_scatter, chi2_scatter, redchi2_scatter, p_scatter = weighted_linear_fit(x_scatter, y, sigma_y)

# =========================
# Plot
# =========================
plt.errorbar(x_recoil, y, yerr=sigma_y, fmt='o', label="Recoil Data")
plt.errorbar(x_scatter, y, yerr=sigma_y, fmt='s', label="Scatter Data")

x_line = np.linspace(min(min(x_recoil), min(x_scatter)),
                     max(max(x_recoil), max(x_scatter)), 200)

plt.plot(x_line, m_recoil * x_line + b_recoil, label="Recoil Fit")
plt.plot(x_line, m_scatter * x_line + b_scatter, linestyle='--', label="Scatter Fit")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Weighted Linear Fits with Chi-Squared")
plt.legend()
plt.grid()
plt.savefig("Calibrated Linear Fits.png", dpi=300)
plt.close()

# =========================
# Print Results
# =========================
print("===== RECOIL FIT =====")
print(f"Slope (m): {m_recoil:.5f} +- {slope_err_recoil:.5f}")
print(f"Intercept (b): {b_recoil:.5f} +- {intercept_err_recoil:.5f}")
print(f"Chi-squared: {chi2_recoil:.3f}")
print(f"Reduced Chi-squared: {redchi2_recoil:.3f}")
print(f"Chi-squared probability (p-value): {p_recoil:.5f}")

print("\n===== SCATTER FIT =====")
print(f"Slope (m): {m_scatter:.5f} +- {slope_err_scatter:.5f}")
print(f"Intercept (b): {b_scatter:.5f} +- {intercept_err_scatter:.5f}")
print(f"Chi-squared: {chi2_scatter:.3f}")
print(f"Reduced Chi-squared: {redchi2_scatter:.3f}")
print(f"Chi-squared probability (p-value): {p_scatter:.5f}")