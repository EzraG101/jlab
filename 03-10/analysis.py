import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import chi2

# ==============================
# Adjustable Parameters
# ==============================
SMOOTH_FACTOR = 120
MIN_PROMINENCE = 0.05
MIN_DISTANCE = 80

RECOIL_SLOPE = 0.70107
RECOIL_INTERCEPT = -8.40091
SCATTER_SLOPE = 0.65156
SCATTER_INTERCEPT = -8.21614

# ==============================
# Files organized by angle
# ==============================
spe_files = {
    205: [
        "03-10-Cs137-Recoil-205.Spe",
        "03-10-Cs137-Scatter-205.Spe"
    ],
    210: [
        "03-10-Cs137-Recoil-210.Spe",
        "03-10-Cs137-Scatter-210.Spe"
    ],
    215: [
        "03-10-Cs137-Recoil-215.Spe",
        "03-10-Cs137-Scatter-215.Spe"
    ],
    220: [
        "03-10-Cs137-Recoil-220.Spe",
        "03-10-Cs137-Scatter-220.Spe"
    ],
    225: [
        "03-10-Cs137-Recoil-225.Spe",
        "03-10-Cs137-Scatter-225.Spe"
    ],
    250: [
        "03-10-Cs137-Recoil-250.Spe",
        "03-10-Cs137-Scatter-250.Spe"
    ],
    310: [
        "03-10-Cs137-Recoil-310.Spe",
        "03-10-Cs137-Scatter-310.Spe"
    ]
}

# ==============================
# Read .Spe histogram
# ==============================
def read_spe_histogram(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    try:
        start_idx = next(i for i, line in enumerate(lines) if "<<DATA>>" in line) + 1
    except StopIteration:
        start_idx = 0

    data = []
    for line in lines[start_idx:start_idx + 2048]:
        try:
            data.append(int(line.strip()))
        except ValueError:
            continue

    return np.array(data)

# ==============================
# Rebin
# ==============================
def rebin(hist, factor=2):
    n = len(hist) // factor
    hist = hist[:n * factor]
    return hist.reshape(n, factor).sum(axis=1)

# ==============================
# Smooth + Peak Detection
# ==============================
def gaussian_linear(x, A, mu, sigma, m, b):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + (m*x + b)

# def fit_and_find_maxima(hist):
#     max_bin = np.argmax(hist)      # bin of absolute maximum
#     return np.array([max_bin])

def fit_peak(hist, window=40):

    bins = np.arange(len(hist))
    max_bin = np.argmax(hist)

    # define fitting region
    left = max(max_bin - window, 0)
    right = min(max_bin + window, len(hist))

    x = bins[left:right]
    y = hist[left:right]

    # initial parameter guesses
    A0 = hist[max_bin]
    mu0 = max_bin
    sigma0 = window / 3
    m0 = 0
    b0 = np.min(y)

    p0 = [A0, mu0, sigma0, m0, b0]

    uncertainties = np.sqrt(y)

    popt, pcov = curve_fit(gaussian_linear, x, y, p0=p0, sigma=uncertainties, absolute_sigma=True, maxfev=10000)

    A, mu, sigma, m, b = popt

    return mu, popt, (left, right)

# ==============================
# Run analysis for each angle
# ==============================
angles = []
energy_sums = []

for angle, files in spe_files.items():

    histograms = [rebin(read_spe_histogram(f), factor=2) for f in files]

    # ==========================================
    # Figure 1 : Full histograms
    # ==========================================
    fig_full, axs_full = plt.subplots(1, 2, figsize=(9,5))

    # ==========================================
    # Figure 2 : Zoomed peak fits
    # ==========================================
    fig_zoom, axs_zoom = plt.subplots(1, 2, figsize=(9,5))

    for i, hist in enumerate(histograms):

        bins = np.arange(len(hist))

        # -------------------------
        # Calibration
        # -------------------------
        if i == 0:
            slope = RECOIL_SLOPE
            intercept = RECOIL_INTERCEPT
            label = "Recoil"
        else:
            slope = SCATTER_SLOPE
            intercept = SCATTER_INTERCEPT
            label = "Scatter"

        energy = slope * bins + intercept

        # -------------------------
        # Peak fit
        # -------------------------
        mu, popt, (left, right) = fit_peak(hist)
        peak_energy = slope * mu + intercept

        if i == 0:
            recoil_energy = peak_energy
        else:
            scatter_energy = peak_energy

        # ==========================================
        # Chi^2 calculation
        # ==========================================
        x = np.arange(left, right)
        y = hist[left:right]

        model = gaussian_linear(x, *popt)

        # Poisson uncertainties
        sigma = np.sqrt(y)
        sigma[sigma == 0] = 1

        chi2_val = np.sum(((y - model) / sigma)**2)

        ndof = len(y) - len(popt)

        chi2_prob = chi2.sf(chi2_val, ndof)

        print(f"{files[i]} peak energy (keV): {peak_energy:.2f}")

        # =====================================================
        # FULL HISTOGRAM PLOT
        # =====================================================
        ax = axs_full[i]

        ax.bar(
            energy,
            hist,
            width=slope,
            color="royalblue",
            label="Histogram"
        )

        max_bin = np.argmax(hist)

        ax.plot(
            slope*max_bin + intercept,
            hist[max_bin],
            "o",
            color="darkgreen",
            label="Maximum bin"
        )

        ax.axvspan(
            slope*left + intercept,
            slope*right + intercept,
            color="gold",
            alpha=0.3,
            label="Fit region"
        )

        ax.set_title(f"{label} ({angle}°)")
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Counts")

        ax.legend()

        # =====================================================
        # ZOOMED FIT PLOT
        # =====================================================
        ax2 = axs_zoom[i]

        x = np.arange(left, right)
        y = hist[left:right]

        energy_zoom = slope * x + intercept

        ax2.bar(
            energy_zoom,
            y,
            width=slope,
            color="royalblue",
            label="Measured histogram"
        )

        # fitted curve
        xfit = np.linspace(left, right, 400)
        yfit = gaussian_linear(xfit, *popt)

        ax2.plot(
            slope*xfit + intercept,
            yfit,
            color="crimson",
            linewidth=2,
            label="Gaussian + linear fit"
        )

        ax2.axvline(
            peak_energy,
            color="darkgreen",
            linestyle="--",
            linewidth=2,
            label=f"Peak = {peak_energy:.1f} keV"
        )

        ax2.set_title(f"{label} Peak Fit ({angle}°)")
        ax2.set_xlabel("Energy (keV)")
        ax2.set_ylabel("Counts")

        ax2.legend()

        ax2.text(
            0.05,
            0.95,
            f"$\\chi^2$ = {chi2_val:.1f}\n"
            f"dof = {ndof}\n"
            f"$P(\\chi^2)$ = {chi2_prob:.3f}",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    # ==========================================
    # Save figures
    # ==========================================
    plt.figure(fig_full.number)
    plt.tight_layout()
    plt.savefig(f"Cs137-{angle}-histograms.png", dpi=300)
    plt.close(fig_full)

    plt.figure(fig_zoom.number)
    plt.tight_layout()
    plt.savefig(f"Cs137-{angle}-peakfits.png", dpi=300)
    plt.close(fig_zoom)

    # store sum after both recoil and scatter processed
    if angle != 310:
        angles.append(angle)
        energy_sums.append(recoil_energy + scatter_energy)

angles = np.array(angles)
energy_sums = np.array(energy_sums)

plt.figure(figsize=(6,5))

plt.scatter(
    angles,
    energy_sums,
    color="royalblue",
    s=60,
    label="Measured sums"
)

plt.axhline(
    662,
    color="crimson",
    linestyle="--",
    label="Expected (662 keV)"
)

plt.axhline(
    np.mean(energy_sums),
    color="blue",
    linestyle="--",
    label="Mean Energy Sum"
)

print(f"Mean energy sum (keV): {np.mean(energy_sums):.1f}")

plt.xlabel("Scattering Angle (degrees)")
plt.ylabel("Scatter + Recoil Energy (keV)")
plt.title("Energy Conservation Check")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("energy_sum_vs_angle.png", dpi=300)
plt.close()