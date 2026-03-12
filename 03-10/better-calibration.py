import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import chi2
import os

# ==============================
# Adjustable Parameters
# ==============================
SMOOTH_FACTOR = 25
MIN_PROMINENCE = 0.05
MIN_DISTANCE = 80
LEFT_IGNORE = 100
WINDOW = 15   # +/- bins for gaussian fit

# ==============================
# List of .Spe files
# ==============================
spe_files = [
    '03-10-Ba133-recoil.Spe',
    '03-10-Ba133-scatter.Spe',
    '03-10-Na22-recoil.Spe',
    '03-10-Na22-scatter.Spe'
]

# ==============================
# Read .Spe histogram
# ==============================
def read_spe_histogram(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    try:
        start_idx = next(i for i, line in enumerate(lines) if '<<DATA>>' in line) + 1
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
# Gaussian + Linear Background
# ==============================
def gaussian_linear(x, A, mu, sigma, m, b):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + m * x + b

# ==============================
# Smooth + Peak Detection
# ==============================
def fit_and_find_maxima(hist):
    x = np.arange(len(hist))

    spline = UnivariateSpline(x, hist, s=SMOOTH_FACTOR * len(hist))
    y_smooth = spline(x)

    x_search = x[LEFT_IGNORE:]
    y_search = y_smooth[LEFT_IGNORE:]

    peaks, properties = find_peaks(
        y_search,
        prominence=MIN_PROMINENCE * np.max(y_search),
        distance=MIN_DISTANCE
    )

    peaks = peaks + LEFT_IGNORE

    return peaks, y_smooth

# ==============================
# Gaussian Fit Around Peak
# ==============================
def fit_peak(hist, peak, filename, peak_index):
    x = np.arange(len(hist))

    # Define zoom window around peak
    left = max(0, peak - WINDOW)
    right = min(len(hist), peak + WINDOW + 1)

    x_fit = x[left:right]
    y_fit = hist[left:right]

    # Poisson uncertainties
    sigma = np.sqrt(np.maximum(y_fit, 1))

    # Initial guesses
    A0 = np.max(y_fit)
    mu0 = peak
    sigma0 = 5
    m0 = 0
    b0 = np.min(y_fit)
    p0 = [A0, mu0, sigma0, m0, b0]

    # Gaussian + linear fit without bounds
    popt, pcov = curve_fit(
        gaussian_linear,
        x_fit,
        y_fit,
        sigma=sigma,
        p0=p0,
        absolute_sigma=True,
        maxfev=10000
    )

    A, mu, sig, m, b = popt
    mu_fit = mu

    # If mu left the zoom window, reset to original peak
    if mu < left or mu > right:
        mu = mu0

    # Chi-square
    y_model = gaussian_linear(x_fit, A, mu_fit, sig, m, b)
    chi2_val = np.sum(((y_fit - y_model) / sigma)**2)
    dof = len(y_fit) - len(popt)
    chi2_prob = chi2.sf(chi2_val, dof)

    # ==============================
    # Plot zoomed region
    # ==============================
    plt.figure(figsize=(5,4))
    plt.bar(x_fit, y_fit, color="gray", alpha=0.8, label="Histogram", width=1)
    plt.plot(x_fit, y_model, color="tab:blue", linewidth=2, label="Gaussian + Linear Fit")
    plt.axvline(mu, color="tab:red", linestyle="--", linewidth=2, label=f"Peak μ = {mu:.2f}")

    plt.xlabel("Bin")
    plt.ylabel("Counts")
    plt.title(f"{filename}\nPeak Fit")

    text = f"$\\chi^2$ = {chi2_val:.1f}\nDOF = {dof}\nP = {chi2_prob:.3f}"
    plt.text(0.97, 0.97, text, transform=plt.gca().transAxes,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
             fontsize=9)

    plt.legend(loc="upper left")
    plt.tight_layout()
    os.makedirs("peak_fits", exist_ok=True)
    plt.savefig(f"peak_fits/{filename}_peak{peak_index}.png", dpi=300)
    plt.close()

    return mu, sig, chi2_val, chi2_prob


# ==============================
# Load histograms
# ==============================
histograms = [rebin(read_spe_histogram(f), factor=2) for f in spe_files]

# ==============================
# Plot Results
# ==============================
fig, axs = plt.subplots(2, 2, figsize=(7, 5))

# ==============================
# Store refined peak locations
# ==============================
all_peak_positions = {}

for i, hist in enumerate(histograms):
    x = np.arange(len(hist))

    peaks, y_smooth = fit_and_find_maxima(hist)

    ax = axs.flat[i]

    ax.bar(x[LEFT_IGNORE:], hist[LEFT_IGNORE:], alpha=0.4, label="Histogram")
    ax.plot(x[LEFT_IGNORE:], y_smooth[LEFT_IGNORE:], linewidth=2, label="Smoothed Curve")

    visible_peaks = peaks[peaks >= LEFT_IGNORE]
    ax.plot(visible_peaks, y_smooth[visible_peaks], "ro", label="Filtered Peaks")

    ax.set_xlim(LEFT_IGNORE, len(hist))
    ax.axvspan(0, LEFT_IGNORE, color='gray', alpha=0.2)

    ax.set_title(spe_files[i])
    ax.set_xlabel("Bin")
    ax.set_ylabel("Counts")
    ax.legend()

    refined_peaks = []

    for j, peak in enumerate(peaks):

        mu, sigma, chi2_val, chi2_prob = fit_peak(hist, peak, spe_files[i], j)

        refined_peaks.append(mu)

    all_peak_positions[spe_files[i]] = refined_peaks

plt.tight_layout()
plt.savefig("calibration_peaks.png", dpi=300)
plt.close()

print("\n==============================")
print("FINAL REFINED PEAK POSITIONS")
print("==============================")

for filename, peaks in all_peak_positions.items():
    print(f"\n{filename}")
    for i, p in enumerate(peaks):
        print(f"  Peak {i}: bin = {p:.3f}")