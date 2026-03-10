import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

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
def fit_and_find_maxima(hist):
    max_bin = np.argmax(hist)      # bin of absolute maximum
    return np.array([max_bin])

# ==============================
# Run analysis for each angle
# ==============================
for angle, files in spe_files.items():

    histograms = [rebin(read_spe_histogram(f), factor=2) for f in files]

    fig, axs = plt.subplots(1, 2, figsize=(7, 5))

    for i, hist in enumerate(histograms):

        bins = np.arange(len(hist))

        # Choose correct calibration
        if i == 0:  # recoil
            slope = RECOIL_SLOPE
            intercept = RECOIL_INTERCEPT
            label = "Recoil"
        else:       # scatter
            slope = SCATTER_SLOPE
            intercept = SCATTER_INTERCEPT
            label = "Scatter"

        energy = slope * bins + intercept

        peaks = fit_and_find_maxima(hist)
        peak_energies = slope * peaks + intercept

        ax = axs[i]
        ax.bar(energy, hist, width=slope)

        ax.set_title(f"{label} ({angle}°)")
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Counts")

        # mark peaks
        ax.plot(peak_energies, hist[peaks], "ro")

        print(f"{files[i]} peaks at energies (keV): {peak_energies}")

    plt.tight_layout()
    plt.savefig(f"Cs137-{angle}-histograms.png", dpi=300)
    plt.close()