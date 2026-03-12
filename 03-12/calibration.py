import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

# ==============================
# Adjustable Parameters
# ==============================
SMOOTH_FACTOR = 25      # Increase if too noisy
MIN_PROMINENCE = 0.05   # Fraction of max height (increase to remove small peaks)
MIN_DISTANCE = 80       # Minimum bin separation between peaks
LEFT_IGNORE = 0       # Ignore peaks in the first N bins (to avoid noise)

# ==============================
# List of .Spe files
# ==============================
spe_files = [
    '03-12-Ba133-recoil.Spe',
    '03-12-Ba133-scatter.Spe',
    '03-12-Na22-recoil.Spe',
    '03-12-Na22-scatter.Spe'
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
# Smooth + Peak Detection
# ==============================
def fit_and_find_maxima(hist):
    x = np.arange(len(hist))

    spline = UnivariateSpline(x, hist, s=SMOOTH_FACTOR * len(hist))
    y_smooth = spline(x)

    # Only search peaks after LEFT_IGNORE
    x_search = x[LEFT_IGNORE:]
    y_search = y_smooth[LEFT_IGNORE:]

    peaks, properties = find_peaks(
        y_search,
        prominence=MIN_PROMINENCE * np.max(y_search),
        distance=MIN_DISTANCE
    )

    # Convert peaks back to original indexing
    peaks = peaks + LEFT_IGNORE

    return peaks, y_smooth

# ==============================
# Load histograms
# ==============================
histograms = [rebin(read_spe_histogram(f), factor=2) for f in spe_files]

# ==============================
# Plot Results
# ==============================
fig, axs = plt.subplots(2, 2, figsize=(7, 5))

for i, hist in enumerate(histograms):
    x = np.arange(len(hist))

    peaks, y_smooth = fit_and_find_maxima(hist)

    ax = axs.flat[i]
    ax.bar(x[LEFT_IGNORE:], hist[LEFT_IGNORE:], alpha=0.4, label="Histogram")
    ax.plot(x[LEFT_IGNORE:], y_smooth[LEFT_IGNORE:], linewidth=2, label="Smoothed Curve")

    # only show peaks that are in the visible region
    visible_peaks = peaks[peaks >= LEFT_IGNORE]
    ax.plot(visible_peaks, y_smooth[visible_peaks], "ro", label="Filtered Peaks")

    ax.set_xlim(LEFT_IGNORE, len(hist))

    ax.axvspan(0, LEFT_IGNORE, color='gray', alpha=0.2)

    ax.set_title(spe_files[i])
    ax.set_xlabel("Bin")
    ax.set_ylabel("Counts")
    ax.legend()

    print(f"{spe_files[i]} peaks at bins: {peaks}")

plt.tight_layout()
plt.savefig("calibration_peaks.png", dpi=300)
plt.close()