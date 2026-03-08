import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

# ==============================
# Adjustable Parameters
# ==============================
SMOOTH_FACTOR = 210      # Increase if too noisy
MIN_PROMINENCE = 0.05   # Fraction of max height (increase to remove small peaks)
MIN_DISTANCE = 80       # Minimum bin separation between peaks

# ==============================
# List of .Spe files
# ==============================
spe_files = [
    '03-05-Cs137-recoil-280.Spe',
    '03-05-Cs-137-scatter-280.Spe',
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
# Smooth + Peak Detection
# ==============================
def fit_and_find_maxima(hist):
    x = np.arange(len(hist))

    # Strong smoothing spline
    spline = UnivariateSpline(x, hist, s=SMOOTH_FACTOR * len(hist))
    y_smooth = spline(x)

    # Peak detection with filtering
    peaks, properties = find_peaks(
        y_smooth,
        prominence=MIN_PROMINENCE * np.max(y_smooth),
        distance=MIN_DISTANCE
    )

    return peaks, y_smooth

# ==============================
# Load histograms
# ==============================
histograms = [read_spe_histogram(f) for f in spe_files]

# ==============================
# Plot Results
# ==============================
fig, axs = plt.subplots(1, 2, figsize=(7, 5))

for i, hist in enumerate(histograms):
    x = np.arange(len(hist))

    peaks, y_smooth = fit_and_find_maxima(hist)

    ax = axs.flat[i]
    ax.bar(x, hist, width=1.0)

    ax.set_title(spe_files[i])
    ax.set_xlabel("Bin")
    ax.set_ylabel("Counts")
    ax.legend()

plt.tight_layout()
plt.show()