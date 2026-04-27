import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 13,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "figure.figsize": (10, 6),
    "axes.grid": False,
})

CBLUE = "#0072B2"
CORANGE = "#E69F00"
CGREEN = "#009E73"
CRED = "#D55E00"
CPURPLE = "#CC79A7"
CBLACK = "#000000"
CGRAY = "#7F7F7F"

# ============================================================
# Helpers
# ============================================================

def parse_spe_filename(filename):
    """
    Expected formats:
      mm-dd-Na22-scatter.Spe
      mm-dd-Ba133-recoil.Spe
      mm-dd-Cs137-scatter-30.Spe
    """
    base = os.path.basename(filename)
    pattern_no_angle = r"^(?P<date>\d{2}-\d{2})-(?P<source>Na22|Ba133|Cs137)-(?P<stype>scatter|recoil)\.Spe$"
    pattern_angle = r"^(?P<date>\d{2}-\d{2})-(?P<source>Na22|Ba133|Cs137)-(?P<stype>scatter|recoil)-(?P<angle>-?\d+(\.\d+)?)\.Spe$"

    m = re.match(pattern_angle, base, re.IGNORECASE)
    if m:
        return m.group("date"), m.group("source"), m.group("stype").lower(), float(m.group("angle"))

    m = re.match(pattern_no_angle, base, re.IGNORECASE)
    if m:
        return m.group("date"), m.group("source"), m.group("stype").lower(), None

    raise ValueError(f"Filename does not match expected format: {filename}")

def read_spe_file(filepath):
    """
    Reads a common ORTEC-style .Spe file.
    Assumes counts are between '$DATA:' and the next section starting with '$'.
    """
    with open(filepath, "r", encoding="latin-1") as f:
        lines = [line.strip() for line in f.readlines()]

    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("$DATA"):
            data_start = i
            break

    if data_start is None:
        raise ValueError(f"Could not find $DATA section in {filepath}")

    counts = []
    started_numbers = False
    for line in lines[data_start + 1:]:
        if line.startswith("$") and started_numbers:
            break
        if not line:
            continue

        parts = line.split()

        if len(parts) == 2 and not started_numbers:
            try:
                int(parts[0]); int(parts[1])
                continue
            except ValueError:
                pass

        try:
            counts.append(float(parts[0]))
            started_numbers = True
        except Exception:
            continue

    if len(counts) == 0:
        raise ValueError(f"No counts found in $DATA section for {filepath}")

    return np.array(counts, dtype=float)

def rebin_counts(counts, factor=2):
    if len(counts) % factor != 0:
        raise ValueError(f"Counts length {len(counts)} not divisible by rebin factor {factor}")
    return counts.reshape(-1, factor).sum(axis=1)

def sanitize_filename(s):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s.strip())

# ============================================================
# Plotting
# ============================================================

def plot_single_histogram(bins, counts, title, outpath):
    fig, ax = plt.subplots()
    ax.bar(bins, counts, width=1.0, color=CBLUE, edgecolor=None, linewidth=0)

    ax.set_xlabel("Bin")
    ax.set_ylabel("Counts")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def plot_daily_calibration_overview(day_data, date, outdir):
    """
    day_data keys are (source, spec_type)
    Layout:
        Na22 scatter | Ba133 scatter
        Na22 recoil  | Ba133 recoil
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    layout = [
        ("Na22", "scatter"),
        ("Ba133", "scatter"),
        ("Na22", "recoil"),
        ("Ba133", "recoil"),
    ]

    for ax, key in zip(axes.flat, layout):
        source, spec_type = key
        if key not in day_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{source} {spec_type}")
            ax.set_xlabel("Bin")
            ax.set_ylabel("Counts")
            continue

        bins, counts, filename = day_data[key]
        ax.bar(bins, counts, width=1.0, color=CBLUE, edgecolor=None, linewidth=0)
        ax.set_title(f"{source} {spec_type}")
        ax.set_xlabel("Bin")
        ax.set_ylabel("Counts")

    fig.suptitle(f"{date} Na22 / Ba133 histograms", fontsize=22)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(outdir, f"{date}_Na22_Ba133_overview.png"), dpi=200)
    plt.close(fig)

# ============================================================
# Main
# ============================================================

def main(data_dir="data", raw_dir="raw"):
    os.makedirs(raw_dir, exist_ok=True)

    daily_calibration_data = {}

    spe_files = sorted(glob.glob(os.path.join(data_dir, "*.Spe")))
    if len(spe_files) == 0:
        print(f"No .Spe files found in {data_dir}")
        return

    for filepath in spe_files:
        filename = os.path.basename(filepath)
        date, source, spec_type, angle = parse_spe_filename(filename)

        counts_raw = read_spe_file(filepath)

        if len(counts_raw) == 2048:
            counts = rebin_counts(counts_raw, factor=2)
        elif len(counts_raw) == 1024:
            counts = counts_raw.copy()
        else:
            raise ValueError(f"Unexpected number of channels in {filename}: {len(counts_raw)}")

        bins = np.arange(len(counts), dtype=float)

        # Title for single raw histogram
        if angle is None:
            title = f"{date} {source} {spec_type}"
        else:
            title = f"{date} {source} {spec_type} {angle:.1f}°"

        outname = sanitize_filename(title) + ".png"
        plot_single_histogram(
            bins=bins,
            counts=counts,
            title=title,
            outpath=os.path.join(raw_dir, outname)
        )

        # Store Na22/Ba133 by day for 2x2 overview
        if source in ["Na22", "Ba133"]:
            daily_calibration_data.setdefault(date, {})
            daily_calibration_data[date][(source, spec_type)] = (bins, counts, filename)

    # Make daily 2x2 overviews
    for date, day_data in sorted(daily_calibration_data.items()):
        plot_daily_calibration_overview(day_data, date, raw_dir)

    print(f"Saved raw rebinned histograms to: {raw_dir}")

if __name__ == "__main__":
    main(data_dir="data", raw_dir="raw")