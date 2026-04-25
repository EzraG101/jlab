# calibration_gain_plot.py
#
# Reads "calibration 3-19 multimeter" sheet from data.xlsx, computes:
#   gain = Vo/Vi
#   mean(gain), sigma(gain), SE(gain)=sigma/sqrt(n-1)   (per your definition)
#   mean(gain^2) = mean(gain)^2
#   SE(gain^2) propagated from SE(gain):  d(g^2)=2*g*dg
#
# Plots frequency vs gain^2 with error bars, SAVES, and CLOSES immediately.
#
# Expected columns (names can vary slightly):
#   f (khz) | Vi (mV) (pre) | Vo (mV) (post) | (optional) gain^2

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# print("python:", sys.executable)
# print("numpy:", np.__file__)
# print("numpy version:", np.__version__)
# print("has trapz:", hasattr(np, "trapz"))

XLSX_PATH = "data.xlsx"
SHEET_NAME = 2
OUTDIR = "plots"
PLOT_BASENAME = "gain2_vs_frequency"


def find_col(df, targets):
    """Find a column in df whose normalized name matches one of targets."""
    norm = {c: " ".join(str(c).strip().lower().split()) for c in df.columns}
    target_norm = {" ".join(t.strip().lower().split()) for t in targets}
    for c, nc in norm.items():
        if nc in target_norm:
            return c
    raise KeyError(f"Could not find any of {targets}. Found columns: {list(df.columns)}")


def save_and_close(fig, outpath, dpi=300):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    raw = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME, engine="openpyxl")

    f_col  = find_col(raw, ["f (khz)", "f(khz)", "frequency (khz)", "freq (khz)"])
    vi_col = find_col(raw, ["vi (mv) (pre)", "vi (mv)", "vi pre"])
    vo_col = find_col(raw, ["vo (mv) (post)", "vo (mv)", "vo post"])

    df = raw[[f_col, vi_col, vo_col]].copy()
    df = df.rename(columns={f_col: "f_khz", vi_col: "Vi_mV", vo_col: "Vo_mV"})

    # numeric + drop bad rows
    for c in ["f_khz", "Vi_mV", "Vo_mV"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["f_khz", "Vi_mV", "Vo_mV"])
    df = df[df["Vi_mV"] != 0]

    # compute gain per measurement
    df["gain"] = df["Vo_mV"] / df["Vi_mV"]
    df["gain2"] = df["gain"] ** 2

    # summarize by frequency (kHz)
    g = df.groupby("f_khz")["gain"]
    summary = g.agg(
        n="count",
        gain_mean="mean",
        gain_sigma=lambda x: x.std(ddof=1),   # sample stdev
    ).reset_index()

    # Your SE definition: sigma/sqrt(n-1)
    summary["gain_se"] = summary["gain_sigma"] / np.sqrt(summary["n"] - 1)

    # Plot target: gain^2 computed from mean gain
    summary["gain2_mean"] = summary["gain_mean"] ** 2

    # Propagate SE: y=g^2 => dy = 2*g*dg (using mean gain and SE of gain)
    summary["gain2_se"] = 2.0 * np.abs(summary["gain_mean"]) * summary["gain_se"]

    summary = summary.sort_values("f_khz")

    # Save summary table
    summary.to_csv("gain_summary_by_frequency.csv", index=False)

    # ---- Plot: frequency vs gain^2 with error bars; save and close immediately ----
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.errorbar(
        summary["f_khz"],
        summary["gain2_mean"],
        yerr=summary["gain2_se"],
        fmt="o",
        capsize=3,
        linewidth=1,
        markersize=4,
        barsabove=True,  # error bars above points
    )
    # ax.set_xscale("log")  # comment out if you want linear frequency axis
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Gain$^2$ (from mean gain)")

    ax.grid(True, which="both", alpha=0.0)

    outpath_png = os.path.join(OUTDIR, f"{PLOT_BASENAME}.png")
    outpath_pdf = os.path.join(OUTDIR, f"{PLOT_BASENAME}.pdf")
    save_and_close(fig, outpath_png, dpi=300)

    # If you also want a PDF copy, make a new figure (since we closed it)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.errorbar(
        summary["f_khz"],
        summary["gain2_mean"],
        yerr=summary["gain2_se"],
        fmt="o",
        capsize=3,
        linewidth=1,
        markersize=4,
        barsabove=True,  # error bars above points
    )
    # ax.set_xscale("log")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Gain$^2$ (from mean gain)")
    ax.grid(True, which="both", alpha=0.0)
    save_and_close(fig, outpath_pdf, dpi=300)

    # Print a compact view
    print(summary[["f_khz", "n", "gain_mean", "gain_se", "gain2_mean", "gain2_se"]])

    results = estimate_G_from_summary(summary)
    print(results)

def integrate_gain2_trapz(f, y):
    """Trapezoid integral ∫ y df with f in ascending order."""
    f = np.asarray(f)
    y = np.asarray(y)
    idx = np.argsort(f)
    f = f[idx]
    y = y[idx]
    return np.trapezoid(y, f)

def estimate_G_from_summary(summary, freq_col="f_khz",
                            y_col="gain2_mean", yerr_col="gain2_se",
                            freq_unit="kHz", n_mc=20000, seed=0,
                            clip_nonnegative=True):
    """
    Returns:
      G_hat: integral of mean curve over measured frequency range
      G_mc_mean, G_mc_std: Monte Carlo mean and std for uncertainty
      (all in units of gain^2 * freq_unit)
    """
    df = summary[[freq_col, y_col, yerr_col]].dropna().copy()
    df = df.sort_values(freq_col)

    f = df[freq_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    s = df[yerr_col].to_numpy(dtype=float)

    # Integral of the mean curve over measured interval
    G_hat = integrate_gain2_trapz(f, y)

    # Monte Carlo uncertainty
    rng = np.random.default_rng(seed)
    # Draw shape: (n_mc, n_points)
    y_draw = rng.normal(loc=y, scale=s, size=(n_mc, len(y)))

    if clip_nonnegative:
        # gain^2 must be >= 0; avoids negative draws if errors are large
        y_draw = np.clip(y_draw, 0.0, None)

    # Integrate each draw
    G_draw = np.trapezoid(y_draw, f, axis=1)

    return {
        "G_hat": G_hat,
        "G_mc_mean": float(G_draw.mean()),
        "G_mc_std": float(G_draw.std(ddof=1)),
        "freq_unit": freq_unit,
        "f_min": float(f.min()),
        "f_max": float(f.max()),
        "n_mc": n_mc
    }




if __name__ == "__main__":
    main()