import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_calibration_data(file_path="data.xlsx", sheet_name="calibration 331 multimeter"):
    """
    Load the first table in the given sheet containing columns
    'f (khz)' and 'gain^2'.

    Returns
    -------
    summary : pandas.DataFrame
        Columns:
            f_khz
            gain2_mean
            gain2_std
            n
            gain2_sem
    """
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    header_row = None
    for i, row in raw.iterrows():
        vals = row.astype(str).str.strip().tolist()
        if "f (khz)" in vals and "gain^2" in vals:
            header_row = i
            break

    if header_row is None:
        raise ValueError(
            f"Could not find 'f (khz)' and 'gain^2' headers in sheet '{sheet_name}'."
        )

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)

    df = df[["f (khz)", "gain^2"]].copy()
    df["f (khz)"] = pd.to_numeric(df["f (khz)"], errors="coerce")
    df["gain^2"] = pd.to_numeric(df["gain^2"], errors="coerce")
    df = df.dropna()

    grouped = df.groupby("f (khz)")["gain^2"]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary.rename(columns={
        "f (khz)": "f_khz",
        "mean": "gain2_mean",
        "std": "gain2_std",
        "count": "n"
    }, inplace=True)

    summary["gain2_std"] = summary["gain2_std"].fillna(0.0)
    summary["gain2_sem"] = summary["gain2_std"] / np.sqrt(summary["n"])
    summary["gain2_sem"] = summary["gain2_sem"].fillna(0.0)

    summary = summary.sort_values("f_khz").reset_index(drop=True)
    return summary


def compute_G_from_arrays(f_hz, gain2, C_F=0.0, R_ohm=0.0):
    """
    Compute
        G = ∫ gain^2(f) / [1 + (2π f C R)^2] df

    Parameters
    ----------
    f_hz : array
        Frequency in Hz
    gain2 : array
        gain^2 values
    C_F : float
        Capacitance in farads
    R_ohm : float
        Resistance in ohms
    """
    weight = 1.0 / (1.0 + (2.0 * np.pi * f_hz * C_F * R_ohm)**2)
    integrand = gain2 * weight
    G = np.trapezoid(integrand, f_hz)
    return G, weight, integrand


def safe_label(x):
    """
    Convert a number to a filename-safe string.
    Example: 50.9 -> '50p9'
    """
    s = f"{x:g}"
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    return s


def compute_G_single(
    calibration_summary,
    C_pF=0.0,
    R_kohm=0.0,
    C_err_pF=0.0,
    R_err_kohm=0.0,
    n_mc=20000,
    save_plot=True,
    plot_path="plot.png",
    save_csv=True,
    csv_path="processed.csv",
    return_details=False
):
    """
    Compute G for one resistance value.

    Inputs use:
      C_pF      in pF
      R_kohm    in kΩ
      C_err_pF  in pF
      R_err_kohm in kΩ

    Returns
    -------
    dict or float
    """
    summary = calibration_summary.copy()

    # Convert to SI units internally
    C_F = C_pF * 1e-12
    R_ohm = R_kohm * 1e3
    C_err_F = C_err_pF * 1e-12
    R_err_ohm = R_err_kohm * 1e3

    f_khz = summary["f_khz"].to_numpy()
    f_hz = f_khz * 1e3
    gain2_mean = summary["gain2_mean"].to_numpy()
    gain2_sem = summary["gain2_sem"].to_numpy()

    # Central value
    G, weight, integrand = compute_G_from_arrays(f_hz, gain2_mean, C_F=C_F, R_ohm=R_ohm)

    # Save processed frequency-by-frequency data
    summary_out = summary.copy()
    summary_out["f_hz"] = f_hz
    summary_out["weight"] = weight
    summary_out["weighted_integrand"] = integrand
    summary_out["C_pF"] = C_pF
    summary_out["R_kohm"] = R_kohm

    if save_csv:
        summary_out.to_csv(csv_path, index=False)

    # Monte Carlo uncertainty propagation
    rng = np.random.default_rng(seed=12345)

    # Sample gain^2
    gain2_samples = np.tile(gain2_mean, (n_mc, 1))
    nonzero_gain = gain2_sem > 0
    if np.any(nonzero_gain):
        gain2_samples[:, nonzero_gain] = rng.normal(
            loc=gain2_mean[nonzero_gain],
            scale=gain2_sem[nonzero_gain],
            size=(n_mc, np.sum(nonzero_gain))
        )
    gain2_samples = np.clip(gain2_samples, 0, None)

    # Sample C and R
    if C_err_F > 0:
        C_samples = rng.normal(loc=C_F, scale=C_err_F, size=n_mc)
    else:
        C_samples = np.full(n_mc, C_F)

    if R_err_ohm > 0:
        R_samples = rng.normal(loc=R_ohm, scale=R_err_ohm, size=n_mc)
    else:
        R_samples = np.full(n_mc, R_ohm)

    # Keep physical values
    C_samples = np.clip(C_samples, 0, None)
    R_samples = np.clip(R_samples, 0, None)

    # Weight per Monte Carlo sample
    weight_samples = 1.0 / (
        1.0 + (2.0 * np.pi * f_hz[None, :] * C_samples[:, None] * R_samples[:, None])**2
    )

    integrand_samples = gain2_samples * weight_samples
    G_samples = np.trapezoid(integrand_samples, f_hz, axis=1)

    G_mc_mean = np.mean(G_samples)
    G_mc_std = np.std(G_samples, ddof=1)
    G_ci_low, G_ci_high = np.percentile(G_samples, [2.5, 97.5])

    if save_plot:
        fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        ax[0].plot(f_khz, gain2_mean, 'o-', label='Mean gain$^2$')
        ax[0].errorbar(
            f_khz, gain2_mean, yerr=gain2_sem,
            fmt='none', capsize=3, label='SEM from repeats'
        )
        ax[0].set_ylabel("gain$^2$")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()

        ax[1].plot(
            f_khz, integrand, 'o-', color='tab:red',
            label=r'$\mathrm{gain}^2/[1+(2\pi f C R)^2]$'
        )
        ax[1].set_xlabel("Frequency (kHz)")
        ax[1].set_ylabel("Weighted integrand")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()

        fig.suptitle(
            f"G = {G:.6g} gain$^2$·Hz\n"
            f"C = {C_pF:.6g} ± {C_err_pF:.3g} pF, "
            f"R = {R_kohm:.6g} ± {R_err_kohm:.3g} kΩ\n"
            f"MC: {G_mc_mean:.6g} ± {G_mc_std:.3g} gain$^2$·Hz"
        )
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    results = {
        "G": G,
        "G_mc_mean": G_mc_mean,
        "G_mc_std": G_mc_std,
        "G_ci_95_low": G_ci_low,
        "G_ci_95_high": G_ci_high,
        "C_pF": C_pF,
        "R_kohm": R_kohm,
        "C_err_pF": C_err_pF,
        "R_err_kohm": R_err_kohm,
        "plot_path": plot_path if save_plot else None,
        "csv_path": csv_path if save_csv else None,
        "summary": summary_out
    }

    if return_details:
        return results
    return G


def compute_G_for_resistances(
    R_values_kohm,
    C_pF=0.0,
    C_err_pF=0.0,
    R_err_kohm=0.0,
    file_path="data.xlsx",
    sheet_name="calibration 331 multimeter",
    n_mc=20000,
    output_dir="G_results",
    summary_csv_name="G_summary.csv"
):
    """
    Compute G for multiple resistance values.

    Parameters
    ----------
    R_values_kohm : list or array
        Resistances in kΩ
    C_pF : float
        Capacitance in pF
    C_err_pF : float
        Uncertainty in capacitance in pF
    R_err_kohm : float or list
        Uncertainty in resistance in kΩ.
        Can be:
            - single number applied to all R
            - list/array same length as R_values_kohm
    """
    os.makedirs(output_dir, exist_ok=True)

    calibration_summary = load_calibration_data(
        file_path=file_path,
        sheet_name=sheet_name
    )

    # Allow scalar or list for resistance uncertainty
    if np.isscalar(R_err_kohm):
        R_err_list = [R_err_kohm] * len(R_values_kohm)
    else:
        R_err_list = list(R_err_kohm)
        if len(R_err_list) != len(R_values_kohm):
            raise ValueError("If R_err_kohm is a list, it must match R_values_kohm length.")

    all_results = []

    for R_kohm, Rerr_kohm in zip(R_values_kohm, R_err_list):
        r_label = safe_label(R_kohm)

        plot_path = os.path.join(output_dir, f"G_plot_R_{r_label}_kohm.png")
        csv_path = os.path.join(output_dir, f"G_processed_R_{r_label}_kohm.csv")

        result = compute_G_single(
            calibration_summary=calibration_summary,
            C_pF=C_pF,
            R_kohm=R_kohm,
            C_err_pF=C_err_pF,
            R_err_kohm=Rerr_kohm,
            n_mc=n_mc,
            save_plot=True,
            plot_path=plot_path,
            save_csv=True,
            csv_path=csv_path,
            return_details=True
        )

        all_results.append({
            "R_kohm": result["R_kohm"],
            "R_err_kohm": result["R_err_kohm"],
            "C_pF": result["C_pF"],
            "C_err_pF": result["C_err_pF"],
            "G": result["G"],
            "G_mc_mean": result["G_mc_mean"],
            "G_mc_std": result["G_mc_std"],
            "G_ci_95_low": result["G_ci_95_low"],
            "G_ci_95_high": result["G_ci_95_high"],
            "plot_path": result["plot_path"],
            "csv_path": result["csv_path"]
        })

    results_df = pd.DataFrame(all_results)
    summary_csv_path = os.path.join(output_dir, summary_csv_name)
    results_df.to_csv(summary_csv_path, index=False)

    return results_df


if __name__ == "__main__":
    # ---------------------------------
    # USER INPUTS
    # ---------------------------------
    file_path = "data.xlsx"
    sheet_name = "calibration 331 multimeter"

    # Capacitance in pF
    C_pF = 25 + 48
    C_err_pF = 0

    # Resistances in kΩ
    R_values_kohm = [334,  2753, 98.5, 815, 999, 178.4, 9.88, 46.7, 26.90]

    # Either one uncertainty for all resistances:
    # R_err_kohm = 2.0

    # Or a list matching R_values_kohm, e.g.
    R_err_kohm = [0, 2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

    results_df = compute_G_for_resistances(
        R_values_kohm=R_values_kohm,
        C_pF=C_pF,
        C_err_pF=C_err_pF,
        R_err_kohm=R_err_kohm,
        file_path=file_path,
        sheet_name=sheet_name,
        n_mc=20000,
        output_dir="G_results",
        summary_csv_name="G_summary.csv"
    )

    print(results_df)
    print("\nSaved summary to G_results/G_summary.csv")