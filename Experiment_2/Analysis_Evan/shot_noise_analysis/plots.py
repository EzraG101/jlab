import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from .calibration import power_law
from .constants import ACCEPTED_E_C
from .stats_utils import linear_model


def apply_style():
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 4.6),
            "font.size": 13,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.grid": False,
        }
    )


def save_both(fig, path_without_suffix):
    fig.tight_layout()
    fig.savefig(path_without_suffix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_gain_curve(cal_rows, gain_result, out_path):
    apply_style()
    f_khz = np.asarray([row["f_khz"] for row in cal_rows], dtype=float)
    f_hz = f_khz * 1e3
    gain2 = np.asarray([row["gain2_mean"] for row in cal_rows], dtype=float)
    gain2_err = np.asarray([row["gain2_sem"] for row in cal_rows], dtype=float)
    tail = gain_result["tail"]
    tail_start_khz = f_khz[tail["mask"]][0]
    nominal_tail = gain_result["nominal_tail"]

    fig, ax = plt.subplots()
    ax.errorbar(f_khz, gain2, yerr=gain2_err, fmt="o", ms=4.5, capsize=0, label="Calibration means")
    if nominal_tail["model"] == "linear_zero":
        last_f = max(f_hz)
        f_zero = nominal_tail["f_zero_hz"]
        dense_hz = np.linspace(last_f, f_zero, 80)
        dense_gain = nominal_tail["slope_gain2_per_hz"] * dense_hz + nominal_tail["intercept_gain2"]
        ax.plot(dense_hz / 1e3, dense_gain, color="tab:red", lw=2, label="linear tail to zero")
    else:
        dense_hz = np.linspace(tail_start_khz * 1e3, max(f_hz) * 1.08, 400)
        ax.plot(
            dense_hz / 1e3,
            power_law(dense_hz, tail["amplitude"], tail["exponent"]),
            color="tab:red",
            lw=2,
            label=rf"tail fit, $p={tail['exponent']:.2f}$",
        )
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel(r"Gain squared, $g^2(f)$")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(bottom=0)
    save_both(fig, out_path)


def plot_shot_fit(shot_rows, fit_result, out_path, label="independent repeats"):
    apply_style()
    arrays = fit_result["arrays"]
    nominal = fit_result["nominal_fit"]
    x = arrays["scaled_x"]
    y = arrays["v0_sq_v2"]
    xerr = arrays["scaled_x_err"]
    yerr = arrays["v0_sq_v2_err"]

    fig, ax = plt.subplots()
    ax.errorbar(x / 1e17, y / 1e-3, xerr=xerr / 1e17, yerr=yerr / 1e-3, fmt="o", capsize=0, ms=4.5, label="Grouped means")
    dense = np.linspace(0, max(x) * 1.03, 300)
    ax.plot(dense / 1e17, linear_model(dense, nominal["slope"], nominal["intercept"]) / 1e-3, color="tab:red", lw=2, label="weighted linear fit")
    ax.set_xlabel(r"$2R_F^2 I_{\rm av}\int g^2(f)\,df$ ($10^{17}$ V$^2$/C)")
    ax.set_ylabel(r"$V_0^2$ ($10^{-3}$ V$^2$)")
    e_text = (
        label
        + "\n"
        rf"$e = ({fit_result['e_C'] / 1e-19:.3f} \pm {fit_result['e_total_err_C'] / 1e-19:.3f})"
        r"\times10^{-19}\ \mathrm{C}$"
        "\n"
        rf"$\chi^2/\nu={nominal['chi2']:.1f}/{nominal['dof']}$, $p={nominal['p_value']:.1e}$"
        "\n"
        rf"$R^2={nominal['r_squared']:.4f}$"
        "\n"
        rf"$e/e_{{\rm acc}}={fit_result['accepted_ratio']:.3f}$"
    )
    ax.text(
        0.04,
        0.94,
        e_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
    )
    ax.legend(loc="lower right", frameon=False)
    save_both(fig, out_path)


def plot_shot_fit_scatter_adjusted(shot_rows, fit_result, out_path):
    apply_style()
    arrays = fit_result["arrays"]
    adjusted = fit_result["scatter_adjusted_fit"]
    x = arrays["scaled_x"]
    y = arrays["v0_sq_v2"]
    xerr = arrays["scaled_x_err"]
    yerr = np.sqrt(arrays["v0_sq_v2_err"] ** 2 + adjusted["extra_y_scatter"] ** 2)

    fig, ax = plt.subplots()
    ax.errorbar(
        x / 1e17,
        y / 1e-3,
        xerr=xerr / 1e17,
        yerr=yerr / 1e-3,
        fmt="o",
        capsize=0,
        ms=4.5,
        label="Grouped means, scatter-adjusted errors",
    )
    dense = np.linspace(0, max(x) * 1.03, 300)
    ax.plot(
        dense / 1e17,
        linear_model(dense, adjusted["slope"], adjusted["intercept"]) / 1e-3,
        color="tab:red",
        lw=2,
        label="weighted linear fit",
    )
    ax.set_xlabel(r"$2R_F^2 I_{\rm av}\int g^2(f)\,df$ ($10^{17}$ V$^2$/C)")
    ax.set_ylabel(r"$V_0^2$ ($10^{-3}$ V$^2$)")
    e_text = (
        rf"$e = ({fit_result['e_external_scatter_C'] / 1e-19:.3f} \pm "
        rf"{fit_result['e_external_scatter_err_C'] / 1e-19:.3f})"
        r"\times10^{-19}\ \mathrm{C}$"
        "\n"
        rf"$\chi^2/\nu={adjusted['chi2']:.1f}/{adjusted['dof']}$, $p={adjusted['p_value']:.2f}$"
        "\n"
        rf"extra scatter $={adjusted['extra_y_scatter'] / 1e-6:.1f}\times10^{{-6}}\ \mathrm{{V}}^2$"
    )
    ax.text(
        0.04,
        0.94,
        e_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
    )
    ax.legend(loc="lower right", frameon=False)
    save_both(fig, out_path)


def plot_shot_fit_front_cut(shot_rows, full_fit_result, cut_fit_result, included_mask, out_path):
    apply_style()
    arrays = full_fit_result["arrays"]
    nominal = cut_fit_result["nominal_fit"]
    x = arrays["scaled_x"]
    y = arrays["v0_sq_v2"]
    xerr = arrays["scaled_x_err"]
    yerr = arrays["v0_sq_v2_err"]
    included_mask = np.asarray(included_mask, dtype=bool)

    fig, ax = plt.subplots()
    ax.errorbar(
        x[~included_mask] / 1e17,
        y[~included_mask] / 1e-3,
        xerr=xerr[~included_mask] / 1e17,
        yerr=yerr[~included_mask] / 1e-3,
        fmt="o",
        capsize=0,
        ms=4.0,
        color="0.65",
        label="Excluded low-signal points",
    )
    ax.errorbar(
        x[included_mask] / 1e17,
        y[included_mask] / 1e-3,
        xerr=xerr[included_mask] / 1e17,
        yerr=yerr[included_mask] / 1e-3,
        fmt="o",
        capsize=0,
        ms=4.8,
        label="Fit points",
    )
    dense = np.linspace(0, max(x) * 1.03, 300)
    ax.plot(
        dense / 1e17,
        linear_model(dense, nominal["slope"], nominal["intercept"]) / 1e-3,
        color="tab:red",
        lw=2,
        label="weighted linear fit",
    )
    ax.set_xlabel(r"$2R_F^2 I_{\rm av}\int g^2(f)\,df$ ($10^{17}$ V$^2$/C)")
    ax.set_ylabel(r"$V_0^2$ ($10^{-3}$ V$^2$)")
    e_text = (
        rf"$e = ({cut_fit_result['e_C'] / 1e-19:.3f} \pm {cut_fit_result['e_total_err_C'] / 1e-19:.3f})"
        r"\times10^{-19}\ \mathrm{C}$"
        "\n"
        rf"$\chi^2/\nu={nominal['chi2']:.1f}/{nominal['dof']}$, $p={nominal['p_value']:.2f}$"
        "\n"
        rf"$e/e_{{\rm acc}}={cut_fit_result['accepted_ratio']:.3f}$"
    )
    ax.text(
        0.04,
        0.94,
        e_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
    )
    ax.legend(loc="lower right", frameon=False)
    save_both(fig, out_path)


def plot_residuals(shot_rows, fit_result, out_path):
    apply_style()
    arrays = fit_result["arrays"]
    nominal = fit_result["nominal_fit"]
    vd_mv = np.asarray([row["vd_mv_mean"] for row in shot_rows], dtype=float)

    fig, ax = plt.subplots()
    ax.axhline(0.0, color="0.35", lw=1.2)
    ax.axhline(2.0, color="0.55", lw=1.0, ls="--")
    ax.axhline(-2.0, color="0.55", lw=1.0, ls="--")
    ax.errorbar(
        vd_mv,
        nominal["standardized_residuals"],
        yerr=np.ones_like(vd_mv),
        fmt="o",
        capsize=0,
        ms=4.5,
    )
    ax.set_xlabel(r"First-stage DC voltage, $V_d$ (mV)")
    ax.set_ylabel("Standardized residual")
    ax.text(
        0.04,
        0.94,
        rf"accepted $e={ACCEPTED_E_C / 1e-19:.3f}\times10^{{-19}}$ C",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
    )
    save_both(fig, out_path)
