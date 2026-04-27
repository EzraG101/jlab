import math

import numpy as np

from .constants import ACCEPTED_E_C, RF_ERR_OHM, RF_OHM, RTEST_ERR_OHM, RTEST_OHM, V0_RMS_FLOOR_MV
from .stats_utils import fit_with_extra_scatter, quadrature, weighted_linear_fit


def prepare_fit_arrays(
    shot_rows,
    rf_ohm=RF_OHM,
    rf_err_ohm=RF_ERR_OHM,
    rtest_ohm=RTEST_OHM,
    rtest_err_ohm=RTEST_ERR_OHM,
    gain_integral=1.0,
    v0_rms_floor_mv=V0_RMS_FLOOR_MV,
):
    vd_v = np.asarray([row["vd_mv_mean"] for row in shot_rows], dtype=float) * 1e-3
    vd_v_err = np.asarray([row["vd_mv_sem"] for row in shot_rows], dtype=float) * 1e-3
    v0_mv = np.asarray([row["v0_mv_mean"] for row in shot_rows], dtype=float)
    v0_sq_v2 = np.asarray([row["v0_sq_mv2_mean"] for row in shot_rows], dtype=float) * 1e-6
    v0_sq_repeat_err_mv2 = np.asarray([row["v0_sq_mv2_sem"] for row in shot_rows], dtype=float)
    v0_sq_floor_err_mv2 = 2.0 * v0_mv * v0_rms_floor_mv
    v0_sq_err_mv2 = np.sqrt(v0_sq_repeat_err_mv2**2 + v0_sq_floor_err_mv2**2)
    v0_sq_v2_err = v0_sq_err_mv2 * 1e-6

    current_a = vd_v / rf_ohm
    current_a_err_stat = vd_v_err / rf_ohm
    current_a_err_rf = vd_v * rf_err_ohm / rf_ohm**2
    current_a_err = np.sqrt(current_a_err_stat**2 + current_a_err_rf**2)

    # The calibration signal is injected through a separate test-input resistor.
    # With measured g_cal = V0 / Vi, the physical AC gain is
    # a(f) = g_cal(f) R_test / R_F. Thus
    # X = 2 R_F^2 I ∫a^2 df = 2 Vd G_cal R_test^2 / R_F.
    # For nominal R_test = R_F = 475 kOhm this reduces to 2 R_F Vd G_cal.
    scaled_x = 2.0 * vd_v * gain_integral * rtest_ohm**2 / rf_ohm
    scaled_x_err = scaled_x * np.divide(vd_v_err, vd_v, out=np.zeros_like(vd_v_err), where=vd_v > 0)
    scaled_x_resistor_err = scaled_x * math.sqrt((2.0 * rtest_err_ohm / rtest_ohm) ** 2 + (rf_err_ohm / rf_ohm) ** 2)

    return {
        "vd_v": vd_v,
        "vd_v_err": vd_v_err,
        "current_a": current_a,
        "current_a_err_stat": current_a_err_stat,
        "current_a_err_rf": current_a_err_rf,
        "current_a_err": current_a_err,
        "v0_sq_v2": v0_sq_v2,
        "v0_sq_repeat_err_v2": v0_sq_repeat_err_mv2 * 1e-6,
        "v0_sq_floor_err_v2": v0_sq_floor_err_mv2 * 1e-6,
        "v0_sq_v2_err": np.maximum(v0_sq_v2_err, 1e-12),
        "scaled_x": scaled_x,
        "scaled_x_err": scaled_x_err,
        "scaled_x_resistor_err": scaled_x_resistor_err,
        "scaled_x_rf_err": scaled_x_resistor_err,
        "v0_rms_floor_mv": v0_rms_floor_mv,
    }


def fit_electron_charge(
    shot_rows,
    gain_integral,
    gain_integral_err,
    rf_ohm=RF_OHM,
    rf_err_ohm=RF_ERR_OHM,
    rtest_ohm=RTEST_OHM,
    rtest_err_ohm=RTEST_ERR_OHM,
    v0_rms_floor_mv=V0_RMS_FLOOR_MV,
):
    arrays = prepare_fit_arrays(
        shot_rows,
        rf_ohm=rf_ohm,
        rf_err_ohm=rf_err_ohm,
        rtest_ohm=rtest_ohm,
        rtest_err_ohm=rtest_err_ohm,
        gain_integral=gain_integral,
        v0_rms_floor_mv=v0_rms_floor_mv,
    )
    nominal = weighted_linear_fit(
        arrays["scaled_x"],
        arrays["v0_sq_v2"],
        arrays["v0_sq_v2_err"],
        xerr=arrays["scaled_x_err"],
    )
    scatter_adjusted = fit_with_extra_scatter(
        arrays["scaled_x"],
        arrays["v0_sq_v2"],
        arrays["v0_sq_v2_err"],
        xerr=arrays["scaled_x_err"],
    )

    e_stat = nominal["slope"]
    e_stat_err = nominal["slope_err"]
    e_gain_err = abs(e_stat) * gain_integral_err / gain_integral if gain_integral > 0 else math.nan
    resistor_rel_err = math.sqrt((2.0 * rtest_err_ohm / rtest_ohm) ** 2 + (rf_err_ohm / rf_ohm) ** 2)
    e_resistor_err = abs(e_stat) * resistor_rel_err
    e_total_err = quadrature(e_stat_err, e_gain_err, e_resistor_err)

    e_external = scatter_adjusted["slope"]
    e_external_err = scatter_adjusted["slope_err"]
    external_resistor_err = abs(e_external) * resistor_rel_err
    e_external_total_err = quadrature(
        e_external_err,
        abs(e_external) * gain_integral_err / gain_integral if gain_integral > 0 else math.nan,
        external_resistor_err,
    )

    return {
        "arrays": arrays,
        "nominal_fit": nominal,
        "scatter_adjusted_fit": scatter_adjusted,
        "e_C": float(e_stat),
        "e_stat_err_C": float(e_stat_err),
        "e_gain_err_C": float(e_gain_err),
        "e_resistor_scale_err_C": float(e_resistor_err),
        "e_rf_err_C": float(e_resistor_err),
        "resistor_scale_rel_err": float(resistor_rel_err),
        "e_total_err_C": float(e_total_err),
        "e_external_scatter_C": float(e_external),
        "e_external_scatter_err_C": float(e_external_total_err),
        "accepted_e_C": ACCEPTED_E_C,
        "accepted_ratio": float(e_stat / ACCEPTED_E_C),
        "accepted_difference_sigma": float((e_stat - ACCEPTED_E_C) / e_total_err) if e_total_err > 0 else math.nan,
    }


def leave_one_out_sensitivity(shot_rows, gain_integral, gain_integral_err):
    rows = []
    for idx, excluded in enumerate(shot_rows):
        kept = [row for j, row in enumerate(shot_rows) if j != idx]
        fit = fit_electron_charge(kept, gain_integral, gain_integral_err)
        rows.append(
            {
                "excluded_bulb_setting": excluded["bulb_setting"],
                "excluded_vd_mv_mean": excluded["vd_mv_mean"],
                "e_C": fit["e_C"],
                "e_total_err_C": fit["e_total_err_C"],
                "chi2_red": fit["nominal_fit"]["chi2_red"],
                "p_value": fit["nominal_fit"]["p_value"],
            }
        )
    return rows
