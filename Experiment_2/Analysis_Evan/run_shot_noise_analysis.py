import csv
import json
import math
import os
import statistics
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

from shot_noise_analysis.calibration import integrate_gain
from shot_noise_analysis.constants import (
    ACCEPTED_E_C,
    CALIBRATION_CSV,
    FIGURE_DIR,
    OUTPUT_DIR,
    RESULT_DIR,
    RF_ERR_OHM,
    RF_OHM,
    RTEST_ERR_OHM,
    RTEST_OHM,
    SHOT_NOISE_CSV,
    TABLE_DIR,
    TAIL_START_HZ,
    V0_RMS_FLOOR_MV,
)
from shot_noise_analysis.data_io import load_calibration_summary, load_shot_noise_summary, write_csv
from shot_noise_analysis.plots import (
    plot_gain_curve,
    plot_residuals,
    plot_shot_fit,
    plot_shot_fit_front_cut,
    plot_shot_fit_scatter_adjusted,
)
from shot_noise_analysis.shot_noise import fit_electron_charge, leave_one_out_sensitivity


def _json_clean(obj):
    if isinstance(obj, dict):
        return {key: _json_clean(value) for key, value in obj.items() if key not in {"cov", "mask", "sigma", "residuals", "standardized_residuals", "arrays"}}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _shot_table_rows(shot_rows, fit_result):
    arrays = fit_result["arrays"]
    residuals = fit_result["nominal_fit"]["residuals"]
    std_resid = fit_result["nominal_fit"]["standardized_residuals"]
    rows = []
    for i, row in enumerate(shot_rows):
        rows.append(
            {
                **row,
                "current_a": arrays["current_a"][i],
                "current_a_err_stat": arrays["current_a_err_stat"][i],
                "current_a_err_rf": arrays["current_a_err_rf"][i],
                "current_a_err": arrays["current_a_err"][i],
                "scaled_x_v2_per_c": arrays["scaled_x"][i],
                "scaled_x_err_v2_per_c": arrays["scaled_x_err"][i],
                "scaled_x_rf_err_v2_per_c": arrays["scaled_x_rf_err"][i],
                "v0_sq_v2": arrays["v0_sq_v2"][i],
                "v0_sq_repeat_err_v2": arrays["v0_sq_repeat_err_v2"][i],
                "v0_sq_floor_err_v2": arrays["v0_sq_floor_err_v2"][i],
                "v0_sq_v2_err": arrays["v0_sq_v2_err"][i],
                "fit_residual_v2": residuals[i],
                "fit_standardized_residual": std_resid[i],
            }
        )
    return rows


def _sensitivity_rows(cal_rows, gain_result, shot_rows):
    f_hz = np.asarray([row["f_hz"] for row in cal_rows], dtype=float)
    gain2 = np.asarray([row["gain2_mean"] for row in cal_rows], dtype=float)
    gain2_err = np.asarray([row["gain2_sem"] for row in cal_rows], dtype=float)
    rows = []
    for start_khz in [45.0, 50.0, 55.0, 60.0, 65.0]:
        try:
            trial_gain = integrate_gain(f_hz, gain2, gain2_err, start_khz * 1e3, n_mc=2000)
            trial_fit = fit_electron_charge(shot_rows, trial_gain["G"], trial_gain["G_err_total"])
            rows.append(
                {
                    "tail_start_khz": start_khz,
                    "G": trial_gain["G"],
                    "G_err_stat": trial_gain["G_err_stat"],
                    "tail_percent": 100.0 * trial_gain["G_tail"] / trial_gain["G"],
                    "e_C": trial_fit["e_C"],
                    "e_total_err_C": trial_fit["e_total_err_C"],
                    "chi2_red": trial_fit["nominal_fit"]["chi2_red"],
                    "p_value": trial_fit["nominal_fit"]["p_value"],
                }
            )
        except Exception as exc:
            rows.append({"tail_start_khz": start_khz, "error": str(exc)})
    return rows


def _current_range_sensitivity_rows(shot_rows, gain_result):
    windows_mv = [
        (0.0, math.inf),
        (20.0, math.inf),
        (50.0, math.inf),
        (100.0, math.inf),
        (200.0, math.inf),
        (500.0, math.inf),
        (0.0, 5000.0),
        (100.0, 5000.0),
        (200.0, 5000.0),
        (500.0, 5000.0),
    ]
    rows = []
    for low_mv, high_mv in windows_mv:
        kept = [
            row
            for row in shot_rows
            if row["vd_mv_mean"] >= low_mv and row["vd_mv_mean"] <= high_mv
        ]
        if len(kept) < 5:
            continue
        fit = fit_electron_charge(kept, gain_result["G"], gain_result["G_err_total"])
        rows.append(
            {
                "vd_min_mv": low_mv,
                "vd_max_mv": high_mv,
                "n_points": len(kept),
                "e_C": fit["e_C"],
                "e_total_err_C": fit["e_total_err_C"],
                "accepted_ratio": fit["accepted_ratio"],
                "chi2_red": fit["nominal_fit"]["chi2_red"],
                "p_value": fit["nominal_fit"]["p_value"],
            }
        )
    return rows


def _scaled_x_threshold_sensitivity_rows(shot_rows, gain_result):
    full_fit = fit_electron_charge(shot_rows, gain_result["G"], gain_result["G_err_total"], v0_rms_floor_mv=0.0)
    scaled_x_display = full_fit["arrays"]["scaled_x"] / 1e17
    rows = []
    for xmin in [0.0, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0]:
        kept = [
            row
            for row, x_display in zip(shot_rows, scaled_x_display)
            if x_display >= xmin
        ]
        if len(kept) < 5:
            continue
        fit = fit_electron_charge(kept, gain_result["G"], gain_result["G_err_total"], v0_rms_floor_mv=0.0)
        rows.append(
            {
                "scaled_x_min_1e17_v2_per_c": xmin,
                "n_points": len(kept),
                "e_C": fit["e_C"],
                "e_total_err_C": fit["e_total_err_C"],
                "accepted_ratio": fit["accepted_ratio"],
                "chi2_red": fit["nominal_fit"]["chi2_red"],
                "p_value": fit["nominal_fit"]["p_value"],
            }
        )
    return rows


def _effective_sample_size_rows(shot_rows, gain_result):
    rows = []
    for n_eff in [1, 2, 3, 4, 5]:
        adjusted_rows = []
        for row in shot_rows:
            adjusted = dict(row)
            effective_n = min(float(n_eff), float(row["n"]))
            adjusted["vd_mv_sem"] = row["vd_mv_std"] / math.sqrt(effective_n)
            adjusted["v0_sq_mv2_sem"] = row["v0_sq_mv2_std"] / math.sqrt(effective_n)
            adjusted_rows.append(adjusted)
        fit = fit_electron_charge(
            adjusted_rows,
            gain_result["G"],
            gain_result["G_err_total"],
            v0_rms_floor_mv=0.0,
        )
        rows.append(
            {
                "n_eff": n_eff,
                "n_points": len(adjusted_rows),
                "e_C": fit["e_C"],
                "e_total_err_C": fit["e_total_err_C"],
                "accepted_ratio": fit["accepted_ratio"],
                "chi2": fit["nominal_fit"]["chi2"],
                "dof": fit["nominal_fit"]["dof"],
                "chi2_red": fit["nominal_fit"]["chi2_red"],
                "p_value": fit["nominal_fit"]["p_value"],
            }
        )
    return rows


def _choose_front_cut_fit(shot_rows, gain_result):
    candidates = _scaled_x_threshold_sensitivity_rows(shot_rows, gain_result)
    acceptable = [row for row in candidates if row["p_value"] >= 0.05]
    selected = acceptable[0] if acceptable else candidates[0]
    full_no_floor = fit_electron_charge(shot_rows, gain_result["G"], gain_result["G_err_total"], v0_rms_floor_mv=0.0)
    scaled_x_display = full_no_floor["arrays"]["scaled_x"] / 1e17
    included_mask = scaled_x_display >= selected["scaled_x_min_1e17_v2_per_c"]
    kept = [row for row, include in zip(shot_rows, included_mask) if include]
    fit = fit_electron_charge(kept, gain_result["G"], gain_result["G_err_total"], v0_rms_floor_mv=0.0)
    return {
        "threshold": selected,
        "fit": fit,
        "included_mask": included_mask,
        "included_bulb_settings": [row["bulb_setting"] for row, include in zip(shot_rows, included_mask) if include],
        "excluded_bulb_settings": [row["bulb_setting"] for row, include in zip(shot_rows, included_mask) if not include],
        "note": "Smallest plotted-x threshold with p >= 0.05 using repeat SEM only and no empirical RMS floor.",
    }


def _raw_repeated_row_diagnostic(path, gain_result):
    raw_rows = []
    current_setting = None
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row["bulb setting"]).strip():
                current_setting = int(float(row["bulb setting"]))
            vd_text = str(row["Vd (mV) (dc, first stage)"]).strip()
            v0_text = str(row["V0 (mV) (ac, second stage"]).strip()
            v2_text = str(row["V0^2"]).strip()
            if not vd_text or not v0_text or not v2_text:
                continue
            vd_mv = float(vd_text)
            v0_mv = float(v0_text)
            v2_mv2 = float(v2_text)
            if v2_mv2 <= 0:
                continue
            raw_rows.append(
                {
                    "bulb_setting": current_setting,
                    "vd_mv_mean": vd_mv,
                    "vd_mv_sem": 0.0,
                    "v0_mv_mean": v0_mv,
                    "v0_sq_mv2_mean": v2_mv2,
                    "v0_sq_mv2_sem": 0.0,
                    "n": 1,
                }
            )

    by_setting = {}
    for row in raw_rows:
        by_setting.setdefault(row["bulb_setting"], []).append(row["v0_sq_mv2_mean"])
    within_stds = [statistics.stdev(values) for values in by_setting.values() if len(values) > 1]
    pooled_single_measurement_std_mv2 = math.sqrt(
        sum(value**2 for value in within_stds) / len(within_stds)
    )
    for row in raw_rows:
        row["v0_sq_mv2_sem"] = pooled_single_measurement_std_mv2

    fit = fit_electron_charge(
        sorted(raw_rows, key=lambda row: row["vd_mv_mean"]),
        gain_result["G"],
        gain_result["G_err_total"],
        v0_rms_floor_mv=0.0,
    )
    return {
        "n_rows": len(raw_rows),
        "pooled_single_measurement_std_mv2": pooled_single_measurement_std_mv2,
        "e_C": fit["e_C"],
        "e_total_err_C": fit["e_total_err_C"],
        "accepted_ratio": fit["accepted_ratio"],
        "chi2": fit["nominal_fit"]["chi2"],
        "dof": fit["nominal_fit"]["dof"],
        "chi2_red": fit["nominal_fit"]["chi2_red"],
        "p_value": fit["nominal_fit"]["p_value"],
        "note": "Diagnostic only: repeated rows from the same bulb setting are not independent current settings.",
    }


def _discrepancy_diagnostics(fit_result, gain_result):
    ratio = fit_result["accepted_ratio"]
    delta_e = fit_result["e_C"] - ACCEPTED_E_C
    max_x = float(np.max(fit_result["arrays"]["scaled_x"]))
    max_extra_v2 = delta_e * max_x
    return {
        "measured_to_accepted_ratio": ratio,
        "excess_percent": 100.0 * (ratio - 1.0),
        "required_rf_ohm_if_rtest_and_gain_correct": RF_OHM / ratio,
        "required_rtest_ohm_if_rf_and_gain_correct": RTEST_OHM * math.sqrt(ratio),
        "required_gain_integral_hz_if_rf_correct": gain_result["G"] * ratio,
        "required_gain_amplitude_scale": math.sqrt(ratio),
        "required_gain_amplitude_percent": 100.0 * (math.sqrt(ratio) - 1.0),
        "required_first_stage_dc_scale_if_rf_and_gain_correct": 1.0 / ratio,
        "effective_fano_or_excess_noise_factor": ratio,
        "max_output_excess_v2": max_extra_v2,
        "max_output_excess_rms_mv": 1e3 * math.sqrt(max_extra_v2) if max_extra_v2 > 0 else math.nan,
        "notes": [
            "A constant amplifier or Johnson-noise background changes the intercept, not the slope.",
            "A dark-current or DC-offset error in Vd also mostly changes the intercept if it is additive.",
            "The observed charge offset requires a multiplicative scale error or a current-proportional excess-noise source.",
        ],
    }


def _write_methods(path, result):
    nominal = result["fit"]["nominal_fit"]
    adjusted = result["fit"]["scatter_adjusted_fit"]
    gain = result["gain"]
    n_eff_2 = next(row for row in result["effective_sample_size_sensitivity"] if row["n_eff"] == 2)
    n_eff_3 = next(row for row in result["effective_sample_size_sensitivity"] if row["n_eff"] == 3)
    lines = [
        "# Shot Noise Analysis Methods",
        "",
        "## Raw data handling",
        "- The calibration and shot-noise CSV files in `shot_noise_data/` are treated as raw inputs.",
        "- Calibration repeats are grouped by exact frequency.",
        "- Shot-noise repeats are grouped by bulb setting, blank continuation rows are carried forward, and blank or zero trailing rows are ignored.",
        "- Shot-noise groups are sorted by measured first-stage voltage `V_d`, not knob setting.",
        f"- The fit error bars include repeat SEM and an RMS-voltage floor of `{V0_RMS_FLOOR_MV:.2f} mV` propagated as `delta(V0^2)=2 V0 delta(V0)`.",
        "",
        "## Unit conversions",
        "- Frequencies are converted from kHz to Hz before integration.",
        "- Voltages are converted from mV to V before fitting.",
        "- The feedback resistor and test-input resistor are each taken as `475 kOhm` with independent 1% tolerances.",
        "- Current is computed as `I = V_d / R_F`.",
        "- The calibration test input means the fitted x scale is proportional to `R_test^2 / R_F`, so resistor tolerance contributes `sqrt((2%)^2 + (1%)^2) = 2.24%` as a common scale uncertainty.",
        "",
        "## Gain integral",
        "- The measured gain-squared curve is integrated with the trapezoid rule through the final measured point at 90 kHz.",
        "- Gain squared is computed as `(mean V0 / mean Vi)^2` at each frequency; the mean of per-row gain-squared values is retained in the calibration summary for comparison.",
        "- The unmeasured right tail beyond 90 kHz uses a power-law fit to the measured high-frequency tail.",
        f"- Nominal result: `G = {gain['G']:.6e} +/- {gain['G_err_total']:.2e} Hz`.",
        f"- The fitted tail contributes `{100 * gain['G_tail'] / gain['G']:.3f}%` of the nominal integral.",
        f"- Trapezoid discretization is estimated by comparing to Simpson integration: `{100 * gain['G_err_discretization'] / gain['G']:.3f}%`.",
        "",
        "## Shot-noise fit",
        "- The fitted line is `V0^2 = e X + V_A^2`, where `X = 2 R_F^2 I_av integral(g^2(f) df)`. The plot axis multiplies by the gain integral; it does not divide by it.",
        "- The slope is the electron charge in coulombs.",
        "- The nominal fit uses one grouped mean per bulb setting. This is the correct default because repeated readings at the same bulb setting are repeated observations of the same current condition.",
        "- Treating all raw rows separately is included only as a diagnostic; it gives a much better-looking chi-squared when a pooled single-measurement scatter is assigned, but it does not move the charge toward the accepted value.",
        "- Repeated-measurement SEM is used for `V0^2`; uncertainty in `V_d` is propagated into the point-by-point `X` uncertainty.",
        "- The five readings within a bulb setting may be time-correlated because the AC voltmeter reports an RMS average with finite response time. As a diagnostic, the group uncertainty is recomputed as `s/sqrt(N_eff)` for `N_eff = 1...5` instead of assuming all five readings are independent.",
        "- The RMS-voltage floor is an alternate empirical way to account for AC-voltmeter/noise-estimator stability not captured by five repeated readings. The effective-sample-size scan is preferred for presentation because it directly connects the enlarged error bars to correlated RMS readings.",
        "- The gain-integral uncertainty is propagated as a separate multiplicative contribution to the slope uncertainty.",
        f"- The resistor-ratio contribution to the all-data uncertainty is `{result['fit']['e_resistor_scale_err_C'] / 1e-19:.4f}e-19 C`.",
        "- A second fit excludes low-signal front-end points. The threshold is chosen as the smallest plotted x-value cut that gives `p >= 0.05` using repeat SEM only, without the empirical RMS-voltage floor.",
        "",
        "## Diagnostics and improvement",
        f"- Nominal fit with RMS floor: `chi2/dof = {nominal['chi2']:.3g}/{nominal['dof']}`, `chi2_red = {nominal['chi2_red']:.3g}`, `p = {nominal['p_value']:.3g}`.",
        f"- Nominal grouped result: `e = ({result['fit']['e_C'] / 1e-19:.4f} +/- {result['fit']['e_total_err_C'] / 1e-19:.4f})e-19 C`.",
        f"- Front-end-cut result: `e = ({result['front_cut_fit']['fit']['e_C'] / 1e-19:.4f} +/- {result['front_cut_fit']['fit']['e_total_err_C'] / 1e-19:.4f})e-19 C`, `chi2_red = {result['front_cut_fit']['fit']['nominal_fit']['chi2_red']:.3g}`, `p = {result['front_cut_fit']['fit']['nominal_fit']['p_value']:.3g}`.",
        f"- The front-end cut excludes bulb settings `{result['front_cut_fit']['excluded_bulb_settings']}`.",
        f"- Accepted value ratio: `e/e_accepted = {result['fit']['accepted_ratio']:.4f}`.",
        f"- Agreement with the accepted value would require the product `(R_test^2/R_F) G` to be larger by `{result['discrepancy_diagnostics']['excess_percent']:.1f}%`, much larger than the propagated calibration-repeat and resistor-ratio uncertainties.",
        f"- If the issue is only AC calibration, the gain amplitude would need to be low by `{result['discrepancy_diagnostics']['required_gain_amplitude_percent']:.1f}%`.",
        f"- If interpreted as a current-proportional excess-noise source, the effective Fano/excess-noise factor is `{result['discrepancy_diagnostics']['effective_fano_or_excess_noise_factor']:.3f}`.",
        f"- The post-fit extra-scatter check now adds `{adjusted.get('extra_y_scatter', 0.0):.3e} V^2`; this is small compared with the output variance scale but shows the RMS floor is still an approximation.",
        f"- Raw-row diagnostic: `e = ({result['raw_row_diagnostic']['e_C'] / 1e-19:.4f} +/- {result['raw_row_diagnostic']['e_total_err_C'] / 1e-19:.4f})e-19 C`, `chi2_red = {result['raw_row_diagnostic']['chi2_red']:.3g}`.",
        f"- Effective-sample-size diagnostic with `N_eff = 2`: `e = ({n_eff_2['e_C'] / 1e-19:.4f} +/- {n_eff_2['e_total_err_C'] / 1e-19:.4f})e-19 C`, `chi2/dof = {n_eff_2['chi2']:.3g}/{n_eff_2['dof']}`, `chi2_red = {n_eff_2['chi2_red']:.3g}`, `p = {n_eff_2['p_value']:.3g}`.",
        f"- Effective-sample-size diagnostic with `N_eff = 3`: `e = ({n_eff_3['e_C'] / 1e-19:.4f} +/- {n_eff_3['e_total_err_C'] / 1e-19:.4f})e-19 C`, `chi2/dof = {n_eff_3['chi2']:.3g}/{n_eff_3['dof']}`, `chi2_red = {n_eff_3['chi2_red']:.3g}`, `p = {n_eff_3['p_value']:.3g}`.",
        f"- Removing points with plotted x-value below 0.5 leaves `{next(row for row in result['scaled_x_threshold_sensitivity'] if row['scaled_x_min_1e17_v2_per_c'] == 0.5)['n_points']}` points and gives `e = ({next(row for row in result['scaled_x_threshold_sensitivity'] if row['scaled_x_min_1e17_v2_per_c'] == 0.5)['e_C'] / 1e-19:.4f} +/- {next(row for row in result['scaled_x_threshold_sensitivity'] if row['scaled_x_min_1e17_v2_per_c'] == 0.5)['e_total_err_C'] / 1e-19:.4f})e-19 C`, `chi2_red = {next(row for row in result['scaled_x_threshold_sensitivity'] if row['scaled_x_min_1e17_v2_per_c'] == 0.5)['chi2_red']:.3g}`.",
        "- Leave-one-out, current-window, scaled-x-threshold, effective-sample-size, and tail-start scans are exported in `shot_noise_results.json`; these are diagnostics, not hidden point-selection rules.",
        "",
        "## Presentation guidance",
        "- Use the nominal fit figure to show the main result and the residual plot to discuss remaining structure.",
        "- For the goodness-of-fit discussion, quote the `N_eff = 2` or `N_eff = 3` uncertainty model as a correlated-readings systematic rather than relying only on the empirical RMS floor.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    cal_rows = load_calibration_summary(CALIBRATION_CSV)
    shot_rows = load_shot_noise_summary(SHOT_NOISE_CSV)

    f_hz = np.asarray([row["f_hz"] for row in cal_rows], dtype=float)
    gain2 = np.asarray([row["gain2_mean"] for row in cal_rows], dtype=float)
    gain2_err = np.asarray([row["gain2_sem"] for row in cal_rows], dtype=float)

    gain_result = integrate_gain(f_hz, gain2, gain2_err, TAIL_START_HZ, n_mc=20000)
    fit_result = fit_electron_charge(shot_rows, gain_result["G"], gain_result["G_err_total"], rf_ohm=RF_OHM, rf_err_ohm=RF_ERR_OHM)
    discrepancy_diagnostics = _discrepancy_diagnostics(fit_result, gain_result)
    sensitivity = _sensitivity_rows(cal_rows, gain_result, shot_rows)
    leave_one_out = leave_one_out_sensitivity(shot_rows, gain_result["G"], gain_result["G_err_total"])
    current_window_sensitivity = _current_range_sensitivity_rows(shot_rows, gain_result)
    scaled_x_threshold_sensitivity = _scaled_x_threshold_sensitivity_rows(shot_rows, gain_result)
    effective_sample_size_sensitivity = _effective_sample_size_rows(shot_rows, gain_result)
    front_cut_fit = _choose_front_cut_fit(shot_rows, gain_result)
    raw_row_diagnostic = _raw_repeated_row_diagnostic(SHOT_NOISE_CSV, gain_result)

    write_csv(TABLE_DIR / "calibration_summary.csv", cal_rows)
    write_csv(TABLE_DIR / "shot_noise_summary.csv", _shot_table_rows(shot_rows, fit_result))
    write_csv(TABLE_DIR / "tail_start_sensitivity.csv", sensitivity)
    write_csv(TABLE_DIR / "leave_one_out_sensitivity.csv", leave_one_out)
    write_csv(TABLE_DIR / "current_window_sensitivity.csv", current_window_sensitivity)
    write_csv(TABLE_DIR / "scaled_x_threshold_sensitivity.csv", scaled_x_threshold_sensitivity)
    write_csv(TABLE_DIR / "effective_sample_size_sensitivity.csv", effective_sample_size_sensitivity)
    write_csv(
        TABLE_DIR / "front_cut_fit_summary.csv",
        [
            {
                "scaled_x_min_1e17_v2_per_c": front_cut_fit["threshold"]["scaled_x_min_1e17_v2_per_c"],
                "n_points": front_cut_fit["threshold"]["n_points"],
                "e_C": front_cut_fit["fit"]["e_C"],
                "e_total_err_C": front_cut_fit["fit"]["e_total_err_C"],
                "e_stat_err_C": front_cut_fit["fit"]["e_stat_err_C"],
                "e_gain_err_C": front_cut_fit["fit"]["e_gain_err_C"],
                "e_rf_err_C": front_cut_fit["fit"]["e_rf_err_C"],
                "accepted_ratio": front_cut_fit["fit"]["accepted_ratio"],
                "chi2_red": front_cut_fit["fit"]["nominal_fit"]["chi2_red"],
                "p_value": front_cut_fit["fit"]["nominal_fit"]["p_value"],
                "excluded_bulb_settings": " ".join(str(item) for item in front_cut_fit["excluded_bulb_settings"]),
            }
        ],
    )
    write_csv(
        TABLE_DIR / "discrepancy_diagnostics.csv",
        [
            {
                key: value
                for key, value in discrepancy_diagnostics.items()
                if key != "notes"
            }
        ],
    )

    full_no_floor_fit = fit_electron_charge(shot_rows, gain_result["G"], gain_result["G_err_total"], v0_rms_floor_mv=0.0)
    plot_gain_curve(cal_rows, gain_result, FIGURE_DIR / "gain_curve")
    plot_shot_fit(shot_rows, full_no_floor_fit, FIGURE_DIR / "shot_noise_fit")
    plot_shot_fit_front_cut(
        shot_rows,
        full_no_floor_fit,
        front_cut_fit["fit"],
        front_cut_fit["included_mask"],
        FIGURE_DIR / "shot_noise_fit_front_cut",
    )
    plot_shot_fit_scatter_adjusted(shot_rows, fit_result, FIGURE_DIR / "shot_noise_fit_scatter_adjusted")
    plot_residuals(shot_rows, fit_result, FIGURE_DIR / "shot_noise_residuals")

    result = {
        "rf_ohm": RF_OHM,
        "rf_err_ohm": RF_ERR_OHM,
        "rtest_ohm": RTEST_OHM,
        "rtest_err_ohm": RTEST_ERR_OHM,
        "accepted_e_C": ACCEPTED_E_C,
        "gain": gain_result,
        "fit": fit_result,
        "independent_repeats_fit": full_no_floor_fit,
        "discrepancy_diagnostics": discrepancy_diagnostics,
        "tail_start_sensitivity": sensitivity,
        "leave_one_out_sensitivity": leave_one_out,
        "current_window_sensitivity": current_window_sensitivity,
        "scaled_x_threshold_sensitivity": scaled_x_threshold_sensitivity,
        "effective_sample_size_sensitivity": effective_sample_size_sensitivity,
        "front_cut_fit": front_cut_fit,
        "raw_row_diagnostic": raw_row_diagnostic,
    }

    with (RESULT_DIR / "shot_noise_results.json").open("w") as handle:
        json.dump(_json_clean(result), handle, indent=2)

    _write_methods(OUTPUT_DIR / "analysis_methods.md", result)

    print(f"G = {gain_result['G']:.6e} +/- {gain_result['G_err_total']:.2e} Hz")
    print(
        "e = "
        f"({fit_result['e_C'] / 1e-19:.4f} +/- {fit_result['e_total_err_C'] / 1e-19:.4f})e-19 C"
    )
    print(f"accepted e ratio = {fit_result['accepted_ratio']:.4f}")
    print(
        "fit chi2/dof = "
        f"{fit_result['nominal_fit']['chi2']:.3f}/{fit_result['nominal_fit']['dof']} "
        f"(p={fit_result['nominal_fit']['p_value']:.3g})"
    )
    print(
        "external-scatter e = "
        f"({fit_result['e_external_scatter_C'] / 1e-19:.4f} +/- "
        f"{fit_result['e_external_scatter_err_C'] / 1e-19:.4f})e-19 C"
    )
    print(
        "raw-row diagnostic e = "
        f"({raw_row_diagnostic['e_C'] / 1e-19:.4f} +/- "
        f"{raw_row_diagnostic['e_total_err_C'] / 1e-19:.4f})e-19 C, "
        f"chi2_red={raw_row_diagnostic['chi2_red']:.3g}"
    )


if __name__ == "__main__":
    main()
