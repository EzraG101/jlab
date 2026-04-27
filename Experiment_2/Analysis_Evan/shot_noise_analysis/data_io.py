import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _float_or_nan(value):
    text = str(value).strip()
    if text == "":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def mean_sem(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan, math.nan, 0, math.nan
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, 0.0, 1, 0.0
    std = float(np.std(arr, ddof=1))
    return mean, std / math.sqrt(len(arr)), len(arr), std


def load_calibration_summary(path: Path):
    grouped = defaultdict(lambda: {"vi_mv": [], "vo_mv": [], "gain2": []})

    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            f_khz = _float_or_nan(row["f (khz)"])
            vi_mv = _float_or_nan(row["Vi (mV) (pre)"])
            vo_mv = _float_or_nan(row["Vo (mV) (post)"])
            gain2 = _float_or_nan(row["gain^2"])
            if not all(np.isfinite([f_khz, vi_mv, vo_mv, gain2])):
                continue
            grouped[f_khz]["vi_mv"].append(vi_mv)
            grouped[f_khz]["vo_mv"].append(vo_mv)
            grouped[f_khz]["gain2"].append(gain2)

    rows = []
    for f_khz in sorted(grouped):
        vi_mean, vi_sem, n, vi_std = mean_sem(grouped[f_khz]["vi_mv"])
        vo_mean, vo_sem, _, vo_std = mean_sem(grouped[f_khz]["vo_mv"])
        row_g_mean, row_g_sem, _, row_g_std = mean_sem(grouped[f_khz]["gain2"])
        ratio_mean_gain2 = (vo_mean / vi_mean) ** 2
        ratio_mean_gain2_sem = ratio_mean_gain2 * math.sqrt(
            (2.0 * vo_sem / vo_mean) ** 2 + (2.0 * vi_sem / vi_mean) ** 2
        )
        rows.append(
            {
                "f_khz": f_khz,
                "f_hz": f_khz * 1e3,
                "vi_mv_mean": vi_mean,
                "vi_mv_sem": vi_sem,
                "vi_mv_std": vi_std,
                "vo_mv_mean": vo_mean,
                "vo_mv_sem": vo_sem,
                "vo_mv_std": vo_std,
                "gain2_mean": ratio_mean_gain2,
                "gain2_sem": ratio_mean_gain2_sem,
                "gain2_std": math.nan,
                "row_gain2_mean": row_g_mean,
                "row_gain2_sem": row_g_sem,
                "row_gain2_std": row_g_std,
                "gain2_reduction": "ratio_of_mean_vo_to_mean_vi",
                "n": n,
            }
        )

    return rows


def load_shot_noise_summary(path: Path):
    grouped = []
    current_setting = None
    current_rows = []

    def flush():
        if current_setting is not None and current_rows:
            grouped.append((current_setting, list(current_rows)))

    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            setting_text = str(row["bulb setting"]).strip()
            if setting_text:
                flush()
                current_setting = int(float(setting_text))
                current_rows = []

            vd_mv = _float_or_nan(row["Vd (mV) (dc, first stage)"])
            v0_mv = _float_or_nan(row["V0 (mV) (ac, second stage"])
            v0_sq_mv2 = _float_or_nan(row["V0^2"])
            if not np.isfinite(vd_mv) or not np.isfinite(v0_sq_mv2):
                continue
            if v0_sq_mv2 <= 0:
                continue
            current_rows.append((vd_mv, v0_mv, v0_sq_mv2))

    flush()

    rows = []
    for setting, values in grouped:
        vd_values = [item[0] for item in values]
        v0_values = [item[1] for item in values if np.isfinite(item[1])]
        v0_sq_values = [item[2] for item in values]

        vd_mean, vd_sem, n, vd_std = mean_sem(vd_values)
        v0_mean, v0_sem, _, v0_std = mean_sem(v0_values)
        v0_sq_mean, v0_sq_sem, _, v0_sq_std = mean_sem(v0_sq_values)
        rows.append(
            {
                "bulb_setting": setting,
                "vd_mv_mean": vd_mean,
                "vd_mv_sem": vd_sem,
                "vd_mv_std": vd_std,
                "v0_mv_mean": v0_mean,
                "v0_mv_sem": v0_sem,
                "v0_mv_std": v0_std,
                "v0_sq_mv2_mean": v0_sq_mean,
                "v0_sq_mv2_sem": v0_sq_sem,
                "v0_sq_mv2_std": v0_sq_std,
                "n": n,
            }
        )

    return sorted(rows, key=lambda row: row["vd_mv_mean"])


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
