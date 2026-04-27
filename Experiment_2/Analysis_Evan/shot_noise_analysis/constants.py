from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "shot_noise_data"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
RESULT_DIR = OUTPUT_DIR / "results"

CALIBRATION_CSV = DATA_DIR / "data - calibration 4_07 multimeter.csv"
SHOT_NOISE_CSV = DATA_DIR / "data - shot noise 4_07.csv"

RF_OHM = 475e3
RF_REL_UNCERTAINTY = 0.01
RF_ERR_OHM = RF_OHM * RF_REL_UNCERTAINTY
RTEST_OHM = 475e3
RTEST_REL_UNCERTAINTY = 0.01
RTEST_ERR_OHM = RTEST_OHM * RTEST_REL_UNCERTAINTY

TAIL_START_HZ = 50e3
ACCEPTED_E_C = 1.602176634e-19

# Empirical per-reading RMS floor for the AC voltmeter/noise measurement.
# This is propagated as delta(V0^2) = 2 V0 delta(V0), in addition to repeat SEM.
V0_RMS_FLOOR_MV = 0.20
