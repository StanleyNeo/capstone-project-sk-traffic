"""
pipeline.py — Part 2, Task 1: Data pipeline for the Smart City Traffic project.

Loads the raw Metro Interstate Traffic Volume CSV, validates the schema,
cleans the known data-quality issues found in Part 1, and writes a tidy
dataset to data/processed/traffic_clean.csv.

Cleaning rules (each is logged):
  1. Duplicate timestamps      -> keep first record (48,204 -> 40,575 rows)
  2. temp == 0 Kelvin          -> impute with the median temp of that month
  3. rain_1h > 100 mm          -> sensor spike, reset to 0.0
  4. traffic_volume == 0       -> KEPT (genuine near-zero event, 2016-07-23)
  5. holiday NaN               -> filled with the string "None" (raw convention)

Run from anywhere:  python part2_python/pipeline.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_CSV = BASE_DIR / "data" / "Metro_Interstate_Traffic_Volume.csv"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_CSV = OUT_DIR / "traffic_clean.csv"
LOG_FILE = BASE_DIR / "pipeline.log"

EXPECTED_COLUMNS = [
    "holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
    "weather_main", "weather_description", "date_time", "traffic_volume",
]

ZERO_KELVIN = 0.0          # physically impossible air temperature
RAIN_SPIKE_MM = 100.0      # 1-hour rainfall above this is a sensor error


def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw CSV and validate its schema."""
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)
    logger.info("Loaded %d rows x %d columns", df.shape[0], df.shape[1])
    return df


def remove_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first record for each timestamp (values are consistent)."""
    before = len(df)
    df = df.drop_duplicates(subset=["date_time"], keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        logger.warning(
            "Dropped %d duplicate-timestamp rows (%d -> %d)", removed, before, len(df)
        )
    return df


def impute_zero_kelvin(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 0 K sensor errors with the median temperature of that month."""
    bad = df["temp"] == ZERO_KELVIN
    n_bad = int(bad.sum())
    if n_bad:
        monthly_median = (
            df.loc[~bad].groupby(df["date_time"].dt.month)["temp"].median()
        )
        months = df.loc[bad, "date_time"].dt.month
        df.loc[bad, "temp"] = months.map(monthly_median)
        logger.warning(
            "Imputed %d zero-Kelvin rows with monthly medians (months: %s)",
            n_bad, sorted(months.unique().tolist()),
        )
    return df


def fix_rain_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """Reset physically impossible 1-hour rainfall spikes to 0."""
    bad = df["rain_1h"] > RAIN_SPIKE_MM
    n_bad = int(bad.sum())
    if n_bad:
        logger.warning(
            "Reset %d rain_1h spike(s) > %.0f mm to 0.0 (max was %.1f mm)",
            n_bad, RAIN_SPIKE_MM, df.loc[bad, "rain_1h"].max(),
        )
        df.loc[bad, "rain_1h"] = 0.0
    return df


def report_zero_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-volume hours are kept: they are a genuine near-zero event."""
    zero = df[df["traffic_volume"] == 0]
    if len(zero):
        dates = [str(t) for t in zero["date_time"]]
        logger.info(
            "Keeping %d zero-traffic rows (genuine near-zero event): %s",
            len(zero), dates,
        )
    return df


def normalise_holiday(df: pd.DataFrame) -> pd.DataFrame:
    """Restore the raw convention: holiday column holds 'None' as a string."""
    n_na = int(df["holiday"].isna().sum())
    df["holiday"] = df["holiday"].fillna("None")
    logger.debug("Filled %d NaN holiday values with 'None'", n_na)
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity checks on the cleaned frame; raises on failure."""
    assert df["date_time"].is_unique, "timestamps must be unique after dedup"
    assert df["date_time"].is_monotonic_increasing, "timestamps must be sorted"
    assert int(df.isna().sum().sum()) == 0, "no NaNs allowed after cleaning"
    temp_c = df["temp"] - 273.15
    logger.debug(
        "Checks passed | temp range %.2f..%.2f C | rain max %.2f mm | rows %d",
        temp_c.min(), temp_c.max(), df["rain_1h"].max(), len(df),
    )
    return df


def run() -> None:
    df = load_raw(RAW_CSV)
    df = remove_duplicate_timestamps(df)
    df = impute_zero_kelvin(df)
    df = fix_rain_spikes(df)
    df = report_zero_traffic(df)
    df = normalise_holiday(df)
    df = validate(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    logger.info("Wrote cleaned data -> %s (%d rows)", OUT_CSV, len(df))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, mode="w"),
                  logging.StreamHandler(sys.stdout)],
    )
    try:
        run()
        logger.info("Pipeline finished successfully")
    except Exception:
        logger.error("Pipeline failed", exc_info=True)
        sys.exit(1)