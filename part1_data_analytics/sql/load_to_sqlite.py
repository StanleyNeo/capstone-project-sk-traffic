"""Load the Metro Interstate Traffic Volume CSV into a SQLite database.

Part 1 - Task 1.1: Load the dataset and verify it loaded correctly.
Uses only the Python standard library (csv + sqlite3).
"""
import csv
import logging
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]  # repo root
CSV_PATH = BASE_DIR / "data" / "Metro_Interstate_Traffic_Volume.csv"
DB_PATH = BASE_DIR / "data" / "traffic.db"

EXPECTED_COLUMNS = [
    "holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
    "weather_main", "weather_description", "date_time", "traffic_volume",
]

CREATE_TABLE_SQL = """
CREATE TABLE traffic (
    holiday             TEXT,
    temp                REAL,
    rain_1h             REAL,
    snow_1h             REAL,
    clouds_all          INTEGER,
    weather_main        TEXT,
    weather_description TEXT,
    date_time           TEXT,
    traffic_volume      INTEGER
);
"""

INSERT_SQL = "INSERT INTO traffic VALUES (?,?,?,?,?,?,?,?,?)"


def load_csv_to_sqlite(csv_path: Path, db_path: Path) -> int:
    """Load the CSV into the traffic table. Returns rows inserted."""
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [c for c in EXPECTED_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise ValueError(f"CSV is missing expected columns: {missing}")
            rows = [tuple(r[c] if r[c] != "" else None for c in EXPECTED_COLUMNS)
                    for r in reader]
    except FileNotFoundError:
        logger.error("CSV file not found: %s", csv_path, exc_info=True)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Schema validation failed: %s", exc, exc_info=True)
        sys.exit(1)

    try:
        with sqlite3.connect(db_path) as conn:
            # DROP first so re-running the loader is fully reproducible
            conn.execute("DROP TABLE IF EXISTS traffic")
            conn.execute(CREATE_TABLE_SQL)
            conn.executemany(INSERT_SQL, rows)
    except sqlite3.Error:
        logger.error("Failed to write to SQLite database %s", db_path, exc_info=True)
        sys.exit(1)

    logger.info("Loaded %d rows x %d columns from %s into %s",
                len(rows), len(EXPECTED_COLUMNS), csv_path.name, db_path.name)
    return len(rows)


def verify_load(db_path: Path) -> None:
    """Task 1.1 verification: row count, date range, sample rows."""
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM traffic").fetchone()[0]
        first, last = conn.execute(
            "SELECT MIN(date_time), MAX(date_time) FROM traffic").fetchone()
        distinct_ts = conn.execute(
            "SELECT COUNT(DISTINCT date_time) FROM traffic").fetchone()[0]
    logger.info("Verification: %d rows in table 'traffic'", count)
    logger.info("Verification: date range %s to %s", first, last)
    logger.info("Verification: %d distinct timestamps "
                "(difference vs row count = duplicate timestamps to handle in Part 2)",
                distinct_ts)
    if count != 48204:
        logger.warning("Row count %d does not match expected 48,204 - check the CSV", count)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )
    load_csv_to_sqlite(CSV_PATH, DB_PATH)
    verify_load(DB_PATH)