"""
main.py — Part 2, Task 4: Command-line app for the traffic intelligence data.

Four commands (run from anywhere):
  python part2_python/cli_app/main.py summary
  python part2_python/cli_app/main.py hourly
  python part2_python/cli_app/main.py weather Clear
  python part2_python/cli_app/main.py top --n 10

Data source: data/processed/traffic_clean.csv (output of pipeline.py).
print() is used for end-user output; logging tracks progress and errors.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CLEAN_CSV = BASE_DIR / "data" / "processed" / "traffic_clean.csv"
LOG_FILE = BASE_DIR / "pipeline.log"


def load_data() -> pd.DataFrame:
    """Load the cleaned dataset produced by pipeline.py."""
    if not CLEAN_CSV.exists():
        logger.error("Cleaned data not found: %s — run pipeline.py first", CLEAN_CSV)
        sys.exit(1)
    df = pd.read_csv(CLEAN_CSV, parse_dates=["date_time"])
    logger.info("Loaded %d rows from %s", len(df), CLEAN_CSV)
    return df


def cmd_summary(df: pd.DataFrame) -> None:
    """Dataset overview: size, span, central tendency, congestion share."""
    tv = df["traffic_volume"]
    q3 = tv.quantile(0.75)
    hourly = tv.groupby(df["date_time"].dt.hour).mean()
    print("\n=== TRAFFIC DATA SUMMARY ===")
    print(f"Rows (cleaned)      : {len(df):,}")
    print(f"Period              : {df['date_time'].min()}  ->  {df['date_time'].max()}")
    print(f"Mean volume         : {tv.mean():,.1f} vehicles/hour")
    print(f"Median volume       : {tv.median():,.0f} vehicles/hour")
    print(f"Std deviation       : {tv.std():,.1f}")
    print(f"Min / Max           : {tv.min():,} / {tv.max():,}")
    print(f"Congestion threshold: Q3 = {q3:,.0f} vehicles/hour")
    print(f"Congested hours     : {(tv >= q3).sum():,} ({(tv >= q3).mean() * 100:.2f}%)")
    print(f"Busiest hour        : {hourly.idxmax():02d}:00 (avg {hourly.max():,.0f})")
    print(f"Quietest hour       : {hourly.idxmin():02d}:00 (avg {hourly.min():,.0f})")


def cmd_hourly(df: pd.DataFrame) -> None:
    """Average traffic for each hour of the day, with a text bar chart."""
    hourly = df.groupby(df["date_time"].dt.hour)["traffic_volume"].mean()
    print("\n=== AVERAGE TRAFFIC BY HOUR ===")
    for hour, avg in hourly.items():
        bar = "#" * int(avg / 100)
        print(f"{hour:02d}:00  {avg:6,.0f}  {bar}")


def cmd_weather(df: pd.DataFrame, condition: str) -> None:
    """Average traffic under one weather condition vs the overall average."""
    choices = sorted(df["weather_main"].unique())
    match = [w for w in choices if w.lower() == condition.lower()]
    if not match:
        logger.warning("Unknown weather condition: %s", condition)
        print(f"Unknown condition '{condition}'. Choose from: {', '.join(choices)}")
        return
    name = match[0]
    subset = df.loc[df["weather_main"] == name, "traffic_volume"]
    overall = df["traffic_volume"].mean()
    rank = (df.groupby("weather_main")["traffic_volume"].mean() > subset.mean()).sum() + 1
    print(f"\n=== WEATHER: {name} ===")
    print(f"Records             : {len(subset):,}")
    print(f"Average volume      : {subset.mean():,.1f} vehicles/hour")
    print(f"Overall average     : {overall:,.1f}")
    print(f"Difference          : {subset.mean() - overall:+,.1f} ({(subset.mean() / overall - 1) * 100:+.1f}%)")
    print(f"Rank (of {len(choices)})       : {rank}  (1 = highest traffic)")
    if len(subset) < 100:
        print("NOTE: small sample — treat this average with caution.")


def cmd_top(df: pd.DataFrame, n: int) -> None:
    """The n most congested hours in the whole dataset."""
    top = df.nlargest(n, "traffic_volume")
    print(f"\n=== TOP {n} MOST CONGESTED HOURS ===")
    for i, row in enumerate(top.itertuples(), 1):
        print(f"{i:2d}. {row.date_time}  {row.traffic_volume:,} veh/h  ({row.weather_main})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traffic-cli",
        description="Smart City Traffic Intelligence — query the cleaned I-94 dataset.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("summary", help="dataset overview and congestion stats")
    sub.add_parser("hourly", help="average traffic by hour of day")
    p_weather = sub.add_parser("weather", help="traffic under a weather condition")
    p_weather.add_argument("condition", help="e.g. Clear, Rain, Fog, Snow ...")
    p_top = sub.add_parser("top", help="most congested hours in the dataset")
    p_top.add_argument("--n", type=int, default=10, help="how many rows (default 10)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = load_data()
    if args.command == "summary":
        cmd_summary(df)
    elif args.command == "hourly":
        cmd_hourly(df)
    elif args.command == "weather":
        cmd_weather(df, args.condition)
    elif args.command == "top":
        cmd_top(df, args.n)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, mode="a")],
    )
    try:
        main()
    except Exception:
        logger.error("CLI command failed", exc_info=True)
        sys.exit(1)