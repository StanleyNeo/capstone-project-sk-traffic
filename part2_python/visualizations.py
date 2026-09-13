"""
visualizations.py — Part 2, Task 3: Visual analysis of the cleaned data.

Reads data/processed/traffic_features.csv and saves four figures to
part2_python/figures/:
  fig1_hourly_profile.png  average traffic by hour, weekday vs weekend
  fig2_heatmap.png         day-of-week x hour congestion heatmap
  fig3_weather_impact.png  average traffic by weather condition
  fig4_monthly_rhythm.png  average traffic by calendar month

Run from anywhere:  python part2_python/visualizations.py
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures without opening windows
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
IN_CSV = BASE_DIR / "data" / "processed" / "traffic_features.csv"
FIG_DIR = BASE_DIR / "part2_python" / "figures"
LOG_FILE = BASE_DIR / "pipeline.log"

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fig1_hourly_profile(df: pd.DataFrame) -> None:
    """Average traffic by hour — weekday vs weekend twin-peak profile."""
    weekday = df[df["is_weekend"] == 0].groupby("hour")["traffic_volume"].mean()
    weekend = df[df["is_weekend"] == 1].groupby("hour")["traffic_volume"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(weekday.index, weekday.values, marker="o", label="Weekday")
    ax.plot(weekend.index, weekend.values, marker="s", label="Weekend")
    ax.set_title("Average Traffic by Hour — Weekday vs Weekend (2012–2018)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average traffic volume (vehicles/hour)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "fig1_hourly_profile.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(
        "Saved %s | weekday peak %02d:00 (%.0f) | weekend peak %02d:00 (%.0f)",
        out, weekday.idxmax(), weekday.max(), weekend.idxmax(), weekend.max(),
    )


def fig2_heatmap(df: pd.DataFrame) -> None:
    """Day-of-week x hour heatmap of average traffic volume."""
    pivot = df.pivot_table(
        index="day_of_week", columns="hour",
        values="traffic_volume", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_title("Traffic Intensity Heatmap — Day of Week × Hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")
    ax.set_xticks(range(0, 24, 2))
    ax.set_yticks(range(7), DAYS)
    fig.colorbar(im, ax=ax, label="Avg vehicles/hour")
    fig.tight_layout()
    out = FIG_DIR / "fig2_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    hottest = pivot.stack().idxmax()
    logger.info(
        "Saved %s | busiest cell %s %02d:00 (%.0f)",
        out, DAYS[hottest[0]], hottest[1], pivot.max().max(),
    )


def fig3_weather_impact(df: pd.DataFrame) -> None:
    """Average traffic by weather condition, with sample sizes."""
    weather_cols = [c for c in df.columns if c.startswith("weather_")]
    records = {}
    for col in weather_cols:
        name = col.replace("weather_", "")
        records[name] = (df.loc[df[col] == 1, "traffic_volume"].mean(),
                         int(df[col].sum()))
    stats = pd.DataFrame(records, index=["mean", "n"]).T.sort_values("mean")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(stats.index, stats["mean"], color="steelblue")
    for i, (mean, n) in enumerate(zip(stats["mean"], stats["n"])):
        ax.text(mean + 30, i, f"{mean:,.0f}  (n={int(n):,})", va="center", fontsize=9)
    ax.set_title("Average Traffic by Weather Condition (cleaned data)")
    ax.set_xlabel("Average traffic volume (vehicles/hour)")
    ax.set_xlim(0, stats["mean"].max() * 1.25)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig3_weather_impact.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(
        "Saved %s | highest %s (%.0f) | lowest well-sampled %s (%.0f)",
        out, stats.index[-1], stats["mean"].iloc[-1],
        stats[stats["n"] > 100].index[0], stats[stats["n"] > 100]["mean"].iloc[0],
    )


def fig4_monthly_rhythm(df: pd.DataFrame) -> None:
    """Average traffic by calendar month — the seasonal rhythm."""
    monthly = df.groupby("month")["traffic_volume"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(MONTHS, monthly.values, color="steelblue")
    ax.axhline(monthly.mean(), color="darkred", linestyle="--",
               label=f"Year average ({monthly.mean():,.0f})")
    ax.set_title("Average Traffic by Month (2012–2018)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average traffic volume (vehicles/hour)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "fig4_monthly_rhythm.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(
        "Saved %s | highest %s (%.0f) | lowest %s (%.0f)",
        out, MONTHS[monthly.idxmax() - 1], monthly.max(),
        MONTHS[monthly.idxmin() - 1], monthly.min(),
    )


def run() -> None:
    logger.info("Loading features from %s", IN_CSV)
    df = pd.read_csv(IN_CSV, parse_dates=["date_time"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_hourly_profile(df)
    fig2_heatmap(df)
    fig3_weather_impact(df)
    fig4_monthly_rhythm(df)
    logger.info("All 4 figures saved to %s", FIG_DIR)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, mode="a"),
                  logging.StreamHandler(sys.stdout)],
    )
    # matplotlib is very chatty at DEBUG — keep third-party noise out of the log
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    try:
        run()
        logger.info("Visualizations finished successfully")
    except Exception:
        logger.error("Visualization failed", exc_info=True)
        sys.exit(1)