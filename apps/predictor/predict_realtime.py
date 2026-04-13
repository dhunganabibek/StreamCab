"""
Every PREDICTION_INTERVAL_SECONDS:
Load the latest Spark aggregates then Run the XGBoost fare predictor per pickup zone.
estimate prediction and write it to file
"""

import json
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import cast

import joblib
import pandas as pd

_PROJECT_ROOT = Path(__file__).parents[2]

AGGREGATE_DIR = Path(os.getenv("AGGREGATE_DIR", str(_PROJECT_ROOT / "data/processed/traffic_agg")))
MODEL_FILE = Path(os.getenv("MODEL_FILE", str(_PROJECT_ROOT / "data/models/traffic_model.joblib")))
PREDICTIONS_FILE = Path(
    os.getenv(
        "PREDICTIONS_FILE",
        str(_PROJECT_ROOT / "data/models/realtime_predictions.parquet"),
    )
)
PREDICTION_INTERVAL_SECONDS = int(os.getenv("PREDICTION_INTERVAL_SECONDS", "30"))


def load_aggregates() -> pd.DataFrame:
    if not AGGREGATE_DIR.exists() or not list(AGGREGATE_DIR.rglob("*.parquet")):
        return pd.DataFrame()

    df = pd.read_parquet(AGGREGATE_DIR)
    if df.empty:
        return df

    df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True, errors="coerce")
    df = df.dropna(subset=["window_start", "window_end"]).sort_values("window_start")

    # Deduplicate: keep last row per (zone, window) from append-mode parquet
    df = df.drop_duplicates(subset=["pu_location_id", "window_start"], keep="last")
    return df


def run_prediction_cycle() -> None:
    if not MODEL_FILE.exists():
        print("Model file not found yet. Waiting...")
        return

    aggregates = load_aggregates()
    if aggregates.empty:
        print("No aggregate data yet. Waiting...")
        return

    bundle = joblib.load(MODEL_FILE)
    fare_model = bundle["fare_model"]
    fare_features = bundle["fare_features"]

    # Use the most recent window per zone
    latest_window_end = aggregates["window_end"].max()
    rows = cast(
        pd.DataFrame,
        aggregates[aggregates["window_end"] == latest_window_end].copy(),
    )

    # Build time features
    rows["hour"] = rows["window_start"].dt.hour
    rows["day_of_week"] = rows["window_start"].dt.dayofweek
    rows["is_weekend"] = (rows["day_of_week"] >= 5).astype(int)

    # Fill any missing feature columns with zone medians
    for col in fare_features:
        if col not in rows.columns:
            rows[col] = 0

    # Fare prediction
    rows["predicted_avg_fare"] = fare_model.predict(rows[fare_features]).clip(min=0).round(2)

    # Tip estimates: 15-20% of predicted fare based from model
    rows["predicted_tip_low"] = (rows["predicted_avg_fare"] * 0.15).round(2)
    rows["predicted_tip_high"] = (rows["predicted_avg_fare"] * 0.20).round(2)

    # Surge multiplier: zone trip count vs median across all zones
    median_trips = float(rows["trip_count"].median()) if "trip_count" in rows.columns else 1.0
    if median_trips > 0 and "trip_count" in rows.columns:
        rows["surge_multiplier"] = (
            (rows["trip_count"] / median_trips).clip(lower=1.0, upper=3.0).round(2)
        )
    else:
        rows["surge_multiplier"] = 1.0

    # Next-window labels
    rows["prediction_generated_at"] = pd.Timestamp.utcnow()
    rows["prediction_for_window_start"] = rows["window_end"]
    rows["prediction_for_window_end"] = rows["window_end"] + timedelta(minutes=10)

    output_cols = [
        "pu_location_id",
        "window_start",
        "window_end",
        "trip_count",
        "avg_speed_mph",
        "avg_trip_distance",
        "predicted_avg_fare",
        "predicted_tip_low",
        "predicted_tip_high",
        "surge_multiplier",
        "prediction_for_window_start",
        "prediction_for_window_end",
        "prediction_generated_at",
    ]
    output_cols = [c for c in output_cols if c in rows.columns]
    output = cast(pd.DataFrame, rows[output_cols]).sort_values(
        by="predicted_avg_fare", ascending=False
    )

    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(PREDICTIONS_FILE, index=False)

    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "zones": len(output),
        "top_zone": int(output.iloc[0]["pu_location_id"]) if not output.empty else None,
        "top_predicted_fare": (
            float(output.iloc[0]["predicted_avg_fare"]) if not output.empty else None
        ),
        "max_surge": (float(output["surge_multiplier"].max()) if not output.empty else None),
    }

    with (PREDICTIONS_FILE.parent / "realtime_predictions_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print(
        f"Predictions written: {len(output)} zones | "
        f"max fare ${summary['top_predicted_fare']:.2f} | "
        f"max surge {summary['max_surge']:.2f}x"
    )


def main() -> None:
    print("StreamCab — Realtime Predictor running.")
    while True:
        try:
            run_prediction_cycle()
        except Exception as ex:
            print(f"Prediction cycle skipped: {ex}")
        time.sleep(PREDICTION_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
