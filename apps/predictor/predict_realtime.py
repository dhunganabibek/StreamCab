"""
Real-time inference loop.

Every PREDICTION_INTERVAL_SECONDS:
  1. Load the latest trained model bundle.
  2. Take the most recent window of Spark aggregates.
  3. Predict next-window trip demand (XGBoost demand model).
  4. Predict avg fare (XGBoost amount model, if available).
  5. Compute surge multiplier = zone demand / city-wide average demand.
  6. Estimate tip range (15–20 % of predicted fare).
  7. Write results to PREDICTIONS_FILE for the dashboard.
"""

import json
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd


AGGREGATE_DIR = Path(os.getenv("AGGREGATE_DIR", "/opt/streamcab/data/processed/traffic_agg"))
MODEL_FILE = Path(os.getenv("MODEL_FILE", "/opt/streamcab/data/models/traffic_model.joblib"))
PREDICTIONS_FILE = Path(
    os.getenv("PREDICTIONS_FILE", "/opt/streamcab/data/models/realtime_predictions.parquet")
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
    df = df.dropna(subset=["window_start", "window_end"]).sort_values(by="window_start")

    # Deduplicate: keep last row per (zone, window) from append-mode parquet
    dedup_keys = ["pu_location_id", "window_start"]
    if set(dedup_keys).issubset(df.columns):
        df = df.drop_duplicates(subset=dedup_keys, keep="last")

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["hour"] = work["window_start"].dt.hour
    work["day_of_week"] = work["window_start"].dt.dayofweek
    work["is_weekend"] = (work["day_of_week"] >= 5).astype(int)
    return work


def compute_surge(predicted_counts: pd.Series) -> pd.Series:
    """
    Surge multiplier: how much higher is this zone's demand vs city average.
    Clamped to [1.0, 3.0] — matching Uber's typical surge range.
    """
    mean_demand = predicted_counts.mean()
    if mean_demand == 0:
        return pd.Series(1.0, index=predicted_counts.index)
    raw_surge = predicted_counts / mean_demand
    return raw_surge.clip(lower=1.0, upper=3.0).round(2)


def run_prediction_cycle() -> None:
    if not MODEL_FILE.exists():
        print("Model file not found yet. Waiting...")
        return

    aggregates = load_aggregates()
    if aggregates.empty:
        print("No aggregate data yet. Waiting...")
        return

    bundle = joblib.load(MODEL_FILE)
    demand_model = bundle["demand_model"]
    demand_features = bundle["demand_features"]
    amount_model = bundle.get("amount_model")
    amount_features = bundle.get("amount_features", [])

    latest_window_end = aggregates["window_end"].max()
    latest_rows = cast(
        pd.DataFrame,
        aggregates[aggregates["window_end"] == latest_window_end].copy(),
    )
    featured = create_features(latest_rows)

    # --- Demand prediction ---
    featured["predicted_trip_count_next_window"] = (
        demand_model.predict(featured[demand_features]).clip(min=0).round(1)
    )

    # --- Surge multiplier ---
    predicted_counts = cast(pd.Series, featured["predicted_trip_count_next_window"])
    featured["surge_multiplier"] = compute_surge(predicted_counts)

    # --- Fare / tip prediction ---
    if amount_model is not None and amount_features:
        missing = set(amount_features) - set(featured.columns)
        if not missing:
            featured["predicted_avg_fare"] = (
                amount_model.predict(featured[amount_features]).clip(min=0).round(2)
            )
        else:
            featured["predicted_avg_fare"] = np.nan
    else:
        # Fallback: estimate from distance and zone demand
        if "avg_trip_distance" in featured.columns:
            featured["predicted_avg_fare"] = (featured["avg_trip_distance"] * 3.0 + 5.0).round(2)
        else:
            featured["predicted_avg_fare"] = np.nan

    featured["predicted_tip_low"] = (featured["predicted_avg_fare"] * 0.15).round(2)
    featured["predicted_tip_high"] = (featured["predicted_avg_fare"] * 0.20).round(2)

    # --- Window labels for the next predicted window ---
    featured["prediction_generated_at"] = pd.Timestamp.utcnow()
    featured["prediction_for_window_start"] = featured["window_end"]
    featured["prediction_for_window_end"] = featured["window_end"] + timedelta(minutes=10)

    output_cols = [
        "pu_location_id",
        "window_start",
        "window_end",
        "trip_count",
        "avg_speed_mph",
        "anomaly_count",
        "predicted_trip_count_next_window",
        "surge_multiplier",
        "predicted_avg_fare",
        "predicted_tip_low",
        "predicted_tip_high",
        "prediction_for_window_start",
        "prediction_for_window_end",
        "prediction_generated_at",
    ]
    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in featured.columns]

    output_df = cast(pd.DataFrame, featured[output_cols])
    output = output_df.sort_values(by="predicted_trip_count_next_window", ascending=False)
    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(PREDICTIONS_FILE, index=False)

    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "rows": int(len(output)),
        "top_zone": int(output.iloc[0]["pu_location_id"]) if not output.empty else None,
        "top_predicted_demand": (
            float(output.iloc[0]["predicted_trip_count_next_window"]) if not output.empty else None
        ),
        "top_surge": (float(output.iloc[0]["surge_multiplier"]) if not output.empty else None),
    }

    with (PREDICTIONS_FILE.parent / "realtime_predictions_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote realtime predictions for {len(output)} zones. Top surge: {summary['top_surge']}x")


def main() -> None:
    print("Realtime predictor is running.")

    while True:
        try:
            run_prediction_cycle()
        except Exception as ex:
            print(f"Prediction cycle skipped: {ex}")

        time.sleep(PREDICTION_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
