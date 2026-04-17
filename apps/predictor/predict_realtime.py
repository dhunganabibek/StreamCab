"""
Every PREDICTION_INTERVAL_SECONDS:
  - Load latest Spark aggregates from Postgres
  - Run XGBoost fare predictor per pickup zone
  - Write predictions back to Postgres
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
import psycopg
import psycopg.rows

_PROJECT_ROOT = Path(__file__).parents[2]

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://streamcab:streamcab@postgres:5432/streamcab")
MODEL_FILE = Path(os.getenv("MODEL_FILE", str(_PROJECT_ROOT / "models/traffic_model.joblib")))
PREDICTION_INTERVAL_SECONDS = int(os.getenv("PREDICTION_INTERVAL_SECONDS", "30"))


def _to_dt(value: object) -> datetime:
    scalar = cast(Any, value)
    if bool(pd.isna(scalar)):
        raise ValueError("Expected a timestamp value, got null")
    return cast(datetime, pd.Timestamp(scalar).to_pydatetime())


def _to_int(value: object, default: int = 0) -> int:
    scalar = cast(Any, value)
    if bool(pd.isna(scalar)):
        return default
    return int(scalar)


def _to_float(value: object, default: float = 0.0) -> float:
    scalar = cast(Any, value)
    if bool(pd.isna(scalar)):
        return default
    return float(scalar)


def wait_for_db() -> None:
    while True:
        try:
            with psycopg.connect(DATABASE_URL):
                pass
            print("Postgres ready.")
            return
        except Exception:
            print("Postgres not ready, retrying in 3 s…")
            time.sleep(3)


def load_aggregates() -> pd.DataFrame:
    try:
        with psycopg.connect(
            DATABASE_URL,
            row_factory=cast(Any, psycopg.rows.dict_row),
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT window_start, window_end, pu_location_id, trip_count,
                           avg_speed_mph, avg_duration_min, avg_trip_distance,
                           avg_total_amount, anomaly_count
                    FROM traffic_agg
                    ORDER BY window_start
                """
                )
                rows = cur.fetchall()
    except Exception as ex:
        print(f"Failed to load aggregates: {ex}")
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True, errors="coerce")
    df = df.dropna(subset=["window_start", "window_end"]).sort_values("window_start")
    # Deduplicate: keep last row per (zone, window) from Spark append mode
    df = df.drop_duplicates(subset=["pu_location_id", "window_start"], keep="last")
    return df


def write_predictions(output: pd.DataFrame) -> None:
    records = [
        (
            _to_int(r["pu_location_id"]),
            _to_dt(r["window_start"]),
            _to_dt(r["window_end"]),
            _to_int(r.get("trip_count", 0)),
            _to_float(r.get("avg_speed_mph", 0.0)),
            _to_float(r.get("avg_trip_distance", 0.0)),
            _to_float(r["predicted_avg_fare"]),
            _to_float(r["predicted_tip_low"]),
            _to_float(r["predicted_tip_high"]),
            _to_float(r["surge_multiplier"], 1.0),
            _to_dt(r["prediction_for_window_start"]),
            _to_dt(r["prediction_for_window_end"]),
            _to_dt(r["prediction_generated_at"]),
        )
        for r in output.to_dict(orient="records")
    ]

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM predictions")
                cur.executemany(
                    """
                    INSERT INTO predictions (
                        pu_location_id, window_start, window_end, trip_count,
                        avg_speed_mph, avg_trip_distance, predicted_avg_fare,
                        predicted_tip_low, predicted_tip_high, surge_multiplier,
                        prediction_for_window_start, prediction_for_window_end,
                        prediction_generated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    records,
                )
            conn.commit()
    except Exception as ex:
        print(f"Failed to write predictions: {ex}")


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

    for col in fare_features:
        if col not in rows.columns:
            rows[col] = 0

    rows["predicted_avg_fare"] = fare_model.predict(rows[fare_features]).clip(min=0).round(2)
    rows["predicted_tip_low"] = (rows["predicted_avg_fare"] * 0.15).round(2)
    rows["predicted_tip_high"] = (rows["predicted_avg_fare"] * 0.20).round(2)

    median_trips = float(rows["trip_count"].median()) if "trip_count" in rows.columns else 1.0
    if median_trips > 0 and "trip_count" in rows.columns:
        rows["surge_multiplier"] = (
            (rows["trip_count"] / median_trips).clip(lower=1.0, upper=3.0).round(2)
        )
    else:
        rows["surge_multiplier"] = 1.0

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

    write_predictions(output)
    print(
        f"Predictions written: {len(output)} zones | "
        f"max fare ${output['predicted_avg_fare'].max():.2f} | "
        f"max surge {output['surge_multiplier'].max():.2f}x"
    )


def main() -> None:
    print("StreamCab — Realtime Predictor running.")
    wait_for_db()
    while True:
        try:
            run_prediction_cycle()
        except Exception as ex:
            print(f"Prediction cycle skipped: {ex}")
        time.sleep(PREDICTION_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
