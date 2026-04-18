"""
Every PREDICTION_INTERVAL_SECONDS:
  Load latest Spark aggregates from Postgres
  Run XGBoost tip predictor per pickup zone
  Write predictions back to Postgres
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
MODEL_FILE = Path(os.getenv("MODEL_FILE", str(_PROJECT_ROOT / "models/tip_model.joblib")))
PREDICTION_INTERVAL_SECONDS = int(os.getenv("PREDICTION_INTERVAL_SECONDS", "30"))
AGG_LOOKBACK_HOURS = int(os.getenv("AGG_LOOKBACK_HOURS", "48"))


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
                    WITH latest AS (
                        SELECT MAX(window_end) AS max_window_end
                        FROM traffic_agg
                    )
                    SELECT DISTINCT ON (t.pu_location_id)
                           t.window_start,
                           t.window_end,
                           t.pu_location_id,
                           t.trip_count,
                           t.avg_speed_mph,
                           t.avg_duration_min,
                           t.avg_trip_distance,
                           t.avg_total_amount,
                           t.anomaly_count
                    FROM traffic_agg AS t
                    CROSS JOIN latest AS l
                    WHERE l.max_window_end IS NOT NULL
                      AND t.window_end >= l.max_window_end - (%s * INTERVAL '1 hour')
                    ORDER BY t.pu_location_id, t.window_end DESC, t.window_start DESC
                """,
                    (AGG_LOOKBACK_HOURS,),
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
    df = df.dropna(subset=["window_start", "window_end"]).sort_values("pu_location_id")
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
    tip_model = bundle.get("tip_model") or bundle.get("fare_model")
    tip_features = bundle.get("tip_features") or bundle.get("fare_features")
    if tip_model is None or tip_features is None:
        print("Model bundle missing model/features keys. Waiting...")
        return

    # SQL already returns the latest window row per zone
    rows = aggregates.copy()

    # Build time features
    rows["hour"] = rows["window_start"].dt.hour
    rows["day_of_week"] = rows["window_start"].dt.dayofweek
    rows["is_weekend"] = (rows["day_of_week"] >= 5).astype(int)

    for col in tip_features:
        if col not in rows.columns:
            rows[col] = 0

    tip_pred = pd.Series(tip_model.predict(rows[tip_features]), index=rows.index).clip(lower=0)
    rows["predicted_tip_low"] = (tip_pred * 0.9).round(2)
    rows["predicted_tip_high"] = (tip_pred * 1.1).round(2)

    if "avg_total_amount" in rows.columns:
        avg_total_amount = cast(
            pd.Series,
            pd.to_numeric(rows["avg_total_amount"], errors="coerce"),
        )
        rows["predicted_avg_fare"] = avg_total_amount.fillna(0).clip(lower=0).round(2)
    else:
        rows["predicted_avg_fare"] = (tip_pred * 5).round(2)

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
    max_tip = output[["predicted_tip_low", "predicted_tip_high"]].max(axis=1).max()
    print(
        f"Predictions written: {len(output)} zones | "
        f"max fare ${output['predicted_avg_fare'].max():.2f} | "
        f"max tip ${max_tip:.2f} | "
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
