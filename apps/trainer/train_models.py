"""
Train demand and fare/tip forecasting models from TLC taxi data.

Run modes
---------
Loop mode (default):
    Retrain every TRAINING_INTERVAL_SECONDS from Spark-aggregated parquet files.
    Used while the live streaming pipeline is running.

ONE_SHOT mode (ONE_SHOT=true):
    Aggregate raw TLC parquet files directly (no Spark needed), train once, save
    the model, and exit.  Use this after downloading historical data:

        # 1. Download data
        python scripts/download_tlc_data.py \\
            --taxi-type yellow --start 2022-01 --end 2024-12 --output data/raw

        # 2. Train on raw data
        docker compose run --rm -e ONE_SHOT=true trainer

        # 3. Start live pipeline (model already loaded)
        docker compose up --build
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


AGGREGATE_DIR = Path(os.getenv("AGGREGATE_DIR", "/opt/streamcab/data/processed/traffic_agg"))
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "/opt/streamcab/data/raw"))
MODEL_OUTPUT_DIR = Path(os.getenv("MODEL_OUTPUT_DIR", "/opt/streamcab/data/models"))
TRAINING_MIN_ROWS = int(os.getenv("TRAINING_MIN_ROWS", "20"))
TRAINING_INTERVAL_SECONDS = int(os.getenv("TRAINING_INTERVAL_SECONDS", "90"))
ONE_SHOT = os.getenv("ONE_SHOT", "false").lower() in ("true", "1", "yes")

WINDOW_SIZE = pd.Timedelta(minutes=10)

DEMAND_FEATURES = [
    "pu_location_id",
    "hour",
    "day_of_week",
    "is_weekend",
    "avg_speed_mph",
    "avg_duration_min",
    "anomaly_count",
]

AMOUNT_FEATURES = [
    "pu_location_id",
    "hour",
    "day_of_week",
    "is_weekend",
    "avg_speed_mph",
    "avg_duration_min",
    "trip_count",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def aggregate_raw_parquet(raw_dir: Path) -> pd.DataFrame:
    """
    Read raw TLC parquet files and compute 10-minute window aggregates —
    the same schema Spark produces.  Used by ONE_SHOT mode so Spark is not needed.
    """
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    print(f"  Found {len(files)} raw parquet file(s) in {raw_dir}")

    chunks = []
    for f in files:
        try:
            chunk = pd.read_parquet(
                f,
                columns=None,  # read all, filter below
            )
            chunks.append(chunk)
            print(f"  Loaded {f.name}  ({len(chunk):,} rows)")
        except Exception as ex:
            print(f"  Skipping {f.name}: {ex}")

    if not chunks:
        return pd.DataFrame()

    raw = pd.concat(chunks, ignore_index=True)
    print(f"  Total raw rows: {len(raw):,}")

    # Resolve column names (yellow vs green vs fhv naming)
    pickup_col = _first_col(
        raw, ["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"]
    )
    dropoff_col = _first_col(
        raw, ["tpep_dropoff_datetime", "lpep_dropoff_datetime", "dropoff_datetime"]
    )
    location_col = _first_col(raw, ["PULocationID", "pu_location_id"])
    distance_col = _first_col(raw, ["trip_distance"])
    amount_col = _first_col(raw, ["total_amount", "fare_amount"])

    if not all([pickup_col, dropoff_col, location_col, distance_col]):
        print("  Raw parquet is missing required columns (pickup, dropoff, location, distance).")
        return pd.DataFrame()

    raw["pickup_ts"] = pd.to_datetime(raw[pickup_col], errors="coerce")
    raw["dropoff_ts"] = pd.to_datetime(raw[dropoff_col], errors="coerce")
    raw["pu_location_id"] = pd.to_numeric(raw[location_col], errors="coerce")
    raw["trip_distance"] = pd.to_numeric(raw[distance_col], errors="coerce")

    raw["duration_min"] = (raw["dropoff_ts"] - raw["pickup_ts"]).dt.total_seconds() / 60
    raw["speed_mph"] = raw["trip_distance"] / (raw["duration_min"] / 60).replace(0, float("nan"))

    # Same cleaning filters as Spark job
    mask = (
        raw["pickup_ts"].notna()
        & raw["dropoff_ts"].notna()
        & (raw["duration_min"] > 1)
        & (raw["duration_min"] < 180)
        & (raw["trip_distance"] > 0.2)
        & (raw["trip_distance"] < 100)
        & (raw["speed_mph"] > 1)
        & (raw["speed_mph"] < 80)
        & (raw["pu_location_id"] > 0)
    )
    raw = raw[mask].copy()
    print(f"  Rows after cleaning: {len(raw):,}")

    if raw.empty:
        return pd.DataFrame()

    # 10-minute tumbling windows (floor pickup to nearest 10 min)
    pickup_ts = cast(pd.Series, raw["pickup_ts"])
    raw["window_start"] = pickup_ts.dt.floor("10min")
    raw["window_end"] = raw["window_start"] + WINDOW_SIZE

    grouped = raw.groupby(["window_start", "window_end", "pu_location_id"])

    agg = grouped.agg(
        trip_count=("trip_distance", "count"),
        avg_speed_mph=("speed_mph", "mean"),
        avg_duration_min=("duration_min", "mean"),
        avg_trip_distance=("trip_distance", "mean"),
    ).reset_index()

    # Anomaly count: speeds < 3 or > 45 mph
    anomaly = (
        raw.assign(is_anomaly=((raw["speed_mph"] < 3) | (raw["speed_mph"] > 45)).astype(int))
        .groupby(["window_start", "window_end", "pu_location_id"])["is_anomaly"]
        .sum()
        .reset_index()
        .rename(columns={"is_anomaly": "anomaly_count"})
    )
    agg = agg.merge(anomaly, on=["window_start", "window_end", "pu_location_id"], how="left")
    agg["anomaly_count"] = agg["anomaly_count"].fillna(0).astype(int)

    if amount_col:
        raw["total_amount_num"] = pd.to_numeric(raw[amount_col], errors="coerce")
        avg_amount = (
            raw.groupby(["window_start", "window_end", "pu_location_id"])["total_amount_num"]
            .mean()
            .reset_index()
            .rename(columns={"total_amount_num": "avg_total_amount"})
        )
        agg = agg.merge(avg_amount, on=["window_start", "window_end", "pu_location_id"], how="left")

    # Localize timestamps to UTC to match Spark output
    agg["window_start"] = pd.to_datetime(agg["window_start"]).dt.tz_localize("UTC")
    agg["window_end"] = pd.to_datetime(agg["window_end"]).dt.tz_localize("UTC")

    print(f"  Aggregated to {len(agg):,} (zone, window) rows")
    return agg


def load_spark_aggregates() -> pd.DataFrame:
    if not AGGREGATE_DIR.exists():
        return pd.DataFrame()

    files = list(AGGREGATE_DIR.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()

    df = pd.read_parquet(AGGREGATE_DIR)
    required = {
        "window_start",
        "window_end",
        "pu_location_id",
        "trip_count",
        "avg_speed_mph",
        "avg_duration_min",
        "anomaly_count",
    }
    if required - set(df.columns):
        return pd.DataFrame()

    return df


# ---------------------------------------------------------------------------
# Feature engineering & training
# ---------------------------------------------------------------------------


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["window_start"] = pd.to_datetime(work["window_start"], utc=True, errors="coerce")
    work = work.dropna(subset=["window_start"])
    window_start = cast(pd.Series, work["window_start"])
    work["hour"] = window_start.dt.hour
    work["day_of_week"] = window_start.dt.dayofweek
    work["is_weekend"] = (work["day_of_week"] >= 5).astype(int)
    # Deduplicate: Spark append-mode can produce duplicate (zone, window) rows
    dedup_keys = ["pu_location_id", "window_start"]
    if set(dedup_keys).issubset(work.columns):
        work = work.sort_values("window_start").drop_duplicates(subset=dedup_keys, keep="last")
    return work.sort_values("window_start")


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denom = y_true.replace(0, 1e-6)
    return float(((y_true - y_pred).abs() / denom).mean() * 100)


def _train_xgb(train: pd.DataFrame, test: pd.DataFrame, features: list, target: str):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(train[features], train[target])
    preds = pd.Series(model.predict(test[features]), index=test.index)
    return (
        model,
        float(mean_absolute_error(test[target], preds)),
        mape(cast(pd.Series, test[target]), cast(pd.Series, preds)),
    )


def fit_and_score(df: pd.DataFrame) -> dict:
    work = _build_features(df)

    split_idx = int(len(work) * 0.8)
    train, test = work.iloc[:split_idx], work.iloc[split_idx:]

    if len(train) < 10 or len(test) < 5:
        raise ValueError("Not enough train/test data yet.")

    # Baseline: historical average by (zone, hour)
    baseline_lookup = train.groupby(["pu_location_id", "hour"], as_index=False)["trip_count"].mean()
    baseline_lookup["baseline_pred"] = baseline_lookup["trip_count"]
    baseline_lookup = baseline_lookup.drop(columns=["trip_count"])
    baseline_scored = test.merge(baseline_lookup, on=["pu_location_id", "hour"], how="left")
    baseline_scored["baseline_pred"] = baseline_scored["baseline_pred"].fillna(
        train["trip_count"].mean()
    )
    baseline_mae = float(
        mean_absolute_error(baseline_scored["trip_count"], baseline_scored["baseline_pred"])
    )
    baseline_mape_val = mape(
        cast(pd.Series, baseline_scored["trip_count"]),
        cast(pd.Series, baseline_scored["baseline_pred"]),
    )

    demand_model, demand_mae, demand_mape = _train_xgb(train, test, DEMAND_FEATURES, "trip_count")

    amount_model, amount_mae, amount_mape = None, None, None
    if "avg_total_amount" in work.columns and bool(work["avg_total_amount"].notna().any()):
        amount_model, amount_mae, amount_mape = _train_xgb(
            train.dropna(subset=["avg_total_amount"]),
            test.dropna(subset=["avg_total_amount"]),
            AMOUNT_FEATURES,
            "avg_total_amount",
        )

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "demand_model": demand_model,
            "demand_features": DEMAND_FEATURES,
            "amount_model": amount_model,
            "amount_features": AMOUNT_FEATURES if amount_model else [],
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_rows": int(len(train)),
        },
        MODEL_OUTPUT_DIR / "traffic_model.joblib",
    )

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(train)),
        "testing_rows": int(len(test)),
        "baseline": {"mae": baseline_mae, "mape": baseline_mape_val},
        "xgboost": {"mae": demand_mae, "mape": demand_mape},
    }
    if amount_mae is not None:
        metrics["xgboost_amount"] = {"mae": amount_mae, "mape": amount_mape}

    with (MODEL_OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if ONE_SHOT:
        print("ONE_SHOT mode: aggregate raw TLC parquet → train → exit.")
        print(f"  Raw data dir : {RAW_DATA_DIR}")
        print(f"  Model output : {MODEL_OUTPUT_DIR}\n")

        df = aggregate_raw_parquet(RAW_DATA_DIR)

        if df.empty:
            # Fallback: wait for Spark aggregates (original behavior)
            print("No raw parquet files found. Waiting for Spark aggregates...")
            while True:
                df = load_spark_aggregates()
                if len(df) >= TRAINING_MIN_ROWS:
                    break
                print(f"  Spark aggregates: {len(df)}/{TRAINING_MIN_ROWS} rows...")
                time.sleep(10)

        try:
            metrics = fit_and_score(df)
            print("\nModel trained successfully:")
            print(json.dumps(metrics, indent=2))
            print(f"\nSaved to: {MODEL_OUTPUT_DIR / 'traffic_model.joblib'}")
            sys.exit(0)
        except Exception as ex:
            print(f"Training failed: {ex}")
            sys.exit(1)

    # Loop mode: retrain periodically from Spark aggregates
    print("Model trainer running in loop mode.")
    while True:
        try:
            df = load_spark_aggregates()
            if len(df) < TRAINING_MIN_ROWS:
                print(f"Waiting for aggregate rows ({len(df)}/{TRAINING_MIN_ROWS})...")
                time.sleep(TRAINING_INTERVAL_SECONDS)
                continue

            metrics = fit_and_score(df)
            print("Model retrained:")
            print(json.dumps(metrics, indent=2))
        except Exception as ex:  # noqa: BLE001
            print(f"Training cycle skipped: {ex}")

        time.sleep(TRAINING_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
