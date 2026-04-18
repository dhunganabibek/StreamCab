"""
Train a single XGBoost fare predictor from NYC  taxi data.
Reads raw parquet files, trains on 80% of data, saves the model, and exits.
Run once before starting the live pipeline:

docker compose run --rm train-once
"""

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import BallTree
from xgboost import XGBRegressor

_PROJECT_ROOT = Path(__file__).parents[2]

RAW_DATA_DIR: str = os.getenv("RAW_DATA_DIR", str(_PROJECT_ROOT / "data/raw-data/parquet"))
MODEL_OUTPUT_DIR = Path(os.getenv("MODEL_OUTPUT_DIR", str(_PROJECT_ROOT / "models")))
ZONE_CENTROIDS_FILE = Path(
    os.getenv("ZONE_CENTROIDS_FILE", str(_PROJECT_ROOT / "data/reference/zone_centroids.csv"))
)
# Set MAX_FILES to limit how many parquet files are used
_MAX_FILES_ENV = os.getenv("MAX_FILES", "")
MAX_FILES: int | None = int(_MAX_FILES_ENV) if _MAX_FILES_ENV.strip() else None
PARQUET_BATCH_SIZE = int(os.getenv("PARQUET_BATCH_SIZE", "500000"))
TRAIN_BATCH_ROWS = int(os.getenv("TRAIN_BATCH_ROWS", "250000"))
VALIDATION_FRACTION = float(os.getenv("VALIDATION_FRACTION", "0.1"))
INIT_TREES = int(os.getenv("INIT_TREES", "120"))
TREES_PER_BATCH = int(os.getenv("TREES_PER_BATCH", "40"))


# Storage abstraction
def _build_filesystem(raw_dir: str) -> tuple[pafs.FileSystem, list[str]]:
    """Return (filesystem, sorted list of parquet paths) for local or S3 sources."""
    if raw_dir.startswith("s3://"):
        s3_prefix = raw_dir[len("s3://") :]
        fs = pafs.S3FileSystem(
            region=os.getenv("AWS_REGION", "us-east-1"),
            anonymous=True,  # public bucket — no credentials needed
        )
        file_info = fs.get_file_info(pafs.FileSelector(s3_prefix, recursive=False))
        files = sorted(
            fi.path
            for fi in file_info
            if fi.type == pafs.FileType.File and fi.path.endswith(".parquet")
        )
        return fs, files
    else:
        local_dir = Path(raw_dir)
        fs = pafs.LocalFileSystem()
        files = sorted(str(f) for f in local_dir.glob("*.parquet"))
        return fs, files


WINDOW_SIZE = pd.Timedelta(minutes=10)

# NYC bounding box — filters out bad lat/lon rows in pre-2017 data
_NYC_LAT = (40.4, 41.0)
_NYC_LON = (-74.5, -73.5)

# Features used by the fare predictor
FARE_FEATURES = [
    "pu_location_id",
    "hour",
    "day_of_week",
    "is_weekend",
    "avg_trip_distance",
    "avg_duration_min",
    "avg_speed_mph",
    "trip_count",
]
TARGET = "avg_total_amount"

_AGG_KEYS = ["window_start", "window_end", "pu_location_id"]
_AGG_SUM_COLS = [
    "trip_count",
    "trip_distance_sum",
    "duration_min_sum",
    "speed_mph_sum",
    "total_amount_sum",
]


# Zone centroid lookup (for pre-2017 lat/lon files)
_zone_tree: BallTree | None = None
_zone_ids: np.ndarray | None = None


def _get_zone_tree() -> tuple[BallTree, np.ndarray]:
    """Load zone centroids once and build a BallTree for nearest-zone lookup."""
    global _zone_tree, _zone_ids
    if _zone_tree is None:
        zones = pd.read_csv(ZONE_CENTROIDS_FILE)
        coords_rad = np.radians(zones[["latitude", "longitude"]].values)
        _zone_tree = BallTree(coords_rad, metric="haversine")
        _zone_ids = cast(np.ndarray, zones["location_id"].to_numpy())
    assert _zone_ids is not None
    return _zone_tree, _zone_ids


def _snap_to_zone(lats: pd.Series, lons: pd.Series) -> pd.Series:
    """Map pickup lat/lon to the nearest TLC zone ID.

    Rounds to 3 decimal places (~100 m) before querying so we only run
    BallTree on unique coordinate pairs (~50 K) instead of all raw rows
    (~14 M), making this ~200x faster.
    """
    tree, zone_ids = _get_zone_tree()

    df = pd.DataFrame({"lat_r": lats.round(3), "lon_r": lons.round(3)})
    unique = df.drop_duplicates().copy()
    coords_rad = np.radians(unique[["lat_r", "lon_r"]].values)
    _, indices = tree.query(coords_rad, k=1)
    unique["zone_id"] = zone_ids[indices.flatten()].astype("int32")

    result = df.merge(unique, on=["lat_r", "lon_r"], how="left")
    return pd.Series(result["zone_id"].values, index=lats.index, dtype="int32")


# Data loading + aggregation
def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


_PICKUP_CANDIDATES = ["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"]
_DROPOFF_CANDIDATES = [
    "tpep_dropoff_datetime",
    "lpep_dropoff_datetime",
    "dropoff_datetime",
]
_LOCATION_CANDIDATES = ["PULocationID", "pu_location_id"]


def _needed_columns(file_cols: set[str]) -> list[str] | None:
    """Return the minimal column list to read, or None if file is unusable."""
    pickup = next((c for c in _PICKUP_CANDIDATES if c in file_cols), None)
    dropoff = next((c for c in _DROPOFF_CANDIDATES if c in file_cols), None)
    distance = "trip_distance" if "trip_distance" in file_cols else None
    amount = next((c for c in ["total_amount", "fare_amount"] if c in file_cols), None)
    location = next((c for c in _LOCATION_CANDIDATES if c in file_cols), None)
    has_latlon = "pickup_latitude" in file_cols and "pickup_longitude" in file_cols

    if not all([pickup, dropoff, distance, amount]) or (not location and not has_latlon):
        return None

    assert (
        pickup is not None and dropoff is not None and distance is not None and amount is not None
    )
    cols: list[str] = [pickup, dropoff, distance, amount]
    if location:
        cols.append(location)
    else:
        cols += ["pickup_latitude", "pickup_longitude"]
    return cols


def _aggregate_chunk(raw: pd.DataFrame, fname: str = "") -> pd.DataFrame:
    """Clean one file's rows and aggregate to 10-min window stats per zone.

    - Pre-2017: pickup_longitude / pickup_latitude  (no zone ID)
    - 2017+   : PULocationID                        (zone ID present)
    """
    pickup_col = _first_col(raw, _PICKUP_CANDIDATES)
    dropoff_col = _first_col(raw, _DROPOFF_CANDIDATES)
    distance_col = _first_col(raw, ["trip_distance"])
    amount_col = _first_col(raw, ["total_amount", "fare_amount"])
    location_col = _first_col(raw, _LOCATION_CANDIDATES)
    lat_col = _first_col(raw, ["pickup_latitude"])
    lon_col = _first_col(raw, ["pickup_longitude"])

    has_zone_id = location_col is not None
    has_latlon = lat_col is not None and lon_col is not None

    missing = [
        n
        for n, c in [
            ("pickup", pickup_col),
            ("dropoff", dropoff_col),
            ("distance", distance_col),
            ("amount", amount_col),
        ]
        if c is None
    ]
    if missing or (not has_zone_id and not has_latlon):
        print(f"  SKIP {fname}: missing required columns {missing or ['location/latlon']}")
        return pd.DataFrame()

    assert pickup_col is not None and dropoff_col is not None
    assert distance_col is not None and amount_col is not None

    # Work with Series directly to avoid copying the full DataFrame
    pickup_ts = cast(pd.Series, pd.to_datetime(raw[pickup_col], errors="coerce"))
    dropoff_ts = cast(pd.Series, pd.to_datetime(raw[dropoff_col], errors="coerce"))
    trip_distance = cast(pd.Series, pd.to_numeric(raw[distance_col], errors="coerce"))
    total_amount = cast(pd.Series, pd.to_numeric(raw[amount_col], errors="coerce"))
    duration_min = cast(pd.Series, (dropoff_ts - pickup_ts).dt.total_seconds() / 60)
    speed_mph = cast(pd.Series, trip_distance / (duration_min / 60).replace(0, float("nan")))

    if has_zone_id:
        assert location_col is not None
        pu_location_id = cast(pd.Series, pd.to_numeric(raw[location_col], errors="coerce"))
        location_mask = cast(pd.Series, pu_location_id > 0)
    else:
        # Pre-2017: filter to NYC bounds then snap to nearest zone
        assert lat_col is not None and lon_col is not None
        lat = pd.to_numeric(raw[lat_col], errors="coerce")
        lon = pd.to_numeric(raw[lon_col], errors="coerce")
        lat_series = cast(pd.Series, lat)
        lon_series = cast(pd.Series, lon)
        location_mask = cast(
            pd.Series,
            lat_series.between(_NYC_LAT[0], _NYC_LAT[1])
            & lon_series.between(_NYC_LON[0], _NYC_LON[1]),
        )
        if not location_mask.any():
            return pd.DataFrame()
        # Only pass the valid lat/lon rows to BallTree — no full-DataFrame copy
        pu_location_id = pd.Series(np.nan, index=raw.index, dtype="float64")
        pu_location_id[location_mask] = _snap_to_zone(
            cast(pd.Series, lat_series[location_mask]),
            cast(pd.Series, lon_series[location_mask]),
        ).to_numpy()
        del lat, lon

    mask = (
        pickup_ts.notna()
        & dropoff_ts.notna()
        & (duration_min > 1)
        & (duration_min < 180)
        & (trip_distance > 0.2)
        & (trip_distance < 100)
        & (speed_mph > 1)
        & (speed_mph < 80)
        & location_mask
        & (total_amount > 0)
        & (total_amount < 500)
    )

    # Assemble only the rows that pass the filter — one small DataFrame instead
    # of copying the full chunk multiple times
    idx = cast(pd.Series, mask[mask]).index
    if idx.empty:
        return pd.DataFrame()

    window_start = cast(pd.Series, pickup_ts[idx]).dt.floor("10min")
    filtered = pd.DataFrame(
        {
            "window_start": window_start,
            "window_end": window_start + WINDOW_SIZE,
            "pu_location_id": cast(pd.Series, pu_location_id[idx]).to_numpy(),
            "trip_distance": cast(pd.Series, trip_distance[idx]).to_numpy(),
            "duration_min": cast(pd.Series, duration_min[idx]).to_numpy(),
            "speed_mph": cast(pd.Series, speed_mph[idx]).to_numpy(),
            "total_amount": cast(pd.Series, total_amount[idx]).to_numpy(),
        }
    )
    del (
        pickup_ts,
        dropoff_ts,
        trip_distance,
        total_amount,
        duration_min,
        speed_mph,
        pu_location_id,
        mask,
    )

    agg = (
        filtered.groupby(["window_start", "window_end", "pu_location_id"])
        .agg(
            trip_count=("trip_distance", "count"),
            trip_distance_sum=("trip_distance", "sum"),
            duration_min_sum=("duration_min", "sum"),
            speed_mph_sum=("speed_mph", "sum"),
            total_amount_sum=("total_amount", "sum"),
        )
        .reset_index()
    )
    del filtered

    agg["window_start"] = pd.to_datetime(agg["window_start"], utc=True, errors="coerce")
    agg["window_end"] = pd.to_datetime(agg["window_end"], utc=True, errors="coerce")
    agg = agg.dropna(subset=["window_start", "window_end"])
    return agg


def _collapse_aggregate_stats(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame(columns=pd.Index([*_AGG_KEYS, *_AGG_SUM_COLS]))

    merged = pd.concat(chunks, ignore_index=True)
    collapsed = cast(
        pd.DataFrame,
        merged.groupby(_AGG_KEYS, as_index=False)[_AGG_SUM_COLS].sum(),
    )
    return collapsed


def _finalize_features(stats: pd.DataFrame) -> pd.DataFrame:
    if stats.empty:
        return pd.DataFrame()

    out = stats.copy()
    denom = out["trip_count"].replace(0, np.nan)
    out["avg_trip_distance"] = out["trip_distance_sum"] / denom
    out["avg_duration_min"] = out["duration_min_sum"] / denom
    out["avg_speed_mph"] = out["speed_mph_sum"] / denom
    out["avg_total_amount"] = out["total_amount_sum"] / denom

    out = out[
        [
            "window_start",
            "window_end",
            "pu_location_id",
            "trip_count",
            "avg_trip_distance",
            "avg_duration_min",
            "avg_speed_mph",
            "avg_total_amount",
        ]
    ]
    out_df = cast(pd.DataFrame, out)
    return cast(pd.DataFrame, out_df.loc[out_df["avg_total_amount"].notna()])


def load_and_aggregate(raw_dir: str) -> pd.DataFrame:
    """Read raw TLC parquets and compute 10-minute window aggregates per zone."""
    fs, all_files = _build_filesystem(raw_dir)
    if not all_files:
        print(f"No parquet files found in {raw_dir}")
        return pd.DataFrame()

    files = all_files[:MAX_FILES] if MAX_FILES is not None else all_files
    suffix = f" (capped at MAX_FILES={MAX_FILES})" if MAX_FILES is not None else ""
    print(f"Found {len(files)} parquet file(s) in {raw_dir}{suffix}")

    aggregated_files: list[pd.DataFrame] = []
    total_raw = 0

    for i, f in enumerate(files, 1):
        fname = f.split("/")[-1]
        try:
            # Peek at schema — read only the columns we need (~70% less I/O)
            file_cols = set(pq.read_schema(f, filesystem=fs).names)
            needed = _needed_columns(file_cols)
            if needed is None:
                print(f"  [{i:>3}/{len(files)}] SKIP {fname}: unusable schema")
                continue

            n = 0
            file_agg_chunks: list[pd.DataFrame] = []
            parquet_file = pq.ParquetFile(f, filesystem=fs)
            for batch in parquet_file.iter_batches(
                batch_size=PARQUET_BATCH_SIZE,
                columns=needed,
            ):
                chunk = batch.to_pandas()
                n += len(chunk)
                agg = _aggregate_chunk(chunk, fname)
                if not agg.empty:
                    file_agg_chunks.append(agg)
                del chunk, agg

            total_raw += n

            file_agg = _collapse_aggregate_stats(file_agg_chunks)
            if not file_agg.empty:
                aggregated_files.append(file_agg)

            print(f"  [{i:>3}/{len(files)}] {fname}  ({n:,} raw → {len(file_agg):,} agg rows)")
        except Exception as ex:
            print(f"  Skipping {fname}: {ex}")

    if not aggregated_files:
        return pd.DataFrame()

    print(f"\nTotal raw rows processed: {total_raw:,}")
    stats = _collapse_aggregate_stats(aggregated_files)
    result = _finalize_features(stats)
    print(f"Aggregated to {len(result):,} (zone, window) rows")
    return result


# Feature engineering + training  (80 / 20 temporal split)
def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denom = y_true.replace(0, 1e-6)
    return float(((y_true - y_pred).abs() / denom).mean() * 100)


def _xgb_params(n_estimators: int) -> dict:
    return {
        "n_estimators": n_estimators,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 2,
    }


def _train_continued_xgb(
    train_core: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[XGBRegressor, list[dict[str, float]], int, float]:
    """Train a single evolving booster over chronological row batches."""
    if len(train_core) < 1:
        raise ValueError("Continued training requires non-empty train_core data.")

    val_x = val[FARE_FEATURES]
    val_y = cast(pd.Series, val[TARGET])

    history: list[dict[str, float]] = []
    model: Any = None
    best_model: Any = None
    best_batch = 0
    best_val_mae = float("inf")

    n_batches = max(1, (len(train_core) + TRAIN_BATCH_ROWS - 1) // TRAIN_BATCH_ROWS)
    print(
        f"Training in continued mode: {n_batches} batch(es), "
        f"{TRAIN_BATCH_ROWS:,} rows per batch (last may be smaller)"
    )

    for batch_idx, start in enumerate(range(0, len(train_core), TRAIN_BATCH_ROWS), 1):
        end = min(start + TRAIN_BATCH_ROWS, len(train_core))
        batch = train_core.iloc[start:end]
        x_batch = batch[FARE_FEATURES]
        y_batch = cast(pd.Series, batch[TARGET])

        if model is None:
            model = XGBRegressor(**_xgb_params(INIT_TREES))
            model.fit(x_batch, y_batch, eval_set=[(val_x, val_y)], verbose=False)
        else:
            next_model = XGBRegressor(**_xgb_params(TREES_PER_BATCH))
            next_model.fit(
                x_batch,
                y_batch,
                xgb_model=model.get_booster(),
                eval_set=[(val_x, val_y)],
                verbose=False,
            )
            model = next_model

        val_pred = pd.Series(model.predict(val_x), index=val.index)
        val_mae = float(mean_absolute_error(val_y, val_pred))
        val_mape = mape(val_y, val_pred)
        history.append({"batch": float(batch_idx), "val_mae": val_mae, "val_mape": val_mape})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_batch = batch_idx
            best_model = model

        print(
            f"  [batch {batch_idx:>2}/{n_batches}] rows={len(batch):,} "
            f"val_mae={val_mae:.4f} val_mape={val_mape:.2f}%"
        )

    assert model is not None
    selected_model = cast(XGBRegressor, best_model if best_model is not None else model)
    print(
        f"Selected best batch {best_batch} with validation MAE {best_val_mae:.4f} "
        f"for final model artifact"
    )
    return selected_model, history, best_batch, best_val_mae


def train_and_save(df: pd.DataFrame) -> dict:
    work = df.copy()
    work["window_start"] = pd.to_datetime(work["window_start"], utc=True, errors="coerce")
    work = work.dropna(subset=["window_start", TARGET]).sort_values("window_start")
    work = work.drop_duplicates(subset=["pu_location_id", "window_start"], keep="last")

    window_start = cast(pd.Series, work["window_start"])
    work["hour"] = window_start.dt.hour
    work["day_of_week"] = window_start.dt.dayofweek
    work["is_weekend"] = (work["day_of_week"] >= 5).astype(int)

    # Temporal 80/20 split — train on earlier data, test on recent data
    split_idx = int(len(work) * 0.8)
    train, test = work.iloc[:split_idx], work.iloc[split_idx:]

    split_date = test["window_start"].iloc[0].date()
    print(f"\n80/20 temporal split — test set starts {split_date}")
    print(f"Training on {len(train):,} rows, testing on {len(test):,} rows...")

    if len(train) < 10 or len(test) < 5:
        raise ValueError(f"Not enough data: {len(train)} train rows, {len(test)} test rows.")

    # Baseline: mean fare per (zone, hour)
    baseline = train.groupby(["pu_location_id", "hour"])[TARGET].mean().reset_index()
    baseline.columns = pd.Index(["pu_location_id", "hour", "baseline_pred"])
    scored = test.merge(baseline, on=["pu_location_id", "hour"], how="left")
    scored["baseline_pred"] = scored["baseline_pred"].fillna(train[TARGET].mean())
    baseline_mae = float(mean_absolute_error(scored[TARGET], scored["baseline_pred"]))
    baseline_mape_val = mape(
        cast(pd.Series, scored[TARGET]), cast(pd.Series, scored["baseline_pred"])
    )

    # Continued training: keep one evolving booster and update it batch by batch
    val_rows = max(5000, int(len(train) * VALIDATION_FRACTION))
    val_rows = min(val_rows, max(1, len(train) // 3))
    train_core = train.iloc[:-val_rows]
    val = train.iloc[-val_rows:]
    if len(train_core) < 10 or len(val) < 5:
        raise ValueError(
            f"Not enough rows after validation split: {len(train_core)} train_core, {len(val)} val."
        )

    print(
        f"Continued-training split — core train: {len(train_core):,}, "
        f"validation: {len(val):,}, test: {len(test):,}"
    )
    model, val_history, best_batch, best_val_mae = _train_continued_xgb(train_core, val)

    train_preds = pd.Series(model.predict(train[FARE_FEATURES]), index=train.index)
    train_mae = float(mean_absolute_error(train[TARGET], train_preds))
    train_mape = mape(cast(pd.Series, train[TARGET]), cast(pd.Series, train_preds))

    val_preds = pd.Series(model.predict(val[FARE_FEATURES]), index=val.index)
    val_mae = float(mean_absolute_error(val[TARGET], val_preds))
    val_mape = mape(cast(pd.Series, val[TARGET]), cast(pd.Series, val_preds))

    preds = pd.Series(model.predict(test[FARE_FEATURES]), index=test.index)
    model_mae = float(mean_absolute_error(test[TARGET], preds))
    model_mape = mape(cast(pd.Series, test[TARGET]), cast(pd.Series, preds))

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "fare_model": model,
            "fare_features": FARE_FEATURES,
            "trained_at": datetime.now(UTC).isoformat(),
            "training_rows": len(train),
        },
        MODEL_OUTPUT_DIR / "traffic_model.joblib",
    )

    metrics = {
        "generated_at": datetime.now(UTC).isoformat(),
        "training_rows": len(train),
        "testing_rows": len(test),
        "test_period_start": str(split_date),
        "target": TARGET,
        "baseline": {
            "mae": round(baseline_mae, 4),
            "mape": round(baseline_mape_val, 2),
        },
        "xgboost": {
            "train": {"mae": round(train_mae, 4), "mape": round(train_mape, 2)},
            "validation": {"mae": round(val_mae, 4), "mape": round(val_mape, 2)},
            "test": {"mae": round(model_mae, 4), "mape": round(model_mape, 2)},
            "continued_training": {
                "batch_rows": TRAIN_BATCH_ROWS,
                "init_trees": INIT_TREES,
                "trees_per_batch": TREES_PER_BATCH,
                "selected_best_batch": best_batch,
                "selected_best_val_mae": round(best_val_mae, 4),
                "history": [
                    {
                        "batch": int(item["batch"]),
                        "val_mae": round(item["val_mae"], 4),
                        "val_mape": round(item["val_mape"], 2),
                    }
                    for item in val_history
                ],
            },
        },
        "improvement_mae": round(baseline_mae - model_mae, 4),
    }

    with (MODEL_OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    print("StreamCab — Fare Predictor Training")
    print(f"  Raw data : {RAW_DATA_DIR}")
    print(f"  Model out: {MODEL_OUTPUT_DIR}\n")

    df = load_and_aggregate(RAW_DATA_DIR)
    if df.empty:
        print("No usable data found. Exiting.")
        sys.exit(1)

    try:
        metrics = train_and_save(df)
        print("\nTraining complete:")
        print(json.dumps(metrics, indent=2))
        print(f"\nModel saved → {MODEL_OUTPUT_DIR / 'traffic_model.joblib'}")
    except Exception as ex:
        print(f"Training failed: {ex}")
        sys.exit(1)


if __name__ == "__main__":
    main()
