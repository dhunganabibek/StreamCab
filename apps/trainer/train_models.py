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
from typing import cast

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


# Storage abstraction — local path, symlink, or S3 all look the same
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
        _zone_ids = zones["location_id"].values
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

    cols = [pickup, dropoff, distance, amount]  # type: ignore[list-item]
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

    # Work with Series directly to avoid copying the full DataFrame
    pickup_ts = pd.to_datetime(raw[pickup_col], errors="coerce")
    dropoff_ts = pd.to_datetime(raw[dropoff_col], errors="coerce")
    trip_distance = pd.to_numeric(raw[distance_col], errors="coerce")
    total_amount = pd.to_numeric(raw[amount_col], errors="coerce")
    duration_min = (dropoff_ts - pickup_ts).dt.total_seconds() / 60
    speed_mph = trip_distance / (duration_min / 60).replace(0, float("nan"))

    if has_zone_id:
        pu_location_id = pd.to_numeric(raw[location_col], errors="coerce")
        location_mask = pu_location_id > 0
    else:
        # Pre-2017: filter to NYC bounds then snap to nearest zone
        lat = pd.to_numeric(raw[lat_col], errors="coerce")
        lon = pd.to_numeric(raw[lon_col], errors="coerce")
        location_mask = lat.between(_NYC_LAT[0], _NYC_LAT[1]) & lon.between(
            _NYC_LON[0], _NYC_LON[1]
        )
        if not location_mask.any():
            return pd.DataFrame()
        # Only pass the valid lat/lon rows to BallTree — no full-DataFrame copy
        pu_location_id = pd.Series(np.nan, index=raw.index, dtype="float64")
        pu_location_id[location_mask] = _snap_to_zone(lat[location_mask], lon[location_mask]).values
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
    idx = mask[mask].index
    if idx.empty:
        return pd.DataFrame()

    window_start = pickup_ts[idx].dt.floor("10min")
    filtered = pd.DataFrame(
        {
            "window_start": window_start,
            "window_end": window_start + WINDOW_SIZE,
            "pu_location_id": pu_location_id[idx].values,
            "trip_distance": trip_distance[idx].values,
            "duration_min": duration_min[idx].values,
            "speed_mph": speed_mph[idx].values,
            "total_amount": total_amount[idx].values,
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
            avg_trip_distance=("trip_distance", "mean"),
            avg_duration_min=("duration_min", "mean"),
            avg_speed_mph=("speed_mph", "mean"),
            avg_total_amount=("total_amount", "mean"),
        )
        .reset_index()
    )
    del filtered

    agg["window_start"] = pd.to_datetime(agg["window_start"]).dt.tz_localize("UTC")
    agg["window_end"] = pd.to_datetime(agg["window_end"]).dt.tz_localize("UTC")
    return agg


def load_and_aggregate(raw_dir: str) -> pd.DataFrame:
    """Read raw TLC parquets and compute 10-minute window aggregates per zone."""
    fs, all_files = _build_filesystem(raw_dir)
    if not all_files:
        print(f"No parquet files found in {raw_dir}")
        return pd.DataFrame()

    files = all_files[:MAX_FILES] if MAX_FILES is not None else all_files
    suffix = f" (capped at MAX_FILES={MAX_FILES})" if MAX_FILES is not None else ""
    print(f"Found {len(files)} parquet file(s) in {raw_dir}{suffix}")

    agg_chunks: list[pd.DataFrame] = []
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

            chunk = pq.read_table(f, filesystem=fs, columns=needed).to_pandas()
            n = len(chunk)
            total_raw += n
            agg = _aggregate_chunk(chunk, fname)
            del chunk
            if not agg.empty:
                agg_chunks.append(agg)
            print(f"  [{i:>3}/{len(files)}] {fname}  ({n:,} raw → {len(agg):,} agg rows)")
        except Exception as ex:
            print(f"  Skipping {fname}: {ex}")

    if not agg_chunks:
        return pd.DataFrame()

    print(f"\nTotal raw rows processed: {total_raw:,}")
    result = pd.concat(agg_chunks, ignore_index=True)
    print(f"Aggregated to {len(result):,} (zone, window) rows")
    return result


# Feature engineering + training  (80 / 20 temporal split)
def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denom = y_true.replace(0, 1e-6)
    return float(((y_true - y_pred).abs() / denom).mean() * 100)


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

    # XGBoost fare predictor
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )
    model.fit(train[FARE_FEATURES], train[TARGET])

    train_preds = pd.Series(model.predict(train[FARE_FEATURES]), index=train.index)
    train_mae = float(mean_absolute_error(train[TARGET], train_preds))
    train_mape = mape(cast(pd.Series, train[TARGET]), cast(pd.Series, train_preds))

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
            "test": {"mae": round(model_mae, 4), "mape": round(model_mape, 2)},
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
