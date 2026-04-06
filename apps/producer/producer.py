import csv
import json
import os
import random
import time
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaTimeoutError, NoBrokersAvailable


BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
TOPIC = os.getenv("KAFKA_TOPIC", "taxi-trips")
DATA_FILE = os.getenv("DATA_FILE", "")
DATA_DIR = os.getenv("DATA_DIR", "/opt/streamcab/data/raw")
REPLAY_SLEEP_SECONDS = float(os.getenv("REPLAY_SLEEP_SECONDS", "0.15"))


REQUIRED_COLUMNS = {
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "total_amount",
}

SOURCE_ALIASES = {
    "pickup": ["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"],
    "dropoff": ["tpep_dropoff_datetime", "lpep_dropoff_datetime", "dropoff_datetime"],
    "passenger_count": ["passenger_count"],
    "trip_distance": ["trip_distance"],
    "pu_location": ["PULocationID", "pu_location_id"],
    "do_location": ["DOLocationID", "do_location_id"],
    "total_amount": ["total_amount", "fare_amount"],
}


def wait_for_kafka() -> KafkaProducer:
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                linger_ms=20,
                acks=1,
            )
            return producer
        except NoBrokersAvailable:
            print("Kafka is not ready yet. Retrying in 3 seconds...")
            time.sleep(3)


def _read_csv_rows(file_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(file_path):
        return []

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []

        if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames)):
            missing = REQUIRED_COLUMNS.difference(set(reader.fieldnames))
            print(f"Missing columns in CSV ({file_path}): {sorted(missing)}")
            return []

        return list(reader)


def _pick_column(row: Dict[str, object], aliases: List[str], default: object = "") -> object:
    for col in aliases:
        if col in row and row[col] is not None:
            return row[col]
    return default


def _normalize_row(row: Dict[str, object]) -> Dict[str, str]:
    pickup = _pick_column(row, SOURCE_ALIASES["pickup"])
    dropoff = _pick_column(row, SOURCE_ALIASES["dropoff"])
    passenger_count = _pick_column(row, SOURCE_ALIASES["passenger_count"], 1)
    trip_distance = _pick_column(row, SOURCE_ALIASES["trip_distance"], 1.0)
    pu_location = _pick_column(row, SOURCE_ALIASES["pu_location"], 132)
    do_location = _pick_column(row, SOURCE_ALIASES["do_location"], 161)
    total_amount = _pick_column(row, SOURCE_ALIASES["total_amount"], 10.0)

    return {
        "tpep_pickup_datetime": str(pickup),
        "tpep_dropoff_datetime": str(dropoff),
        "passenger_count": str(passenger_count),
        "trip_distance": str(trip_distance),
        "PULocationID": str(pu_location),
        "DOLocationID": str(do_location),
        "total_amount": str(total_amount),
    }


def _span_from_df(df: pd.DataFrame) -> timedelta:
    """Compute data time span from a DataFrame using percentiles to exclude outliers."""
    pickup_col = next((c for c in SOURCE_ALIASES["pickup"] if c in df.columns), None)
    dropoff_col = next((c for c in SOURCE_ALIASES["dropoff"] if c in df.columns), None)
    if not pickup_col or not dropoff_col:
        return timedelta(days=31)
    try:
        pickups = pd.to_datetime(df[pickup_col], errors="coerce").dropna()
        dropoffs = pd.to_datetime(df[dropoff_col], errors="coerce").dropna()
        if pickups.empty or dropoffs.empty:
            return timedelta(days=31)
        # Use 1st/99th percentile to exclude timestamp outliers in raw taxi data
        n_p = len(pickups)
        n_d = len(dropoffs)
        p_min = pickups.sort_values().iloc[int(n_p * 0.01)]
        p_max = dropoffs.sort_values().iloc[min(n_d - 1, int(n_d * 0.99))]
        if p_max > p_min:
            return (p_max - p_min) + timedelta(minutes=1)
    except Exception:
        pass
    return timedelta(days=31)


def _read_parquet_rows(file_path: str) -> Tuple[List[Dict[str, str]], timedelta]:
    """Read parquet file and return (rows, loop_span).

    Span is computed from the DataFrame before converting to string dicts —
    this avoids re-parsing 2M+ timestamps from strings later.
    """
    if not os.path.exists(file_path):
        return [], timedelta(days=31)

    try:
        df = pd.read_parquet(file_path)
    except Exception as ex:  # noqa: BLE001
        print(f"Failed to read parquet file {file_path}: {ex}")
        return [], timedelta(days=31)

    if df.empty:
        return [], timedelta(days=31)

    span = _span_from_df(df)

    rows = [_normalize_row(record) for record in df.to_dict(orient="records")]
    rows = [
        r
        for r in rows
        if r["tpep_pickup_datetime"]
        and r["tpep_dropoff_datetime"]
        and r["PULocationID"]
        and r["DOLocationID"]
    ]
    return rows, span


def _span_from_rows(rows: List[Dict[str, str]]) -> timedelta:
    """Compute span from string-dict rows (used for CSV files, usually much smaller)."""
    try:
        pickup_strs = [r["tpep_pickup_datetime"] for r in rows if r.get("tpep_pickup_datetime")]
        dropoff_strs = [r["tpep_dropoff_datetime"] for r in rows if r.get("tpep_dropoff_datetime")]
        if pickup_strs and dropoff_strs:
            pickups = pd.to_datetime(pickup_strs, errors="coerce").dropna().tolist()
            dropoffs = pd.to_datetime(dropoff_strs, errors="coerce").dropna().tolist()
            pickup_datetimes = [p.to_pydatetime() for p in pickups if isinstance(p, pd.Timestamp)]
            dropoff_datetimes = [d.to_pydatetime() for d in dropoffs if isinstance(d, pd.Timestamp)]
            if pickup_datetimes and dropoff_datetimes:
                pickup_min = min(pickup_datetimes)
                dropoff_max = max(dropoff_datetimes)
                return (dropoff_max - pickup_min) + timedelta(minutes=1)
    except Exception:
        pass
    return timedelta(hours=2)


def resolve_input_file() -> str:
    if DATA_FILE:
        return DATA_FILE

    parquet_candidates = sorted(glob(str(Path(DATA_DIR) / "*.parquet")))
    if parquet_candidates:
        return parquet_candidates[0]

    csv_candidates = sorted(glob(str(Path(DATA_DIR) / "*.csv")))
    if csv_candidates:
        return csv_candidates[0]

    return ""


def _load_input() -> Tuple[List[Dict[str, str]], timedelta]:
    """Load all rows and compute replay span before connecting to Kafka.

    Separating data loading from producer creation ensures the Kafka connection
    is established fresh right before streaming starts, not minutes earlier.
    """
    input_file = resolve_input_file()

    if input_file.endswith(".parquet"):
        rows, span = _read_parquet_rows(input_file)
    elif input_file.endswith(".csv"):
        rows = _read_csv_rows(input_file)
        span = _span_from_rows(rows)
    else:
        return [], timedelta(hours=2)

    if rows:
        print(f"Loaded {len(rows)} rows from {input_file}. Streaming in loop mode.")
        print(f"Data time span: {span}. Timestamps will advance each loop.")

    return rows, span


def _synthetic_rows() -> Iterable[Dict[str, str]]:
    zones = [132, 138, 161, 186, 230, 234, 237, 48, 68, 79, 113, 170, 249]

    while True:
        pickup = datetime.utcnow() - timedelta(seconds=random.randint(0, 120))
        duration_min = random.randint(5, 35)
        dropoff = pickup + timedelta(minutes=duration_min)
        distance = round(random.uniform(1.2, 10.2), 2)

        yield {
            "tpep_pickup_datetime": pickup.strftime("%Y-%m-%d %H:%M:%S"),
            "tpep_dropoff_datetime": dropoff.strftime("%Y-%m-%d %H:%M:%S"),
            "passenger_count": str(random.randint(1, 4)),
            "trip_distance": str(distance),
            "PULocationID": str(random.choice(zones)),
            "DOLocationID": str(random.choice(zones)),
            "total_amount": str(round(distance * random.uniform(2.5, 4.2), 2)),
        }


def stream_rows(rows: List[Dict[str, str]], span: timedelta) -> Iterable[Dict[str, str]]:
    """Yield rows in an infinite loop, shifting timestamps forward by span each iteration."""
    loop_offset: timedelta = timedelta()

    while True:
        for row in rows:
            if loop_offset:
                r = row.copy()
                try:
                    pickup = pd.to_datetime(r["tpep_pickup_datetime"]) + loop_offset
                    dropoff = pd.to_datetime(r["tpep_dropoff_datetime"]) + loop_offset
                    r["tpep_pickup_datetime"] = pickup.strftime("%Y-%m-%d %H:%M:%S")
                    r["tpep_dropoff_datetime"] = dropoff.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    r = row
                yield r
            else:
                yield row
        loop_offset += span


def to_event(row: Dict[str, str]) -> Dict[str, object]:
    return {
        "pickup_datetime": row["tpep_pickup_datetime"],
        "dropoff_datetime": row["tpep_dropoff_datetime"],
        "passenger_count": int(float(row["passenger_count"] or 0)),
        "trip_distance": float(row["trip_distance"] or 0),
        "pu_location_id": int(float(row["PULocationID"] or 0)),
        "do_location_id": int(float(row["DOLocationID"] or 0)),
        "total_amount": float(row["total_amount"] or 0),
        "emitted_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }


def main() -> None:
    # Load data BEFORE connecting to Kafka so the producer connection is fresh
    # when streaming begins, not stale after minutes of data loading.
    rows, span = _load_input()

    if rows:
        row_iter: Iterable[Dict[str, str]] = stream_rows(rows, span)
    else:
        print("No valid input parquet/csv found. Falling back to synthetic rows.")
        row_iter = _synthetic_rows()

    producer = wait_for_kafka()
    sent = 0

    for row in row_iter:
        try:
            event = to_event(row)
            producer.send(TOPIC, event)
            sent += 1

            if sent % 500 == 0:
                producer.flush(timeout=30)
                print(f"Sent {sent} events...")
        except (KafkaTimeoutError, NoBrokersAvailable) as ex:
            print(f"Kafka error, reconnecting producer... ({ex})")
            try:
                producer.close()
            except Exception:  # noqa: BLE001
                pass
            producer = wait_for_kafka()
            continue
        except Exception as ex:  # noqa: BLE001
            print(f"Skipping one record due to error: {ex}")
            continue

        time.sleep(REPLAY_SLEEP_SECONDS)


if __name__ == "__main__":
    main()
