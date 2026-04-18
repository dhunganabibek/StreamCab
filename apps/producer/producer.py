"""
Kafka producer — replays NYC parquet files in an infinite loop,
advancing timestamps each cycle so the data always looks "live".
By default it only publishes to Kafka.
"""

import os
import time
from collections.abc import Iterable
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from kafka import KafkaProducer
from kafka.errors import KafkaTimeoutError, NoBrokersAvailable

_PROJECT_ROOT = Path(__file__).parents[2]

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
TOPIC = os.getenv("KAFKA_TOPIC", "taxi-trips")
DATA_DIR = os.getenv("DATA_DIR", str(_PROJECT_ROOT / "data/raw-data/parquet"))
REPLAY_SLEEP_SECONDS = float(os.getenv("REPLAY_SLEEP_SECONDS", "0.15"))
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://streamcab:streamcab@postgres:5432/streamcab")
ZONE_CENTROIDS_FILE = os.getenv(
    "ZONE_CENTROIDS_FILE", str(_PROJECT_ROOT / "data/reference/zone_centroids.csv")
)

LIVE_LOG_MAX = 200
LIVE_LOG_FLUSH_INTERVAL = 2.0  # seconds between DB flushes
WRITE_LIVE_TRIPS_FROM_PRODUCER = os.getenv("WRITE_LIVE_TRIPS_FROM_PRODUCER", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

SOURCE_ALIASES = {
    "pickup": ["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"],
    "dropoff": ["tpep_dropoff_datetime", "lpep_dropoff_datetime", "dropoff_datetime"],
    "passenger_count": ["passenger_count"],
    "trip_distance": ["trip_distance"],
    "pu_location": ["PULocationID", "pu_location_id"],
    "do_location": ["DOLocationID", "do_location_id"],
    "pickup_lat": ["pickup_latitude"],
    "pickup_lon": ["pickup_longitude"],
    "dropoff_lat": ["dropoff_latitude"],
    "dropoff_lon": ["dropoff_longitude"],
    "total_amount": ["total_amount", "fare_amount"],
}

# In-memory queue for optional direct DB writes from producer
_live_buffer: list[dict] = []
_last_flush: float = 0.0
_zone_ids: np.ndarray | None = None
_zone_lats: np.ndarray | None = None
_zone_lons: np.ndarray | None = None
_coord_cache: dict[tuple[float, float], int] = {}


# DB helpers
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


def flush_to_postgres() -> None:
    global _last_flush
    now = time.monotonic()
    if now - _last_flush < LIVE_LOG_FLUSH_INTERVAL or not _live_buffer:
        return

    records = list(_live_buffer)
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO live_trips (
                        pickup_datetime, dropoff_datetime, passenger_count,
                        trip_distance, pu_location_id, do_location_id,
                        total_amount, emitted_at
                    ) VALUES (
                        %(pickup_datetime)s, %(dropoff_datetime)s, %(passenger_count)s,
                        %(trip_distance)s, %(pu_location_id)s, %(do_location_id)s,
                        %(total_amount)s, %(emitted_at)s
                    )
                    """,
                    records,
                )
                # Keep only the last 10 minutes of trips
                cur.execute(
                    "DELETE FROM live_trips WHERE emitted_at < NOW() - INTERVAL '10 minutes'"
                )
            conn.commit()
        _live_buffer.clear()
    except Exception as ex:
        print(f"DB flush failed: {ex}")

    _last_flush = now


# Kafka
def wait_for_kafka() -> KafkaProducer:
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=BOOTSTRAP_SERVERS,
                value_serializer=lambda v: __import__("json").dumps(v).encode("utf-8"),
                linger_ms=20,
                acks=1,
            )
            return producer
        except NoBrokersAvailable:
            print("Kafka not ready, retrying in 3 s…")
            time.sleep(3)


# Data loading
_WANTED_COLS = [c for aliases in SOURCE_ALIASES.values() for c in aliases]
BATCH_SIZE = int(os.getenv("PARQUET_BATCH_SIZE", "500"))


def _pick(row: dict, aliases: list[str], default=None):
    for col in aliases:
        if col in row and row[col] is not None:
            return row[col]
    return default


def _to_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _parse_zone_id(value) -> int | None:
    n = _to_float(value)
    if n is None:
        return None
    zid = int(n)
    return zid if zid > 0 else None


def _load_zone_centroids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _zone_ids, _zone_lats, _zone_lons
    if _zone_ids is None or _zone_lats is None or _zone_lons is None:
        try:
            z = pd.read_csv(ZONE_CENTROIDS_FILE)
            _zone_ids = z["location_id"].astype(int).to_numpy()
            _zone_lats = z["latitude"].astype(float).to_numpy()
            _zone_lons = z["longitude"].astype(float).to_numpy()
        except Exception:
            _zone_ids = np.array([161], dtype=int)
            _zone_lats = np.array([40.7549], dtype=float)
            _zone_lons = np.array([-73.9840], dtype=float)
    return _zone_ids, _zone_lats, _zone_lons


def _nearest_zone_id(lat, lon) -> int | None:
    lat_f = _to_float(lat)
    lon_f = _to_float(lon)
    if lat_f is None or lon_f is None:
        return None

    key = (round(lat_f, 3), round(lon_f, 3))
    cached = _coord_cache.get(key)
    if cached is not None:
        return cached

    zone_ids, zone_lats, zone_lons = _load_zone_centroids()
    d2 = (zone_lats - lat_f) ** 2 + (zone_lons - lon_f) ** 2
    idx = int(np.argmin(d2))
    zid = int(zone_ids[idx])
    _coord_cache[key] = zid
    return zid


def _normalize_row(row: dict) -> dict[str, str]:
    pu_id = _parse_zone_id(_pick(row, SOURCE_ALIASES["pu_location"], None))
    do_id = _parse_zone_id(_pick(row, SOURCE_ALIASES["do_location"], None))

    if pu_id is None:
        pu_id = _nearest_zone_id(
            _pick(row, SOURCE_ALIASES["pickup_lat"], None),
            _pick(row, SOURCE_ALIASES["pickup_lon"], None),
        )
    if do_id is None:
        do_id = _nearest_zone_id(
            _pick(row, SOURCE_ALIASES["dropoff_lat"], None),
            _pick(row, SOURCE_ALIASES["dropoff_lon"], None),
        )

    if pu_id is None:
        pu_id = 161
    if do_id is None:
        do_id = 161

    return {
        "tpep_pickup_datetime": str(_pick(row, SOURCE_ALIASES["pickup"], "")),
        "tpep_dropoff_datetime": str(_pick(row, SOURCE_ALIASES["dropoff"], "")),
        "passenger_count": str(_pick(row, SOURCE_ALIASES["passenger_count"], 1)),
        "trip_distance": str(_pick(row, SOURCE_ALIASES["trip_distance"], 1.0)),
        "PULocationID": str(pu_id),
        "DOLocationID": str(do_id),
        "total_amount": str(_pick(row, SOURCE_ALIASES["total_amount"], 10.0)),
    }


def find_data_files() -> list[str]:
    return sorted(glob(str(Path(DATA_DIR) / "*.parquet")))


def stream_file_batched(path: str, time_offset: timedelta) -> Iterable[dict[str, str]]:
    import pyarrow.parquet as pq

    try:
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        cols = [c for c in _WANTED_COLS if c in available] or None
    except Exception as ex:
        print(f"Cannot open {path}: {ex}")
        return

    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=cols):
        df = batch.to_pandas()
        if df.empty:
            continue
        for r in df.to_dict(orient="records"):
            row = _normalize_row(r)
            if not (
                row["tpep_pickup_datetime"]
                and row["tpep_dropoff_datetime"]
                and row["PULocationID"]
                and row["DOLocationID"]
            ):
                continue
            if time_offset:
                try:
                    pickup = pd.to_datetime(row["tpep_pickup_datetime"]) + time_offset
                    dropoff = pd.to_datetime(row["tpep_dropoff_datetime"]) + time_offset
                    row["tpep_pickup_datetime"] = pickup.strftime("%Y-%m-%d %H:%M:%S")
                    row["tpep_dropoff_datetime"] = dropoff.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
            yield row


def to_event(row: dict[str, str]) -> dict:
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


#  Synthetic fallback
def synthetic_rows() -> Iterable[dict[str, str]]:
    import random

    zones = [132, 138, 161, 186, 230, 234, 237, 48, 68, 79, 113, 170, 249]
    while True:
        pickup = datetime.utcnow() - timedelta(seconds=random.randint(0, 120))
        duration = random.randint(5, 35)
        distance = round(random.uniform(1.2, 10.2), 2)
        yield {
            "tpep_pickup_datetime": pickup.strftime("%Y-%m-%d %H:%M:%S"),
            "tpep_dropoff_datetime": (pickup + timedelta(minutes=duration)).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "passenger_count": str(random.randint(1, 4)),
            "trip_distance": str(distance),
            "PULocationID": str(random.choice(zones)),
            "DOLocationID": str(random.choice(zones)),
            "total_amount": str(round(distance * random.uniform(2.5, 4.2), 2)),
        }


def rotating_rows() -> Iterable[dict[str, str]]:
    data_files = find_data_files()
    if not data_files:
        print(f"No parquet files found in {DATA_DIR}. Using synthetic data.")
        yield from synthetic_rows()
        return

    print(f"Found {len(data_files)} parquet file(s) — streaming in {BATCH_SIZE}-row batches.")
    time_offset = timedelta()
    cycle = 0

    while True:
        for path in data_files:
            fname = Path(path).name
            print(f"  Streaming {fname} (cycle {cycle})")
            yield from stream_file_batched(path, time_offset)
            time_offset += timedelta(days=31)
        cycle += 1


# Main
def main() -> None:
    if WRITE_LIVE_TRIPS_FROM_PRODUCER:
        wait_for_db()
        print("Producer direct DB logging enabled.")
    else:
        print("Producer direct DB logging disabled (consumer writes live_trips).")

    producer = wait_for_kafka()
    sent = 0

    for row in rotating_rows():
        try:
            event = to_event(row)
            producer.send(TOPIC, event)
            if WRITE_LIVE_TRIPS_FROM_PRODUCER:
                _live_buffer.append(event)
                if len(_live_buffer) > LIVE_LOG_MAX:
                    # Keep memory bounded if DB is temporarily unavailable.
                    _live_buffer[:] = _live_buffer[-LIVE_LOG_MAX:]
                flush_to_postgres()
            sent += 1

            if sent % 500 == 0:
                producer.flush(timeout=30)
                print(f"Sent {sent:,} events…")

        except (KafkaTimeoutError, NoBrokersAvailable) as ex:
            print(f"Kafka error, reconnecting… ({ex})")
            try:
                producer.close()
            except Exception:
                pass
            producer = wait_for_kafka()
            continue
        except Exception as ex:
            print(f"Skipping record: {ex}")
            continue

        time.sleep(REPLAY_SLEEP_SECONDS)


if __name__ == "__main__":
    main()
