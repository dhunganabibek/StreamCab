import json
import os
import time
from datetime import datetime
from typing import Any

import psycopg
from kafka import KafkaConsumer

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "taxi-trips")
KAFKA_CONSUMER_GROUP_ID = os.getenv("KAFKA_CONSUMER_GROUP_ID", "streamcab-live-consumer")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://streamcab:streamcab@postgres:5432/streamcab")
INSERT_BATCH_SIZE = int(os.getenv("LIVE_TRIPS_INSERT_BATCH_SIZE", "200"))
INSERT_FLUSH_INTERVAL_SECONDS = float(os.getenv("LIVE_TRIPS_FLUSH_INTERVAL_SECONDS", "1.0"))
LIVE_TRIPS_RETENTION_MINUTES = int(os.getenv("LIVE_TRIPS_RETENTION_MINUTES", "10"))


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def wait_for_postgres() -> None:
    while True:
        try:
            with psycopg.connect(DATABASE_URL):
                print("Postgres ready for raw trip consumer.")
                return
        except Exception:
            print("Postgres not ready for raw trip consumer, retrying in 3s...")
            time.sleep(3)


def build_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_CONSUMER_GROUP_ID,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        consumer_timeout_ms=1000,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )


def insert_rows(rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        return

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO live_trips (
                    pickup_datetime,
                    dropoff_datetime,
                    passenger_count,
                    trip_distance,
                    pu_location_id,
                    do_location_id,
                    total_amount,
                    emitted_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
            # Keep the live table bounded so dashboard reads stay fast.
            cur.execute(
                "DELETE FROM live_trips WHERE emitted_at < NOW() - (%s * INTERVAL '1 minute')",
                (LIVE_TRIPS_RETENTION_MINUTES,),
            )
        conn.commit()


def to_row(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _parse_ts(payload.get("pickup_datetime")),
        _parse_ts(payload.get("dropoff_datetime")),
        payload.get("passenger_count"),
        payload.get("trip_distance"),
        payload.get("pu_location_id"),
        payload.get("do_location_id"),
        payload.get("total_amount"),
        _parse_ts(payload.get("emitted_at")),
    )


def main() -> None:
    print(f"Raw trip consumer starting. topic={KAFKA_TOPIC} group_id={KAFKA_CONSUMER_GROUP_ID}")
    wait_for_postgres()
    consumer = build_consumer()

    buffer: list[tuple[Any, ...]] = []
    last_flush = time.monotonic()
    try:
        while True:
            had_messages = False
            for message in consumer:
                had_messages = True
                payload = message.value if isinstance(message.value, dict) else {}
                buffer.append(to_row(payload))

                if len(buffer) >= INSERT_BATCH_SIZE:
                    insert_rows(buffer)
                    consumer.commit()
                    print(f"Inserted {len(buffer)} live trip rows.")
                    buffer.clear()
                    last_flush = time.monotonic()

            # Flush partial batches periodically so finite bursts appear in UI promptly.
            now = time.monotonic()
            if buffer and ((now - last_flush) >= INSERT_FLUSH_INTERVAL_SECONDS or not had_messages):
                insert_rows(buffer)
                consumer.commit()
                print(f"Inserted {len(buffer)} live trip rows.")
                buffer.clear()
                last_flush = now
    except KeyboardInterrupt:
        pass
    finally:
        if buffer:
            insert_rows(buffer)
            consumer.commit()
            print(f"Inserted final {len(buffer)} live trip rows.")
        consumer.close()


if __name__ == "__main__":
    main()
