import os

import psycopg
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    from_json,
    to_date,
    to_timestamp,
    unix_timestamp,
    when,
    window,
)
from pyspark.sql.functions import (
    sum as spark_sum,
)
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
INPUT_TOPIC = os.getenv("KAFKA_TOPIC", os.getenv("INPUT_TOPIC", "taxi-trips"))
KAFKA_STARTING_OFFSETS = os.getenv("KAFKA_STARTING_OFFSETS", "latest")
KAFKA_FAIL_ON_DATA_LOSS = os.getenv("KAFKA_FAIL_ON_DATA_LOSS", "true")
SPARK_CHECKPOINT_DIR = os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/streamcab-checkpoints/traffic_agg")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://streamcab:streamcab@postgres:5432/streamcab")
POSTGRES_TABLE = "traffic_agg"


def ensure_postgres_indexes() -> None:
    """Ensure upsert target has a unique index even on pre-existing databases."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Older database volumes may already contain duplicate aggregate keys.
            # Remove duplicates before enforcing uniqueness for ON CONFLICT upserts.
            cur.execute(
                """
                WITH ranked AS (
                    SELECT ctid,
                           ROW_NUMBER() OVER (
                               PARTITION BY window_start, window_end, pu_location_id
                               ORDER BY ctid
                           ) AS rn
                    FROM traffic_agg
                )
                DELETE FROM traffic_agg t
                USING ranked r
                WHERE t.ctid = r.ctid
                  AND r.rn > 1
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS traffic_agg_unique_idx
                ON traffic_agg (window_start, window_end, pu_location_id)
                """
            )
        conn.commit()


def build_session() -> SparkSession:
    return (
        SparkSession.builder.appName("streamcab-traffic-pipeline")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def input_schema() -> StructType:
    return StructType(
        [
            StructField("pickup_datetime", StringType()),
            StructField("dropoff_datetime", StringType()),
            StructField("passenger_count", IntegerType()),
            StructField("trip_distance", DoubleType()),
            StructField("pu_location_id", IntegerType()),
            StructField("do_location_id", IntegerType()),
            StructField("total_amount", DoubleType()),
            StructField("emitted_at", StringType()),
        ]
    )


def parse_stream(spark: SparkSession) -> DataFrame:
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", INPUT_TOPIC)
        .option("startingOffsets", KAFKA_STARTING_OFFSETS)
        .option("failOnDataLoss", KAFKA_FAIL_ON_DATA_LOSS)
        .load()
    )

    parsed = raw.select(
        from_json(col("value").cast("string"), input_schema()).alias("trip")
    ).select("trip.*")

    return (
        parsed.withColumn("pickup_ts", to_timestamp(col("pickup_datetime")))
        .withColumn("dropoff_ts", to_timestamp(col("dropoff_datetime")))
        .withColumn(
            "duration_min",
            (unix_timestamp(col("dropoff_ts")) - unix_timestamp(col("pickup_ts"))) / 60.0,
        )
        .withColumn("speed_mph", col("trip_distance") / (col("duration_min") / 60.0))
    )


def clean_stream(df: DataFrame) -> DataFrame:
    return df.filter(
        (col("pickup_ts").isNotNull())
        & (col("dropoff_ts").isNotNull())
        & (col("duration_min") > 1)
        & (col("duration_min") < 180)
        & (col("trip_distance") > 0.2)
        & (col("trip_distance") < 100)
        & (col("speed_mph") > 1)
        & (col("speed_mph") < 80)
        & (col("pu_location_id") > 0)
    )


def aggregate(df: DataFrame) -> DataFrame:
    return (
        df.withWatermark("pickup_ts", "5 minutes")
        .groupBy(window(col("pickup_ts"), "10 minutes", "5 minutes"), col("pu_location_id"))
        .agg(
            spark_sum(when((col("speed_mph") < 3) | (col("speed_mph") > 45), 1).otherwise(0)).alias(
                "anomaly_count"
            ),
            avg("speed_mph").alias("avg_speed_mph"),
            avg("duration_min").alias("avg_duration_min"),
            avg("trip_distance").alias("avg_trip_distance"),
            avg("total_amount").alias("avg_total_amount"),
            spark_sum(when(col("speed_mph").isNotNull(), 1).otherwise(0)).alias("trip_count"),
        )
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("pu_location_id"),
            col("trip_count"),
            col("avg_speed_mph"),
            col("avg_duration_min"),
            col("avg_trip_distance"),
            col("avg_total_amount"),
            col("anomaly_count"),
        )
        .withColumn("service_date", to_date(col("window_start")))
    )


def write_batch(batch_df: DataFrame, _: int) -> None:
    if batch_df.isEmpty():
        return

    select_cols = [
        "window_start",
        "window_end",
        "pu_location_id",
        "trip_count",
        "avg_speed_mph",
        "avg_duration_min",
        "avg_trip_distance",
        "avg_total_amount",
        "anomaly_count",
        "service_date",
    ]
    rows = [
        tuple(r[c] for c in select_cols) for r in batch_df.select(*select_cols).toLocalIterator()
    ]
    if not rows:
        return

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {POSTGRES_TABLE} (
                    window_start,
                    window_end,
                    pu_location_id,
                    trip_count,
                    avg_speed_mph,
                    avg_duration_min,
                    avg_trip_distance,
                    avg_total_amount,
                    anomaly_count,
                    service_date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (window_start, window_end, pu_location_id)
                DO UPDATE SET
                    trip_count = EXCLUDED.trip_count,
                    avg_speed_mph = EXCLUDED.avg_speed_mph,
                    avg_duration_min = EXCLUDED.avg_duration_min,
                    avg_trip_distance = EXCLUDED.avg_trip_distance,
                    avg_total_amount = EXCLUDED.avg_total_amount,
                    anomaly_count = EXCLUDED.anomaly_count,
                    service_date = EXCLUDED.service_date
                """,
                rows,
            )
        conn.commit()


def main() -> None:
    ensure_postgres_indexes()
    spark = build_session()
    parsed = parse_stream(spark)
    cleaned = clean_stream(parsed)
    aggregated = aggregate(cleaned)

    query = (
        aggregated.writeStream.outputMode("update")
        .option("checkpointLocation", SPARK_CHECKPOINT_DIR)
        .foreachBatch(write_batch)
        .trigger(processingTime="2 seconds")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
