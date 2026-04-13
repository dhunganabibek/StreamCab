import os

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
INPUT_TOPIC = os.getenv("INPUT_TOPIC", "taxi-trips")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/opt/streamcab/data/checkpoints/traffic_agg")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/opt/streamcab/data/processed/traffic_agg")


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
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .option("kafka.group.id", "streamcab-spark")  # visible by name in Control Center
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

    (batch_df.coalesce(1).write.mode("append").partitionBy("service_date").parquet(OUTPUT_DIR))


def main() -> None:
    spark = build_session()
    parsed = parse_stream(spark)
    cleaned = clean_stream(parsed)
    aggregated = aggregate(cleaned)

    query = (
        aggregated.writeStream.outputMode("update")
        .option("checkpointLocation", CHECKPOINT_DIR)
        .foreachBatch(write_batch)
        .trigger(processingTime="2 seconds")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
