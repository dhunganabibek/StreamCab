# StreamCab

StreamCab turns historical taxi parquet data into a live pipeline with Kafka, Spark, model inference, and a Streamlit dashboard.

## Architecture

This project runs as 6 services:

1. `kafka` (Kafka broker + producer)
2. `postgres` (state store)
3. `app` (spark + predictor + dashboard)
4. `consumer` (raw Kafka consumer -> Postgres `live_trips`)
5. `train-once` (manual one-shot training job)
6. `control-center` (Kafka UI)

The `app` container runs three processes:

1. Spark streaming pipeline (Kafka -> Postgres `traffic_agg`)
2. Realtime predictor (`traffic_agg` -> `predictions`)
3. Streamlit dashboard

The `consumer` container runs:

1. Raw Kafka consumer (Kafka -> Postgres `live_trips`)

## Quick Start

### Install all the dependencies and set precommit hooks

```bash
uv sync --all-extras
uv run pre-commit install
```

### Configure environment

```bash
cp .env.example .env
```

Local mode defaults are already set.

- Download and runtime both use the same local host folder: `RAW_DATA_HOST_PATH` (default: `./data/raw-data`).
- `train-once` uses local data by default.
- For S3 training, run `train-once` with a one-time override:
  `docker compose run --rm -e RAW_DATA_DIR=s3://your-bucket/streamcab/parquet train-once`

### Download a small data slice

This API has rate limiting, so use small date ranges.

```bash
python scripts/download_tlc_data.py --output data/raw-data/parquet --start 2022-01 --end 2022-03
```

### Train model once

```bash
docker compose run --rm train-once
```

Expected outputs:

- `models/traffic_model.joblib`
- `models/metrics.json`

### Start runtime stack

```bash
docker compose up --build -d
```

Open dashboard:

- http://localhost:8501

Check the kafka UI:

- http://localhost:9021/

You should see a consumer group named `streamcab-live-consumer` after startup.

## Useful Commands

```bash
# Start core services

docker compose up --build -d

# Tail app logs

docker compose logs -f app

# kafka + producer logs

docker compose logs -f kafka

# Stop everything

docker compose down
```

## Notes

- Postgres host port defaults to `5433` to avoid local conflicts.
- One topic setting is enough: `KAFKA_TOPIC` is used by both producer and stream processor.
- The app and producer read data from `/opt/streamcab/data/raw-data/parquet` in containers.
- `RAW_DATA_HOST_PATH` controls what host folder is mounted to that container path.
- `RAW_DATA_DIR` is an optional `train-once` override (for example `s3://...`).
- `train-once` runs only when invoked manually:
  `docker compose run --rm train-once`

## Diagram

![Architectural Diagram](./assets/architecture.png)
