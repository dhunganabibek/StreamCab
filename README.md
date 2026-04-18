# StreamCab

StreamCab turns historical NYC taxi parquet data into a live streaming pipeline with Kafka, Spark Structured Streaming, ML inference, and a Streamlit dashboard.

## Architecture

![Architectural Diagram](./assets/architecture.png)

### Services

| Service          | Container                  | What it does                                      |
| ---------------- | -------------------------- | ------------------------------------------------- |
| `postgres`       | `streamcab-postgres`       | State store (live trips, aggregates, predictions) |
| `kafka`          | `streamcab-kafka`          | KRaft broker + producer replaying parquet data    |
| `app`            | `streamcab-app`            | Spark streaming + predictor + Streamlit dashboard |
| `consumer`       | `streamcab-consumer`       | Kafka → Postgres `live_trips` writer              |
| `control-center` | `streamcab-control-center` | Confluent Kafka UI                                |
| `train-once`     | _(manual)_                 | One-shot model training job                       |

The `app` container runs three processes concurrently:

1. **Spark** — reads from Kafka, aggregates into `traffic_agg`
2. **Predictor** — reads `traffic_agg`, writes predictions to `predictions`
3. **Dashboard** — Streamlit on port `8501`

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose v2)
- [uv](https://docs.astral.sh/uv/) — only needed for local dev / pre-commit hooks

## Quick Start

### 1. Clone and install dev tools

```bash
git clone git@github.com:dhunganabibek/StreamCab.git && cd StreamCab
uv sync --all-extras
uv run pre-commit install
```

### 2. Configure environment

```bash
cp .env.example .env
```

All defaults work for local development. Edit `.env` only if you need to override something (e.g. a different Postgres host port).

### 3. Download data

This API has rate limiting — keep date ranges small.

```bash
uv run python scripts/download_tlc_data.py \
  --output data/raw-data/parquet \
  --start 2022-01 --end 2022-03
```

### 4. Train the model

```bash
docker compose run --rm train-once
```

Expected outputs: `models/tip_model.joblib`, `models/metrics.json`

### 5. Start the stack

```bash
docker compose up --build -d
```

| URL                     | Service                             |
| ----------------------- | ----------------------------------- |
| <http://localhost:8501> | Streamlit dashboard                 |
| <http://localhost:9021> | Confluent Control Center (Kafka UI) |

After startup you should see a consumer group `streamcab-live-consumer` in the Kafka UI.

## Useful Commands

```bash
# Tail app logs (Spark + predictor + dashboard)
docker compose logs -f app

# Tail producer/Kafka logs
docker compose logs -f kafka

# Stop and remove containers
docker compose down

# Stop and also remove volumes (wipes Postgres + Kafka data)
docker compose down -v
```

## Database Quick Info

- Connect from host tools (DBeaver, psql): `postgresql://streamcab:streamcab@localhost:5433/streamcab`
- Connect from containers: `postgresql://streamcab:streamcab@postgres:5432/streamcab`
- `live_trips`: recent raw trip events written by the consumer.
- `traffic_agg`: Spark 10-minute aggregates per pickup zone.
- `predictions`: latest model outputs (fare/tip range/surge) used by the dashboard.

## Environment Variables

All variables have working defaults. See [.env.example](.env.example) for the full list with explanations.

Key overrides:

| Variable                         | Default           | Notes                                |
| -------------------------------- | ----------------- | ------------------------------------ |
| `POSTGRES_HOST_PORT`             | `5433`            | Change if port 5433 is taken locally |
| `RAW_DATA_HOST_PATH`             | `./data/raw-data` | Host path mounted into containers    |
| `REPLAY_SLEEP_SECONDS`           | `0.15`            | Seconds between producer events      |
| `WRITE_LIVE_TRIPS_FROM_PRODUCER` | `false`           | `true` bypasses the consumer service |

## Notes

- Postgres is exposed on **5433** (not 5432) to avoid conflicts with a local Postgres.
- The Kafka broker runs in KRaft mode (no ZooKeeper).
- `train-once` uses the `manual` Compose profile — it only runs when invoked explicitly.
- For S3 training data: `docker compose run --rm -e RAW_DATA_DIR=s3://your-bucket/path train-once`
