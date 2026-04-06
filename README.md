# StreamCab

StreamCab is an end-to-end, real-time analytics project for NYC taxi demand, from ingestion to live predictions and visualization.

## Tools Used

Tools used:  
Python, Kafka, Spark Structured Streaming, XGBoost, Streamlit, Docker Compose, and uv.

## Summary

Real-time NYC taxi traffic pipeline:

- Kafka ingestion (`producer`)
- Spark streaming aggregation (`spark-streaming`)
- Model training (`trainer`)
- Realtime inference (`predictor`)
- Dashboard (`dashboard`)

## Ruining the project

```bash
docker compose up --build
```

Open:

- Dashboard: <http://localhost:8501>
- Control Center: <http://localhost:9021>

## Useful commands

```bash
docker compose up --build -d
docker compose logs -f producer spark-streaming trainer predictor dashboard
docker compose down
```

## Data locations

- Input: [data/raw](data/raw)
- Aggregates: [data/processed](data/processed)
- Checkpoints: [data/checkpoints](data/checkpoints)
- Models and outputs: [data/models](data/models)

Expected files:

- [data/models/metrics.json](data/models/metrics.json)
- [data/models/realtime_predictions.parquet](data/models/realtime_predictions.parquet)

## Download TLC parquet data

```bash
python scripts/download_tlc_data.py --taxi-type yellow --start 2024-01 --end 2024-03 --output data/raw
```
