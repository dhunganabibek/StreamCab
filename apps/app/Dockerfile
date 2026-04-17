FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /opt/streamcab

# Java is required for Spark runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN uv pip install --system --no-cache ".[predictor,dashboard]" "pyspark==3.5.1" "kafka-python==2.0.2"

COPY apps /opt/streamcab/apps
COPY data/reference /opt/streamcab/data/reference

EXPOSE 8501
