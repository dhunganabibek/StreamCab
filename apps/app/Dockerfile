FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /opt/streamcab

# Java is required for Spark runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ARG EXTRAS="predictor,dashboard"

COPY pyproject.toml .
COPY src/streamcab /opt/streamcab/src/streamcab
RUN uv pip install --system --no-cache ".[$EXTRAS]"

COPY apps /opt/streamcab/apps
COPY data/reference /opt/streamcab/data/reference

EXPOSE 8501
