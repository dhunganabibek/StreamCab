FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /opt/streamcab

RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN uv pip install --system --no-cache ".[trainer]"

COPY apps/trainer /opt/streamcab/apps/trainer
COPY data/reference /opt/streamcab/data/reference

CMD ["python", "-u", "/opt/streamcab/apps/trainer/train_models.py"]
