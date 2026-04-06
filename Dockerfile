FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS base
WORKDIR /opt/streamcab
COPY pyproject.toml .

FROM base AS producer
RUN uv pip install --system --no-cache ".[producer]"
COPY apps/producer /opt/streamcab/apps/producer
CMD ["python", "-u", "/opt/streamcab/apps/producer/producer.py"]

FROM base AS dashboard
RUN uv pip install --system --no-cache ".[dashboard]"
COPY apps/dashboard /opt/streamcab/apps/dashboard
EXPOSE 8501
CMD ["streamlit", "run", "/opt/streamcab/apps/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM base AS trainer
RUN uv pip install --system --no-cache ".[trainer]"
COPY apps/trainer /opt/streamcab/apps/trainer
CMD ["python", "-u", "/opt/streamcab/apps/trainer/train_models.py"]

FROM base AS predictor
RUN uv pip install --system --no-cache ".[predictor]"
COPY apps/predictor /opt/streamcab/apps/predictor
CMD ["python", "-u", "/opt/streamcab/apps/predictor/predict_realtime.py"]
