FROM python:3.12-slim

WORKDIR /opt/streamcab

RUN pip install --no-cache-dir \
    "kafka-python-ng>=2.2.3" \
    "pandas==2.2.2" \
    "pyarrow==16.1.0" \
    "psycopg[binary]>=3.1" \
    "numpy>=1.26"

COPY apps/producer /opt/streamcab/apps/producer
COPY data/reference /opt/streamcab/data/reference
