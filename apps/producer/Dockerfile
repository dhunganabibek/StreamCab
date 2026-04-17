FROM confluentinc/cp-kafka:7.6.1

WORKDIR /opt/streamcab

USER root
RUN python3.9 -m pip install --no-cache-dir --upgrade pip \
	&& python3.9 -m pip install --no-cache-dir \
	   "kafka-python==2.0.2" \
	   "pandas==2.2.2" \
	   "pyarrow==16.1.0" \
	   "psycopg[binary]>=3.1" \
	   "numpy>=1.26"

COPY apps/producer /opt/streamcab/apps/producer
COPY data/reference /opt/streamcab/data/reference

USER appuser
