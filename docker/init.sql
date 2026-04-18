-- StreamCab — Postgres schema
-- Runs once on first container start (docker-entrypoint-initdb.d)

CREATE TABLE IF NOT EXISTS live_trips (
    id SERIAL PRIMARY KEY,
    pickup_datetime TIMESTAMP,
    dropoff_datetime TIMESTAMP,
    passenger_count INTEGER,
    trip_distance DOUBLE PRECISION,
    pu_location_id INTEGER,
    do_location_id INTEGER,
    total_amount DOUBLE PRECISION,
    emitted_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS live_trips_emitted_at_idx ON live_trips (emitted_at DESC);

CREATE TABLE IF NOT EXISTS traffic_agg (
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    pu_location_id INTEGER,
    trip_count INTEGER,
    avg_speed_mph DOUBLE PRECISION,
    avg_duration_min DOUBLE PRECISION,
    avg_trip_distance DOUBLE PRECISION,
    avg_total_amount DOUBLE PRECISION,
    anomaly_count INTEGER,
    service_date DATE
);

CREATE INDEX IF NOT EXISTS traffic_agg_window_idx ON traffic_agg (window_start, pu_location_id);

CREATE UNIQUE INDEX IF NOT EXISTS traffic_agg_unique_idx ON traffic_agg (
    window_start,
    window_end,
    pu_location_id
);

CREATE INDEX IF NOT EXISTS traffic_agg_zone_window_end_idx ON traffic_agg (
    pu_location_id,
    window_end DESC
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    pu_location_id INTEGER,
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    trip_count INTEGER,
    avg_speed_mph DOUBLE PRECISION,
    avg_trip_distance DOUBLE PRECISION,
    predicted_avg_fare DOUBLE PRECISION,
    predicted_tip_low DOUBLE PRECISION,
    predicted_tip_high DOUBLE PRECISION,
    surge_multiplier DOUBLE PRECISION,
    prediction_for_window_start TIMESTAMP,
    prediction_for_window_end TIMESTAMP,
    prediction_generated_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS predictions_zone_time_idx ON predictions (
    pu_location_id,
    prediction_generated_at DESC
);