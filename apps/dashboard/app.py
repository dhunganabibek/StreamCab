"""StreamCab — Live NYC Taxi Traffic Analytics Dashboard.

Shows real-time Kafka→Spark streaming aggregates alongside XGBoost demand
forecasts, surge pricing multipliers, and tip/fare estimates.
"""

import json
import os
import time
from pathlib import Path
from typing import cast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


AGGREGATE_DIR = Path(os.getenv("AGGREGATE_DIR", "/opt/streamcab/data/processed/traffic_agg"))
METRICS_FILE = Path(os.getenv("METRICS_FILE", "/opt/streamcab/data/models/metrics.json"))
PREDICTIONS_FILE = Path(
    os.getenv("PREDICTIONS_FILE", "/opt/streamcab/data/models/realtime_predictions.parquet")
)
ZONE_CENTROIDS_FILE = Path(
    os.getenv("ZONE_CENTROIDS_FILE", "/opt/streamcab/data/reference/zone_centroids.csv")
)

AUTO_REFRESH_SECONDS = 15


@st.cache_data(ttl=AUTO_REFRESH_SECONDS)
def load_aggregates() -> pd.DataFrame:
    if not AGGREGATE_DIR.exists() or not list(AGGREGATE_DIR.rglob("*.parquet")):
        return pd.DataFrame()

    df = pd.read_parquet(AGGREGATE_DIR)
    if df.empty:
        return df

    df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
    df["window_end"] = pd.to_datetime(df["window_end"], utc=True, errors="coerce")
    df = df.dropna(subset=["window_start", "window_end"])

    dedup_keys = ["pu_location_id", "window_start"]
    if set(dedup_keys).issubset(df.columns):
        df = df.drop_duplicates(subset=dedup_keys, keep="last")

    return df.sort_values(by="window_start")


@st.cache_data(ttl=60)
def load_metrics() -> dict:
    if not METRICS_FILE.exists():
        return {}
    with METRICS_FILE.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=AUTO_REFRESH_SECONDS)
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists():
        return pd.DataFrame()

    pred = pd.read_parquet(PREDICTIONS_FILE)
    if pred.empty:
        return pred

    for col_name in [
        "prediction_for_window_start",
        "prediction_for_window_end",
        "prediction_generated_at",
    ]:
        if col_name in pred.columns:
            pred[col_name] = pd.to_datetime(pred[col_name], utc=True, errors="coerce")

    return pred


@st.cache_data(ttl=3600)
def load_zone_centroids() -> pd.DataFrame:
    if not ZONE_CENTROIDS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(ZONE_CENTROIDS_FILE)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="StreamCab Analytics",
    layout="wide",
)

st.title("StreamCab — NYC Taxi Analytics")
st.caption("Real-time pipeline: Kafka ingestion → Spark aggregation → XGBoost forecast")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = load_aggregates()
metrics = load_metrics()
predictions = load_predictions()
centroids = load_zone_centroids()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Controls")
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()
    auto_refresh = st.checkbox("Auto-refresh every 15 s", value=True)

    st.markdown("---")

    # Pipeline status
    st.subheader("Pipeline Status")
    st.markdown(
        f"- Streaming aggregates: {'**Live**' if not df.empty else '*Warming up*'}\n"
        f"- Model metrics: {'**Ready**' if metrics else '*Training*'}\n"
        f"- Predictions: {'**Live**' if not predictions.empty else '*Waiting for data*'}"
    )

    # Model metrics in sidebar
    st.markdown("---")
    with st.expander("Model Metrics", expanded=False):
        if not metrics:
            st.info("No metrics yet. Trainer needs at least 20 rows of aggregate data.")
        else:
            st.markdown("**Baseline** — historical average by zone + hour")
            st.metric("MAE", round(metrics["baseline"]["mae"], 3))
            st.metric("MAPE", f"{metrics['baseline']['mape']:.2f}%")

            xgb = metrics.get("xgboost", {})
            if xgb:
                st.markdown("**XGBoost — Demand**")
                st.metric("MAE", round(xgb["mae"], 3))
                st.metric("MAPE", f"{xgb['mape']:.2f}%")
                improvement = metrics["baseline"]["mae"] - xgb["mae"]
                st.metric("vs Baseline", f"{improvement:+.3f} MAE")

            xgb_amt = metrics.get("xgboost_amount")
            if xgb_amt:
                st.markdown("**XGBoost — Fare**")
                st.metric("Fare MAE", f"${round(xgb_amt['mae'], 2)}")
                st.metric("Fare MAPE", f"{xgb_amt['mape']:.2f}%")

            st.caption(
                f"Trained on {metrics.get('training_rows', '?')} rows | "
                f"Generated: {str(metrics.get('generated_at', '?'))[:16]}"
            )

    with st.expander("Data pipeline", expanded=False):
        st.markdown(
            "- Producer streams TLC trip records → Kafka topic `taxi-trips`\n"
            "- Spark aggregates trips into 5-min windows per pickup zone\n"
            "- Trainer fits XGBoost on zone-window features\n"
            "- Predictor writes next-window forecasts every cycle\n\n"
            "[Kafka Control Center :9021](http://localhost:9021)"
        )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_traffic, tab_predictions = st.tabs(["Live Traffic", "Demand Forecast"])

# ===== TAB 1: Live Traffic =====
with tab_traffic:
    st.markdown(
        "Each row below represents one 5-minute aggregation window across all NYC pickup zones. "
        "Spark reads from Kafka in real time and updates these values every cycle."
    )

    if df.empty:
        st.info(
            "No streaming aggregates yet. "
            "Keep the producer and Spark containers running — data usually appears within 1–2 minutes. "
            "This page auto-refreshes every 15 seconds."
        )
    else:
        latest_ts = df["window_end"].max()
        latest = df[df["window_end"] == latest_ts]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Latest Window",
            str(latest_ts)[:16],
            help="End timestamp of the most recent 5-minute aggregation window",
        )
        col2.metric(
            "Trips (this window)",
            int(latest["trip_count"].sum()),
            help="Total number of taxi trips across all zones in the latest window",
        )
        col3.metric(
            "Avg Speed (mph)",
            f"{float(latest['avg_speed_mph'].mean()):.1f}",
            help="Mean trip speed across all active zones in the latest window",
        )
        col4.metric(
            "Anomalous Trips",
            int(latest["anomaly_count"].sum()),
            help="Trips flagged as unusually slow (<5 mph) or unusually fast (>60 mph)",
        )

        st.markdown("---")

        # Time series aggregated across all zones
        by_window = cast(
            pd.DataFrame,
            df.groupby("window_start", as_index=False)
            .agg(
                total_trips=("trip_count", "sum"),
                avg_speed_mph=("avg_speed_mph", "mean"),
                anomalies=("anomaly_count", "sum"),
            )
            .sort_values(by="window_start"),
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Total Trips per Window**")
            st.caption("Sum of all taxi pickups across NYC, aggregated in 5-minute windows by Spark.")
            fig_trips = px.area(
                by_window,
                x="window_start",
                y="total_trips",
                labels={"window_start": "Window Start", "total_trips": "Trip Count"},
                color_discrete_sequence=["#2563eb"],
            )
            fig_trips.update_layout(margin=dict(t=10, b=0))
            st.plotly_chart(fig_trips, use_container_width=True)

        with c2:
            st.markdown("**Average Trip Speed (mph)**")
            st.caption(
                "Mean speed across all trips in each window. "
                "The dashed line marks the congestion threshold (10 mph)."
            )
            fig_speed = px.line(
                by_window,
                x="window_start",
                y="avg_speed_mph",
                labels={"window_start": "Window Start", "avg_speed_mph": "Speed (mph)"},
                color_discrete_sequence=["#16a34a"],
            )
            fig_speed.add_hline(
                y=10,
                line_dash="dot",
                line_color="red",
                annotation_text="Congestion threshold (10 mph)",
            )
            fig_speed.update_layout(margin=dict(t=10, b=0))
            st.plotly_chart(fig_speed, use_container_width=True)

        if by_window["anomalies"].sum() > 0:
            st.markdown("**Anomalous Trips per Window**")
            st.caption(
                "Count of trips per window where the recorded speed was below 5 mph or above 60 mph. "
                "These may indicate GPS errors, stalled vehicles, or data quality issues."
            )
            fig_anom = px.bar(
                by_window,
                x="window_start",
                y="anomalies",
                color_discrete_sequence=["#dc2626"],
                labels={"window_start": "Window Start", "anomalies": "Anomalous Trips"},
            )
            fig_anom.update_layout(margin=dict(t=10, b=0))
            st.plotly_chart(fig_anom, use_container_width=True)

        # Zone snapshot table
        st.markdown("---")
        st.markdown("**Busiest Pickup Zones — Latest Window**")
        st.caption(
            f"Showing per-zone statistics for the window ending at {str(latest_ts)[:16]}. "
            "Zones are ranked by trip volume."
        )
        zone_snap = cast(
            pd.DataFrame,
            df[df["window_end"] == latest_ts]
            .groupby("pu_location_id", as_index=False)
            .agg(
                trip_count=("trip_count", "sum"),
                avg_speed_mph=("avg_speed_mph", "mean"),
                anomaly_count=("anomaly_count", "sum"),
            )
            .sort_values(by="trip_count", ascending=False),
        )
        zone_snap["avg_speed_mph"] = zone_snap["avg_speed_mph"].round(1)

        if not centroids.empty:
            zone_snap = zone_snap.merge(
                centroids[["location_id", "zone_name"]],
                left_on="pu_location_id",
                right_on="location_id",
                how="left",
            )
            display_cols = ["zone_name", "pu_location_id", "trip_count", "avg_speed_mph", "anomaly_count"]
            display_cols = [c for c in display_cols if c in zone_snap.columns]
            st.dataframe(zone_snap[display_cols].head(20), use_container_width=True)
        else:
            st.dataframe(zone_snap.head(20), use_container_width=True)

        # Map
        if not centroids.empty:
            map_df = zone_snap.merge(
                centroids[["location_id", "latitude", "longitude"]],
                left_on="pu_location_id",
                right_on="location_id",
                how="left",
            ).dropna(subset=["latitude", "longitude"])

            if not map_df.empty:
                st.markdown("---")
                st.markdown("**NYC Pickup Activity Map — Latest Window**")
                st.caption(
                    "Each circle represents one TLC pickup zone. "
                    "Circle size reflects the number of trips originating from that zone. "
                    "Color indicates average speed: green = fast-moving traffic, red = congested or slow."
                )
                fig_map = px.scatter_mapbox(
                    map_df,
                    lat="latitude",
                    lon="longitude",
                    size="trip_count",
                    color="avg_speed_mph",
                    hover_name=("zone_name" if "zone_name" in map_df.columns else "pu_location_id"),
                    hover_data={
                        "trip_count": True,
                        "avg_speed_mph": ":.1f",
                        "anomaly_count": True,
                    },
                    color_continuous_scale="RdYlGn",
                    size_max=40,
                    zoom=10,
                    mapbox_style="open-street-map",
                    labels={
                        "trip_count": "Trips",
                        "avg_speed_mph": "Avg Speed (mph)",
                        "anomaly_count": "Anomalies",
                    },
                )
                fig_map.update_coloraxes(colorbar_title="Speed (mph)")
                fig_map.update_layout(margin=dict(t=10, b=0), height=460)
                st.plotly_chart(fig_map, use_container_width=True)


# ===== TAB 2: Demand Forecast =====
with tab_predictions:
    st.markdown(
        "The XGBoost model is trained on historical zone-level aggregates "
        "(trip count, speed, time of day, day of week). "
        "For each pickup zone it predicts the **trip count, surge multiplier, and estimated fare "
        "for the next 5-minute window** based on the most recent observed data."
    )

    if predictions.empty:
        st.info(
            "Predictions are not available yet. "
            "The model trains automatically once enough aggregate data has accumulated (≥ 20 windows). "
            "Check the Pipeline Status in the sidebar."
        )
    else:
        top = predictions.sort_values("predicted_trip_count_next_window", ascending=False)

        # Show which window is being forecast
        forecast_start = (
            predictions["prediction_for_window_start"].max()
            if "prediction_for_window_start" in predictions.columns
            else None
        )
        forecast_end = (
            predictions["prediction_for_window_end"].max()
            if "prediction_for_window_end" in predictions.columns
            else None
        )
        gen_at = (
            predictions["prediction_generated_at"].max()
            if "prediction_generated_at" in predictions.columns
            else None
        )

        if forecast_start is not None and forecast_end is not None:
            st.info(
                f"Forecasting window: **{str(forecast_start)[:16]} — {str(forecast_end)[:16]} UTC**  \n"
                f"Generated at: {str(gen_at)[:16] if gen_at is not None else '—'} UTC"
            )

        # KPI row
        p1, p2, p3, p4 = st.columns(4)

        hottest_zone = (
            str(int(top.iloc[0]["pu_location_id"])) if not top.empty else "—"
        )
        if not centroids.empty and not top.empty:
            zone_row = centroids[centroids["location_id"] == top.iloc[0]["pu_location_id"]]
            if not zone_row.empty:
                hottest_zone = zone_row.iloc[0]["zone_name"]

        p1.metric(
            "Busiest Zone (next window)",
            hottest_zone,
            help="Zone with the highest predicted trip count in the next window",
        )
        p2.metric(
            "Peak Predicted Demand",
            f"{top.iloc[0]['predicted_trip_count_next_window']:.0f} trips" if not top.empty else "—",
            help="Highest predicted trip count across all zones for the next window",
        )
        if "surge_multiplier" in top.columns:
            p3.metric(
                "Max Surge Multiplier",
                f"{top['surge_multiplier'].max():.2f}x",
                help=(
                    "Surge = predicted demand / historical average for this zone-hour. "
                    "Values above 1.0 indicate above-average demand."
                ),
            )
        if "predicted_avg_fare" in top.columns and not top.empty:
            p4.metric(
                "Est. Fare — Busiest Zone",
                f"${top.iloc[0]['predicted_avg_fare']:.2f}",
                help="Predicted average base fare (before surge) for the busiest zone",
            )

        st.markdown("---")

        # Forecast chart: historical trend + predicted next point
        if not df.empty:
            by_window = cast(
                pd.DataFrame,
                df.groupby("window_start", as_index=False)
                .agg(total_trips=("trip_count", "sum"))
                .sort_values("window_start"),
            )

            total_predicted = (
                top["predicted_trip_count_next_window"].sum() if not top.empty else None
            )

            st.markdown("**Trip Volume: Historical vs. Forecast**")
            st.caption(
                "The solid line shows observed trip counts per 5-minute window (from Spark). "
                "The dashed segment and marker show the XGBoost forecast for the next window."
            )

            fig_forecast = go.Figure()
            fig_forecast.add_trace(
                go.Scatter(
                    x=by_window["window_start"],
                    y=by_window["total_trips"],
                    mode="lines",
                    name="Observed",
                    line=dict(color="#2563eb", width=2),
                )
            )

            if total_predicted is not None and forecast_start is not None:
                last_obs_time = by_window["window_start"].max()
                last_obs_val = by_window.loc[
                    by_window["window_start"] == last_obs_time, "total_trips"
                ].values[0]

                fig_forecast.add_trace(
                    go.Scatter(
                        x=[last_obs_time, forecast_start],
                        y=[last_obs_val, total_predicted],
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(color="#f97316", width=2, dash="dash"),
                        marker=dict(size=10, symbol="diamond"),
                    )
                )

            fig_forecast.update_layout(
                xaxis_title="Window Start",
                yaxis_title="Total Trips",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=10, b=0),
                height=320,
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

        st.markdown("---")

        # Zone-level demand and surge
        surge_df = top.copy()
        if not centroids.empty:
            surge_df = surge_df.merge(
                centroids[["location_id", "zone_name"]],
                left_on="pu_location_id",
                right_on="location_id",
                how="left",
            )
            surge_df["label"] = surge_df.get(
                "zone_name", surge_df["pu_location_id"].astype(str)
            )
        else:
            surge_df["label"] = surge_df["pu_location_id"].astype(str)

        col_demand, col_surge = st.columns(2)

        with col_demand:
            st.markdown("**Predicted Demand by Zone**")
            st.caption(
                "Forecast trip count per zone for the next 5-minute window. "
                "Higher bars = more pickups expected."
            )
            fig_demand = px.bar(
                surge_df.head(15),
                x="label",
                y="predicted_trip_count_next_window",
                color="predicted_trip_count_next_window",
                color_continuous_scale="Blues",
                labels={
                    "label": "Zone",
                    "predicted_trip_count_next_window": "Predicted Trips",
                },
            )
            fig_demand.update_layout(
                margin=dict(t=10, b=60),
                showlegend=False,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_demand, use_container_width=True)

        with col_surge:
            if "surge_multiplier" in surge_df.columns:
                st.markdown("**Surge Multiplier by Zone**")
                st.caption(
                    "Surge = predicted demand / baseline average for this zone and hour. "
                    "Values above 1.0 mean demand is above normal; riders may see higher fares."
                )
                fig_surge = px.bar(
                    surge_df.head(15),
                    x="label",
                    y="surge_multiplier",
                    color="surge_multiplier",
                    color_continuous_scale="YlOrRd",
                    labels={"label": "Zone", "surge_multiplier": "Surge Multiplier"},
                )
                fig_surge.add_hline(
                    y=1.0,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Baseline (1.0x)",
                )
                fig_surge.update_layout(
                    margin=dict(t=10, b=60),
                    showlegend=False,
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_surge, use_container_width=True)
            else:
                st.info("Surge multiplier not available in current predictions.")

        # Fare guidance table
        if "predicted_tip_low" in top.columns and "predicted_avg_fare" in top.columns:
            st.markdown("---")
            st.markdown("**Fare and Tip Guidance — Next Window**")
            st.caption(
                "Estimated base fare and tip range per zone. "
                "Surge-adjusted fare = base fare × surge multiplier."
            )
            tip_df = cast(
                pd.DataFrame,
                top[
                    [
                        "pu_location_id",
                        "predicted_avg_fare",
                        "predicted_tip_low",
                        "predicted_tip_high",
                        "surge_multiplier",
                    ]
                ].copy(),
            )
            if "surge_multiplier" in tip_df.columns:
                tip_df["surge_adjusted_fare"] = (
                    tip_df["predicted_avg_fare"] * tip_df["surge_multiplier"]
                ).round(2)
            rename_map = {
                "pu_location_id": "Zone ID",
                "predicted_avg_fare": "Base Fare ($)",
                "surge_adjusted_fare": "Surge Fare ($)",
                "predicted_tip_low": "Tip Low ($)",
                "predicted_tip_high": "Tip High ($)",
                "surge_multiplier": "Surge",
            }
            tip_df.columns = [rename_map.get(str(col), str(col)) for col in tip_df.columns]
            st.dataframe(tip_df.head(20), use_container_width=True)

        with st.expander("Full prediction table (all zones)"):
            st.dataframe(top, use_container_width=True)


# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh:
    time.sleep(AUTO_REFRESH_SECONDS)
    st.rerun()
