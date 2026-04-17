"""StreamCab — Driver Dashboard"""

import os
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psycopg
import psycopg.rows
import streamlit as st

try:
    streamlit_geolocation = import_module("streamlit_geolocation").streamlit_geolocation
except Exception:
    streamlit_geolocation = None

_PROJECT_ROOT = Path(__file__).parents[2]

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://streamcab:streamcab@postgres:5432/streamcab")
MODEL_FILE = Path(os.getenv("MODEL_FILE", str(_PROJECT_ROOT / "models/traffic_model.joblib")))
ZONE_CENTROIDS_FILE = Path(
    os.getenv("ZONE_CENTROIDS_FILE", str(_PROJECT_ROOT / "data/reference/zone_centroids.csv"))
)

TRIP_REFRESH = 3  # seconds — left panel refreshes
MAP_REFRESH = 15  # seconds — map refreshes
LIVE_SAMPLE_LIMIT = int(os.getenv("LIVE_SAMPLE_LIMIT", "200"))


# Loaders  (cached to avoid re-reading disk on every fragment rerun)
@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    try:
        return joblib.load(MODEL_FILE)
    except Exception:
        return None


@st.cache_data(ttl=TRIP_REFRESH)
def load_live_trips() -> pd.DataFrame:
    try:
        with psycopg.connect(DATABASE_URL, row_factory=psycopg.rows.dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pickup_datetime, dropoff_datetime, passenger_count,
                           trip_distance, pu_location_id, do_location_id,
                           total_amount, emitted_at
                    FROM live_trips
                    ORDER BY emitted_at DESC
                    LIMIT %s
                """,
                    (LIVE_SAMPLE_LIMIT,),
                )
                rows = cur.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=TRIP_REFRESH)
def load_trips_per_minute() -> int:
    try:
        with psycopg.connect(DATABASE_URL, row_factory=psycopg.rows.dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) AS trip_count
                    FROM live_trips
                    WHERE emitted_at >= NOW() - INTERVAL '60 seconds'
                    """
                )
                row = cur.fetchone()
        if not row:
            return 0
        return int(row["trip_count"] or 0)
    except Exception:
        return 0


@st.cache_data(ttl=MAP_REFRESH)
def load_predictions() -> pd.DataFrame:
    """Latest prediction per zone, written by the predictor service."""
    try:
        with psycopg.connect(DATABASE_URL, row_factory=psycopg.rows.dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (pu_location_id)
                        pu_location_id,
                        trip_count,
                        avg_speed_mph,
                        avg_trip_distance,
                        predicted_avg_fare  AS predicted_fare,
                        predicted_tip_low,
                        predicted_tip_high,
                        surge_multiplier    AS surge
                    FROM predictions
                    ORDER BY pu_location_id, prediction_generated_at DESC
                """
                )
                rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["predicted_tip"] = ((df["predicted_tip_low"] + df["predicted_tip_high"]) / 2).round(2)
        return df.sort_values("predicted_fare", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_zone_centroids() -> pd.DataFrame:
    if not ZONE_CENTROIDS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(ZONE_CENTROIDS_FILE)


# Page setup


st.set_page_config(page_title="StreamCab", layout="wide")

st.markdown(
    """
<style>
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stAppViewContainer"] {
    background: #f8fafc;
}
.block-container { padding: 1rem 1.6rem 0.8rem; max-width: 1500px; }
[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlock"] {
    gap: 0.5rem;
}
[data-testid="stHorizontalBlock"] { gap: 0.55rem !important; }
h1, h2, h3, h4 { margin: 0.1rem 0 0.2rem 0 !important; letter-spacing: .01em; }
h3 { font-size: 0.86rem !important; font-weight: 700 !important; color: #0f172a; }
hr { margin: 0.35rem 0 0.45rem 0 !important; }
[data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; color: #0f172a; }
[data-testid="stMetricLabel"] {
    font-size: 0.62rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .08em; color: #64748b;
}
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: .45rem .65rem;
    box-shadow: none;
}
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    overflow: hidden;
    background: white;
}
.section-title {
    font-size: 0.80rem;
    font-weight: 700;
    color: #0f172a;
    padding: 0 0 6px 0;
    margin: 2px 0 6px 0;
    border-bottom: 1px solid #dbeafe;
}
.section-wrap {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 6px;
}
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stDataFrame"] thead th {
    background: #eff6ff !important;
    color: #1e3a8a !important;
    font-weight: 700 !important;
}
.hero {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    background: #ffffff;
    border: 1px solid #dbeafe;
    border-left: 4px solid #2563eb;
    border-radius: 16px;
    padding: 10px 12px;
    margin-bottom: 2px;
    box-shadow: none;
}
.hero-title { font-size: 1.0rem; font-weight: 750; color: #1e3a8a; }
.hero-sub { font-size: .70rem; color: #475569; margin-top: 1px; }
.hero-pill {
    border-radius: 999px;
    border: 1px solid #dbeafe;
    background: #f8fafc;
    color: #334155;
    font-size: .67rem;
    font-weight: 700;
    padding: 5px 10px;
    white-space: nowrap;
}
.zone-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 6px;
    margin-top: 2px;
    width: 100%;
}
.zone-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 6px 8px;
    box-shadow: none;
    min-width: 0;
}
.zone-name { font-size: .80rem; font-weight: 700; color: #0f172a; }
.zone-fare { font-size: 1.00rem; font-weight: 750; color: #0f172a; }
.zone-tip  { font-size: .60rem; color: #64748b; }
.zone-meta { font-size: .50rem; color: #64748b; margin-top: 2px; }
.badge { display:inline-block; border-radius:5px; padding:1px 6px;
         font-size:.64rem; font-weight:700; margin-left:6px; }
.badge-red, .badge-yellow, .badge-green { background:#f1f5f9; color:#475569; }
</style>
""",
    unsafe_allow_html=True,
)


# Static data (loaded once)


model_bundle = load_model()
centroids = load_zone_centroids()

zone_names: dict[int, str] = {}
zone_lat: dict[int, float] = {}
zone_lon: dict[int, float] = {}
if not centroids.empty:
    zone_names = dict(
        zip(centroids["location_id"].astype(int), centroids["zone_name"], strict=False)
    )
    zone_lat = dict(zip(centroids["location_id"].astype(int), centroids["latitude"], strict=False))
    zone_lon = dict(zip(centroids["location_id"].astype(int), centroids["longitude"], strict=False))


def zone_label(zone_id: int) -> str:
    name = zone_names.get(zone_id)
    return name if name else f"Unknown Zone {zone_id}"


@st.cache_data(ttl=TRIP_REFRESH)
def predict_trip_fares(_model_bundle, trips_hash: str, _trips: pd.DataFrame) -> pd.Series:
    """Predict fare per trip using the trained model features."""
    if _trips.empty or _model_bundle is None:
        return pd.Series(dtype=float)

    fare_model = _model_bundle["fare_model"]
    fare_features = _model_bundle["fare_features"]

    t = _trips.copy()
    pickup_col = (
        t["pickup_datetime"] if "pickup_datetime" in t else pd.Series(pd.NaT, index=t.index)
    )
    dropoff_col = (
        t["dropoff_datetime"] if "dropoff_datetime" in t else pd.Series(pd.NaT, index=t.index)
    )
    t["pickup_dt"] = pd.to_datetime(pickup_col, errors="coerce")
    t["dropoff_dt"] = pd.to_datetime(dropoff_col, errors="coerce")
    t["avg_duration_min"] = (
        ((t["dropoff_dt"] - t["pickup_dt"]).dt.total_seconds() / 60)
        .clip(lower=1, upper=180)
        .fillna(10)
    )
    t["avg_trip_distance"] = pd.to_numeric(
        t.get("trip_distance", pd.Series(0, index=t.index)), errors="coerce"
    ).fillna(0)
    t["avg_speed_mph"] = (
        (t["avg_trip_distance"] / (t["avg_duration_min"] / 60).replace(0, np.nan))
        .clip(lower=1, upper=80)
        .fillna(15)
    )
    t["hour"] = t["pickup_dt"].dt.hour.fillna(12).astype(int)
    t["day_of_week"] = t["pickup_dt"].dt.dayofweek.fillna(0).astype(int)
    t["is_weekend"] = (t["day_of_week"] >= 5).astype(int)
    t["trip_count"] = 1

    model_df = pd.DataFrame(index=t.index)
    model_df["pu_location_id"] = pd.to_numeric(
        t.get("pu_location_id", pd.Series(0, index=t.index)), errors="coerce"
    ).fillna(0)
    model_df["hour"] = t["hour"]
    model_df["day_of_week"] = t["day_of_week"]
    model_df["is_weekend"] = t["is_weekend"]
    model_df["avg_trip_distance"] = t["avg_trip_distance"]
    model_df["avg_duration_min"] = t["avg_duration_min"]
    model_df["avg_speed_mph"] = t["avg_speed_mph"]
    model_df["trip_count"] = t["trip_count"]

    for col in fare_features:
        if col not in model_df.columns:
            model_df[col] = 0

    preds = fare_model.predict(model_df[fare_features]).clip(min=0)
    return pd.Series(preds, index=t.index, dtype=float)


# Header
model_status = "Model loaded" if model_bundle else "No model — run train-once"
st.markdown(
    f"""
    <div class="hero">
        <div>
            <div class="hero-title">StreamCab</div>
            <div class="hero-sub">Real-time yellow taxi intelligence · one-screen driver view</div>
        </div>
        <div class="hero-pill">{model_status}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_row = st.empty()  # filled by the live fragment below
st.divider()


# LEFT Portion — trips + zone rankings, refreshes every TRIP_REFRESH seconds
@st.fragment(run_every=TRIP_REFRESH)
def live_panel():
    live_raw = load_live_trips()
    trips_per_min = load_trips_per_minute()

    trips = pd.DataFrame()
    if not live_raw.empty:
        trips = live_raw.copy()
        total_amount_col = (
            trips["total_amount"] if "total_amount" in trips else pd.Series(0, index=trips.index)
        )
        trip_distance_col = (
            trips["trip_distance"] if "trip_distance" in trips else pd.Series(0, index=trips.index)
        )
        pu_col = (
            trips["pu_location_id"]
            if "pu_location_id" in trips
            else pd.Series(0, index=trips.index)
        )
        do_col = (
            trips["do_location_id"]
            if "do_location_id" in trips
            else pd.Series(0, index=trips.index)
        )
        emitted_col = (
            trips["emitted_at"] if "emitted_at" in trips else pd.Series(pd.NaT, index=trips.index)
        )
        trips["total_amount"] = pd.to_numeric(total_amount_col, errors="coerce").fillna(0)
        trips["trip_distance"] = pd.to_numeric(trip_distance_col, errors="coerce").fillna(0)
        trips["pu_location_id"] = pd.to_numeric(pu_col, errors="coerce").fillna(0).astype(int)
        trips["do_location_id"] = pd.to_numeric(do_col, errors="coerce").fillna(0).astype(int)
        trips["emitted_at"] = pd.to_datetime(emitted_col, errors="coerce")
        trips["pickup_name"] = trips["pu_location_id"].astype(int).map(zone_label)
        trips["dropoff_name"] = trips["do_location_id"].astype(int).map(zone_label)

    # Zone predictions come from the predictor service via Postgres
    zone_preds = load_predictions()

    with kpi_row.container():
        k1, k2, k3, k4, k5 = st.columns(5)
        if not trips.empty:
            top_zid = int(trips["pu_location_id"].mode().iloc[0])
            n = len(trips)
            k1.metric("Trips / min", str(trips_per_min), help="Last 60 s")
            k2.metric(f"Avg Fare ({n})", f"${trips['total_amount'].mean():.2f}")
            k3.metric(f"Avg Miles ({n})", f"{trips['trip_distance'].mean():.1f} mi")
            k4.metric("Hottest Zone", zone_names.get(top_zid, f"Zone {top_zid}"))
        else:
            for k in (k1, k2, k3, k4):
                k.metric("—", "—")
        if not zone_preds.empty:
            ms = float(zone_preds["surge"].max())
            k5.metric(
                "Max Surge",
                "High" if ms >= 2 else ("Moderate" if ms >= 1.3 else "Normal"),
            )
        else:
            k5.metric("Max Surge", "—")

    st.markdown('<div class="section-title">Pickup Zones</div>', unsafe_allow_html=True)
    if zone_preds.empty:
        st.info(
            "Waiting for predictor — run `docker compose run --rm train-once` first if model is missing."
        )
    else:
        medals = ["1", "2", "3", "4", "5", "6"]
        html = '<div class="zone-grid">'
        for rank, row in enumerate(zone_preds.head(6).to_dict(orient="records")):
            surge = float(row["surge"])
            fare = float(row["predicted_fare"])
            tip = float(row["predicted_tip"])
            speed = float(row.get("avg_speed_mph", 0))
            n_trips = int(row["trip_count"])
            name = zone_names.get(int(row["pu_location_id"]), f"Zone {int(row['pu_location_id'])}")
            medal = medals[rank] if rank < len(medals) else str(rank + 1)
            badge = (
                f'<span class="badge badge-red">{surge:.1f}x</span>'
                if surge >= 2.0
                else (
                    f'<span class="badge badge-yellow">{surge:.1f}x</span>'
                    if surge >= 1.3
                    else f'<span class="badge badge-green">{surge:.1f}x</span>'
                )
            )
            html += f"""
            <div class="zone-card">
              <div><span class="zone-name">{medal} {name}</span>{badge}</div>
              <div style="display:flex;align-items:baseline;gap:2px">
                <span class="zone-fare">${fare:.2f}</span>
                                <span class="zone-tip">est. tip ${tip:.2f}</span>
              </div>
              <div class="zone-meta">{speed:.0f} mph &nbsp;·&nbsp; {n_trips} trip{"s" if n_trips != 1 else ""}</div>
            </div>"""
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)


@st.fragment(run_every=TRIP_REFRESH)
def trips_panel():
    live_raw = load_live_trips()

    trips = pd.DataFrame()
    if not live_raw.empty:
        trips = live_raw.copy()
        trips["total_amount"] = pd.to_numeric(
            (trips["total_amount"] if "total_amount" in trips else pd.Series(0, index=trips.index)),
            errors="coerce",
        ).fillna(0)
        trips["trip_distance"] = pd.to_numeric(
            (
                trips["trip_distance"]
                if "trip_distance" in trips
                else pd.Series(0, index=trips.index)
            ),
            errors="coerce",
        ).fillna(0)
        trips["pu_location_id"] = (
            pd.to_numeric(
                (
                    trips["pu_location_id"]
                    if "pu_location_id" in trips
                    else pd.Series(0, index=trips.index)
                ),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )
        trips["do_location_id"] = (
            pd.to_numeric(
                (
                    trips["do_location_id"]
                    if "do_location_id" in trips
                    else pd.Series(0, index=trips.index)
                ),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )
        trips["emitted_at"] = pd.to_datetime(
            (
                trips["emitted_at"]
                if "emitted_at" in trips
                else pd.Series(pd.NaT, index=trips.index)
            ),
            errors="coerce",
        )
        trips["pickup_name"] = trips["pu_location_id"].astype(int).map(zone_label)
        trips["dropoff_name"] = trips["do_location_id"].astype(int).map(zone_label)

        if model_bundle is not None:
            trips_hash = str(len(trips)) + str(trips["pu_location_id"].sum())
            trip_pred_fares = predict_trip_fares(model_bundle, trips_hash + "_tbl", trips)
            trip_pred_num = pd.to_numeric(trip_pred_fares, errors="coerce").fillna(0)
            trips["model_tip"] = (
                (trip_pred_num * 0.18).round(2) if not trip_pred_fares.empty else np.nan
            )
        else:
            trips["model_tip"] = np.nan

    st.markdown('<div class="section-title">Live Trips</div>', unsafe_allow_html=True)
    if trips.empty:
        st.info("Waiting for Kafka data…")
    else:
        last_ts = trips["emitted_at"].max()
        last_text = last_ts.strftime("%H:%M:%S") if pd.notna(last_ts) else "—"
        st.caption(
            f"Latest sample: {last_text} · Rows show most recent trips with model-estimated tip"
        )
        show = trips.sort_values("emitted_at", ascending=False).head(20).copy()
        show["Time"] = show["emitted_at"].dt.strftime("%H:%M:%S")
        miles_vals = pd.to_numeric(show["trip_distance"], errors="coerce").fillna(0)
        show["Miles"] = [f"{v:.1f} mi" for v in miles_vals.round(1).to_list()]
        show["Fare"] = show["total_amount"].round(2)
        show["Model Tip"] = show["model_tip"].round(2)
        st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
        st.dataframe(
            show[["Time", "pickup_name", "dropoff_name", "Miles", "Fare", "Model Tip"]].rename(
                columns={"pickup_name": "Pickup", "dropoff_name": "Dropoff"}
            ),
            use_container_width=True,
            hide_index=True,
            height=320,
            column_config={
                "Time": st.column_config.TextColumn("Time", width="small"),
                "Pickup": st.column_config.TextColumn("Pickup"),
                "Dropoff": st.column_config.TextColumn("Dropoff"),
                "Miles": st.column_config.TextColumn("Miles", width="small"),
                "Fare": st.column_config.NumberColumn("Fare ($)", format="$%.2f", width="small"),
                "Model Tip": st.column_config.NumberColumn(
                    "Est Tip ($)", format="$%.2f", width="small"
                ),
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)


# RIGHT portion — map, refreshes every MAP_REFRESH seconds
@st.fragment(run_every=MAP_REFRESH)
def map_panel():
    live_raw = load_live_trips()
    trips = pd.DataFrame()
    if not live_raw.empty:
        trips = live_raw.copy()
        total_amount_col = (
            trips["total_amount"] if "total_amount" in trips else pd.Series(0, index=trips.index)
        )
        trip_distance_col = (
            trips["trip_distance"] if "trip_distance" in trips else pd.Series(0, index=trips.index)
        )
        pu_col = (
            trips["pu_location_id"]
            if "pu_location_id" in trips
            else pd.Series(0, index=trips.index)
        )
        do_col = (
            trips["do_location_id"]
            if "do_location_id" in trips
            else pd.Series(0, index=trips.index)
        )
        emitted_col = (
            trips["emitted_at"] if "emitted_at" in trips else pd.Series(pd.NaT, index=trips.index)
        )
        trips["total_amount"] = pd.to_numeric(total_amount_col, errors="coerce").fillna(0)
        trips["trip_distance"] = pd.to_numeric(trip_distance_col, errors="coerce").fillna(0)
        trips["pu_location_id"] = pd.to_numeric(pu_col, errors="coerce").fillna(0).astype(int)
        trips["do_location_id"] = pd.to_numeric(do_col, errors="coerce").fillna(0).astype(int)
        trips["emitted_at"] = pd.to_datetime(emitted_col, errors="coerce")

    st.markdown("### City Map — All Stations & Most Popular")

    if centroids.empty:
        st.info("No station coordinates found — check zone_centroids.csv.")
        return

    station_base = centroids[["location_id", "zone_name", "latitude", "longitude"]].copy()
    station_base["location_id"] = station_base["location_id"].astype(int)

    if trips.empty:
        station_base["pickup_trips"] = 0
        station_base["dropoff_trips"] = 0
        station_base["total_trips"] = 0
    else:
        pickup_counts = (
            trips.groupby("pu_location_id")
            .size()
            .reset_index(name="pickup_trips")
            .rename(columns={"pu_location_id": "location_id"})
        )
        dropoff_counts = (
            trips.groupby("do_location_id")
            .size()
            .reset_index(name="dropoff_trips")
            .rename(columns={"do_location_id": "location_id"})
        )
        station_base = station_base.merge(pickup_counts, on="location_id", how="left")
        station_base = station_base.merge(dropoff_counts, on="location_id", how="left")
        station_base["pickup_trips"] = station_base["pickup_trips"].fillna(0).astype(int)
        station_base["dropoff_trips"] = station_base["dropoff_trips"].fillna(0).astype(int)
        station_base["total_trips"] = station_base["pickup_trips"] + station_base["dropoff_trips"]

    popular = station_base.sort_values("total_trips", ascending=False).head(12).copy()
    top_total = max(int(popular["total_trips"].max()), 1)
    popular["size"] = popular["total_trips"].apply(lambda v: 12 + (float(v) / top_total * 20))
    station_hover = station_base.apply(
        lambda r: (
            f"{r['zone_name']}<br>Pickup: {int(r['pickup_trips'])}"
            f"<br>Dropoff: {int(r['dropoff_trips'])}"
        ),
        axis=1,
    )
    popular_hover = popular.apply(
        lambda r: (
            f"{r['zone_name']}<br>Total: {int(r['total_trips'])}"
            f"<br>Pickup: {int(r['pickup_trips'])}"
            f"<br>Dropoff: {int(r['dropoff_trips'])}"
        ),
        axis=1,
    )

    fig_map = go.Figure()
    fig_map.add_trace(
        go.Scattermapbox(
            mode="markers",
            uid="all-stations",
            lon=station_base["longitude"],
            lat=station_base["latitude"],
            marker={"size": 8, "color": "#64748b"},
            name="All Stations",
            hovertext=station_hover,
            hoverinfo="text",
            opacity=0.82,
        )
    )

    fig_map.add_trace(
        go.Scattermapbox(
            mode="markers+text",
            uid="popular-stations",
            lon=popular["longitude"],
            lat=popular["latitude"],
            marker={
                "size": popular["size"],
                "color": popular["total_trips"],
                "colorscale": "Blues",
                "showscale": True,
                "colorbar": {"title": "Trips"},
            },
            name="Most Popular",
            text=popular["zone_name"],
            textposition="top center",
            hovertext=popular_hover,
            hoverinfo="text",
            opacity=0.95,
        )
    )

    if streamlit_geolocation is not None:
        geo = streamlit_geolocation()
        if (
            isinstance(geo, dict)
            and geo.get("latitude") is not None
            and geo.get("longitude") is not None
        ):
            fig_map.add_trace(
                go.Scattermapbox(
                    mode="markers",
                    uid="user-location",
                    lon=[float(geo["longitude"])],
                    lat=[float(geo["latitude"])],
                    marker={"size": 15, "color": "#3b82f6"},
                    name="Your Location",
                    hovertext=["Your location"],
                    hoverinfo="text",
                )
            )

    fig_map.update_layout(
        uirevision="city-map-lock",
        mapbox={
            "style": "carto-positron",
            "zoom": 10,
            "center": {"lat": 40.75, "lon": -73.98},
            "uirevision": "city-map-lock",
        },
        margin={"t": 0, "b": 0},
        height=310,
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    st.plotly_chart(fig_map, use_container_width=True, key="city_map_chart")

    m1, m2, m3 = st.columns(3)
    active_pickup_zones = int((station_base["pickup_trips"] > 0).sum())
    active_dropoff_zones = int((station_base["dropoff_trips"] > 0).sum())
    top_station = popular.iloc[0]["zone_name"] if not popular.empty else "—"
    m1.metric("Active pickup zones", str(active_pickup_zones))
    m2.metric("Active dropoff zones", str(active_dropoff_zones))
    m3.metric("Most active station", str(top_station))

    if int(popular["total_trips"].max()) == 0:
        st.caption(
            f"Showing all {len(station_base)} stations from centroid file · waiting for Kafka trips to rank popularity"
        )
    else:
        st.caption(
            f"Showing all {len(station_base)} stations + top {len(popular)} most popular by pickup+dropoff count · updates every {MAP_REFRESH}s"
        )

    if streamlit_geolocation is None:
        st.caption(
            "To show your live blue location marker, install dashboard dependency `streamlit-geolocation` and rebuild the dashboard image."
        )


# Render fragments
live_panel()

col_left, col_right = st.columns([1, 1.35], gap="large")
with col_left:
    trips_panel()
with col_right:
    map_panel()
