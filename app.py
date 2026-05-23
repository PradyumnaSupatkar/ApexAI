import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from data_loader import load_session, get_available_drivers, get_laps
from telemetry_engine import get_telemetry_for_lap
from delta_engine import compute_delta
from fingerprint_engine import (
    extract_driver_features, run_fingerprinting,
    label_cluster, FEATURE_COLUMNS
)
from pace_model_engine import (
    build_season_dataset, train_global_model,
    predict_for_driver, format_laptime,
    TIRE_COLORS
)


#Utility 
def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """
    '#RRGGBB' → 'rgba(r,g,b,alpha)'
    Plotly rejects 8-char hex like '#E8002D33' for fillcolor — always use rgba().
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


#Page
st.set_page_config(
    page_title="ApexAI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏎️ ApexAI — Telemetry Intelligence Platform")
st.caption("Performance analytics · Predictive ML  · Driver behavior classification")
st.divider()


#Sidebar
GP_OPTIONS = [
    "Bahrain", "Saudi Arabia", "Australia", "Azerbaijan",
    "Monaco", "Spain", "Canada", "Austria",
    "Silverstone", "Hungary", "Belgium",
    "Netherlands", "Monza", "Singapore",
    "Japan", "Qatar", "USA", "Mexico", "Brazil",
    "Las Vegas", "Abu Dhabi"
]

with st.sidebar:
    st.header("Session Config")
    year = st.selectbox("Season", [2023, 2022], index=0)
    gp   = st.selectbox("Grand Prix", GP_OPTIONS, index=0)
    session_type = "R"

    st.divider()
    st.header("Clustering")
    n_clusters = st.slider("Fingerprint Clusters", 2, 5, 3)

    st.divider()
    st.header("ML Race Pace Forecasting")
    st.caption("Model trained on full 2022 season data. Select a 2023 race to predict.")
    quick_gps = ["Bahrain", "Saudi Arabia", "Australia", "Monaco",
                 "Silverstone", "Monza", "Abu Dhabi"]
    train_gps = quick_gps  # fixed — not user-editable
    test_gp = st.selectbox(
        "Race to predict (2023)",
        options=GP_OPTIONS,
        index=0,
        help="The model will predict lap times for this single 2023 race."
    )


#Session Load
with st.spinner(f"Loading {year} {gp} Race session... (first load ~30s, then cached)"):
    try:
        session = load_session(year=year, gp=gp, session_type=session_type)
        drivers = get_available_drivers(session)
        laps    = get_laps(session)
        st.success(f"{year} {gp} Race — {len(drivers)} drivers loaded")
    except Exception as e:
        st.error(f"Session load failed: {e}")
        st.stop()

st.divider()


#TABS
tab1, tab2, tab3 = st.tabs([
    "Telemetry & Delta",
    "ML Race Pace Forecasting",
    "Driver Fingerprinting",
])


#tab1 - Telemetry
with tab1:
    st.subheader("Driver & Lap Selection")

    col1, col2 = st.columns(2)
    with col1:
        driver1 = st.selectbox("Driver 1", drivers, index=0, key="t1d1")
    with col2:
        driver2 = st.selectbox("Driver 2", drivers,
                               index=min(1, len(drivers)-1), key="t1d2")

    if driver1 == driver2:
        st.warning("Select two different drivers.")
        st.stop()

    #clean laps(improvised)
    def get_clean_lap_numbers(session, driver):
        dl = laps[(laps["Driver"] == driver) & laps["LapTime"].notna()].copy()
        dl = dl[dl["PitInTime"].isna() & dl["PitOutTime"].isna()]
        median = dl["LapTime"].dt.total_seconds().median()
        dl = dl[dl["LapTime"].dt.total_seconds() <= median * 1.10]
        return set(dl["LapNumber"].astype(int).tolist())

    clean1 = get_clean_lap_numbers(session, driver1)
    clean2 = get_clean_lap_numbers(session, driver2)
    shared_laps = sorted(clean1 & clean2)

    if not shared_laps:
        st.error("No clean shared laps found for this pairing. Try different drivers.")
        st.stop()

    default_lap = 5 if 5 in shared_laps else shared_laps[0]
    lap_num = st.selectbox(
        f"Lap ({len(shared_laps)} clean laps shared by both drivers)",
        shared_laps,
        index=shared_laps.index(default_lap),
    )

    with st.spinner(f"Fetching telemetry for {driver1} & {driver2}..."):
        tel1 = get_telemetry_for_lap(session, driver1, lap_num)
        tel2 = get_telemetry_for_lap(session, driver2, lap_num)

    if tel1.empty:
        st.error(f"No telemetry for **{driver1}** on Lap {lap_num}.")
        st.stop()
    if tel2.empty:
        st.error(f"No telemetry for **{driver2}** on Lap {lap_num}.")
        st.stop()

    with st.expander("Raw Telemetry Preview", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**{driver1}** — {len(tel1)} points")
            st.dataframe(tel1.head(10), use_container_width=True)
        with c2:
            st.write(f"**{driver2}** — {len(tel2)} points")
            st.dataframe(tel2.head(10), use_container_width=True)

    #Speed Comparison
    st.subheader("Speed Comparison")
    fig_speed = go.Figure()
    fig_speed.add_trace(go.Scatter(
        x=tel1["Distance"], y=tel1["Speed"],
        name=driver1, line=dict(color="#E8002D", width=2)
    ))
    fig_speed.add_trace(go.Scatter(
        x=tel2["Distance"], y=tel2["Speed"],
        name=driver2, line=dict(color="#0090D0", width=2)
    ))
    fig_speed.update_layout(
        template="plotly_dark",
        xaxis_title="Distance (m)", yaxis_title="Speed (km/h)",
        height=350, margin=dict(t=20)
    )
    st.plotly_chart(fig_speed, use_container_width=True)

    #Throttle & Brake
    with st.expander("Throttle & Brake Traces", expanded=False):
        fig_tb = go.Figure()
        fig_tb.add_trace(go.Scatter(x=tel1["Distance"], y=tel1["Throttle"],
            name=f"{driver1} Throttle", line=dict(color="#E8002D")))
        fig_tb.add_trace(go.Scatter(x=tel2["Distance"], y=tel2["Throttle"],
            name=f"{driver2} Throttle", line=dict(color="#0090D0")))
        fig_tb.add_trace(go.Scatter(
            x=tel1["Distance"],
            y=tel1["Brake"].astype(float) * 100,
            name=f"{driver1} Brake",
            line=dict(color="#FF8800", dash="dot")))
        fig_tb.update_layout(template="plotly_dark", height=300,
            xaxis_title="Distance (m)", yaxis_title="%", margin=dict(t=10))
        st.plotly_chart(fig_tb, use_container_width=True)

    #Delta Analysis
    st.subheader("Delta Analysis")
    delta_df = compute_delta(tel1, tel2)

    if delta_df.empty:
        st.warning("Delta could not be computed — telemetry too short or misaligned.")
    else:
        delta_df["DeltaSmooth"] = delta_df["Delta"].rolling(10, min_periods=1).mean()

        fig_delta = go.Figure()
        fig_delta.add_trace(go.Scatter(
            x=delta_df["Distance"], y=delta_df["DeltaSmooth"],
            mode="lines", name="Delta", line=dict(width=2, color="white")
        ))
        fig_delta.add_trace(go.Scatter(
            x=delta_df["Distance"], y=[0] * len(delta_df),
            mode="lines", name="Equal",
            line=dict(dash="dash", color="gray", width=1)
        ))
        fig_delta.add_trace(go.Scatter(
            x=delta_df["Distance"],
            y=delta_df["DeltaSmooth"].where(delta_df["DeltaSmooth"] < 0),
            fill="tozeroy", mode="none",
            name=f"{driver1} faster", fillcolor=hex_to_rgba("#00FF00", 0.2)
        ))
        fig_delta.add_trace(go.Scatter(
            x=delta_df["Distance"],
            y=delta_df["DeltaSmooth"].where(delta_df["DeltaSmooth"] > 0),
            fill="tozeroy", mode="none",
            name=f"{driver2} faster", fillcolor=hex_to_rgba("#FF3232", 0.2)
        ))
        fig_delta.update_layout(
            title=f"Delta: {driver1} vs {driver2} | Lap {lap_num}",
            template="plotly_dark",
            xaxis_title="Distance (m)", yaxis_title="Time Delta (s)",
            height=430
        )
        st.plotly_chart(fig_delta, use_container_width=True)

        final_delta = delta_df["Delta"].iloc[-1]
        faster = driver1 if final_delta < 0 else driver2
        st.metric("Final Gap at Lap End", f"{abs(final_delta):.3f}s",
                  delta=f"{faster} faster overall")




#tab2 - Race Pace Prediction 
with tab2:
    st.subheader("ML Race Pace Forecasting — Train 2022 → Predict 2023")
    st.markdown("""
    **True temporal generalisation:** Model trained on 2022 race laps, predicting 2023 pace.""")

    #Build datasets
    st.info(f"Predicting **2023 {test_gp}**")

    with st.spinner("Loading 2022 training data....."):
        train_df = build_season_dataset(season=2022, gp_list=train_gps)

    if train_df.empty:
        st.error("No training data loaded. Check your 2022 GP selection.")
        st.stop()

    with st.spinner(f"Loading 2023 {test_gp} race data..."):
        test_df = build_season_dataset(season=2023, gp_list=[test_gp])

    if test_df.empty:
        st.error(f"No data loaded for 2023 {test_gp}. Try a different race.")
        st.stop()

    st.success(
        f"✅ Training set: **{len(train_df):,} laps** from 2022 | "
        f"Test set: **{len(test_df):,} laps** — 2023 {test_gp}"
    )

    #Train model
    with st.spinner("Training RandomForest on 2022 data..."):
        model = train_global_model(train_df)

    if model is None:
        st.error("Not enough training data. Add more GPs.")
        st.stop()

    st.success("✅ Model trained on 2022 season. Ready to predict 2023.")

    #Driver selection for display
    available_test_drivers = sorted(test_df["Driver"].unique().tolist())
    pred_driver = st.selectbox(
        "Select driver to visualise predictions",
        available_test_drivers,
        help="Predictions are computed for ALL drivers but displayed one at a time."
    )

    #Predict
    result_df, metrics = predict_for_driver(model, test_df, pred_driver)

    if result_df is None:
        st.error(f"Not enough 2023 data for {pred_driver}. Try another driver or a different race.")
        st.stop()

    #Prediction chart
    st.subheader(f"{pred_driver} — Actual vs Predicted Lap Times (2023 {test_gp})")

    #Build y-axis ticks in M:SS.mmm format
    y_min = result_df["LapTime_s"].min()
    y_max = result_df["LapTime_s"].max()
    tick_vals = np.arange(np.floor(y_min) - 1, np.ceil(y_max) + 2, 1.0)
    tick_text = [format_laptime(v) for v in tick_vals]

    #Colour actual dots by compound
    fig_pred = go.Figure()

    for compound, group in result_df.groupby("Compound"):
        color = TIRE_COLORS.get(compound, "#AAAAAA")
        hover = [
            f"Lap {int(ln)}<br>Actual: {format_laptime(t)}<br>Compound: {compound}"
            for ln, t in zip(group["LapNumber"], group["LapTime_s"])
        ]
        fig_pred.add_trace(go.Scatter(
            x=group["LapNumber"], y=group["LapTime_s"],
            mode="markers", name=f"Actual ({compound})",
            marker=dict(color=color, size=8, line=dict(width=1, color="white")),
            text=hover, hoverinfo="text"
        ))

    #Smooth predicted line
    hover_pred = [
        f"Lap {int(ln)}<br>Predicted: {format_laptime(t)}"
        for ln, t in zip(result_df["LapNumber"], result_df["PredictedSmooth"])
    ]
    fig_pred.add_trace(go.Scatter(
        x=result_df["LapNumber"], y=result_df["PredictedSmooth"],
        mode="lines", name="Predicted (smoothed)",
        line=dict(color="#BB86FC", width=2.5),
        text=hover_pred, hoverinfo="text"
    ))

    fig_pred.update_layout(
        template="plotly_dark",
        xaxis_title="Lap Number",
        yaxis=dict(title="Lap Time", tickvals=tick_vals, ticktext=tick_text),
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=20)
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    #Metrics
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("MAE (test set)", f"{metrics['mae']:.3f}s")
    mc2.metric("R² Score", f"{metrics['r2']:.3f}")
    mc3.metric("Laps predicted", str(metrics["n_laps"]))

    #Residuals chart
    with st.expander("Residuals (Predicted − Actual)", expanded=False):
        residuals = result_df["Predicted_s"] - result_df["LapTime_s"]
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=result_df["LapNumber"], y=residuals,
            mode="markers+lines", name="Residual",
            line=dict(color="#BB86FC", width=1.5),
            marker=dict(size=6)
        ))
        fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_res.update_layout(
            template="plotly_dark",
            xaxis_title="Lap Number", yaxis_title="Residual (s)",
            height=280, margin=dict(t=10)
        )
        st.plotly_chart(fig_res, use_container_width=True)
        st.caption("Positive = model predicted slower than actual. "
                   "Negative = model predicted faster than actual. "
                   "Large spikes often correspond to safety car laps that passed the filter.")




#tab3 - Driver Fingerprinting
with tab3:
    st.subheader("Driver Fingerprinting — KMeans + PCA + Radar")
    st.markdown(f"""
    Converts raw race telemetry into **behavioral driving style profiles** using unsupervised ML.""")

    with st.spinner("Extracting driver features from race telemetry..."):
        features_df = extract_driver_features(session)

    if features_df.empty:
        st.error("Could not extract features. Try a different session.")
        st.stop()

    st.success(f"Features extracted for {len(features_df)} drivers")

    with st.expander("Raw Feature Matrix", expanded=False):
        st.dataframe(features_df.round(3), use_container_width=True)

    result_df_fp, explained, cluster_profiles = run_fingerprinting(
        features_df, n_clusters=n_clusters
    )

    if result_df_fp is None:
        st.warning(f"Not enough drivers for {n_clusters} clusters. Reduce the slider.")
        st.stop()

    #PCA Cluster Map
    st.subheader("PCA Cluster Map")
    COLOR_MAP = {
        "0": "#E8002D", "1": "#0090D0",
        "2": "#39B54A", "3": "#FF8800", "4": "#9B59B6"
    }
    fig_pca = px.scatter(
        result_df_fp, x="PC1", y="PC2",
        color="Cluster", text="Driver",
        color_discrete_map=COLOR_MAP,
        hover_data={col: ":.2f" for col in FEATURE_COLUMNS},
        title=(
            f"Driver Style Clusters — "
            f"PC1: {explained[0]:.1f}% | PC2: {explained[1]:.1f}% variance explained"
        )
    )
    fig_pca.update_traces(
        textposition="top center",
        marker=dict(size=14, line=dict(width=1.5, color="white"))
    )
    fig_pca.update_layout(template="plotly_dark", height=500, font=dict(size=13))
    st.plotly_chart(fig_pca, use_container_width=True)

    #Radar Charts
    st.subheader("Driver Style Radars")
    RADAR_LABELS = {
        "throttle_aggression":    "Corner Throttle",
        "brake_pressure":         "Brake Pressure",
        "coast_fraction":         "Coasting",
        "corner_entry_speed":     "Entry Speed",
        "traction_zone_throttle": "Traction Aggr.",
        "high_speed_stability":   "HS Stability",
        "brake_release_rate":     "Throttle Snap",
    }
    categories = [RADAR_LABELS[c] for c in FEATURE_COLUMNS]

    radar_df = features_df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        mn, mx = radar_df[col].min(), radar_df[col].max()
        radar_df[col] = (
            ((radar_df[col] - mn) / (mx - mn) * 100) if mx > mn else 50.0
        )

    col_r1, col_r2 = st.columns(2)
    for idx, driver in enumerate(result_df_fp["Driver"].tolist()[:8]):
        if driver not in radar_df.index:
            continue
        vals = radar_df.loc[driver, FEATURE_COLUMNS].tolist()
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        cluster_id = result_df_fp[result_df_fp["Driver"] == driver]["Cluster"].values[0]
        color = COLOR_MAP.get(cluster_id, "#FFFFFF")

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself", name=driver,
            line=dict(color=color, width=2),
            fillcolor=hex_to_rgba(color, alpha=0.2)
        ))
        fig_r.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
                bgcolor="rgba(0,0,0,0)"
            ),
            template="plotly_dark",
            title=dict(text=f"{driver} — Cluster {cluster_id}", font=dict(size=13)),
            height=280, margin=dict(t=50, b=10, l=20, r=20),
            showlegend=False
        )
        if idx % 2 == 0:
            with col_r1:
                st.plotly_chart(fig_r, use_container_width=True)
        else:
            with col_r2:
                st.plotly_chart(fig_r, use_container_width=True)

    #Cluster profiling  
    st.subheader("Cluster Profiles")
    for cluster_id, row in cluster_profiles.iterrows():
        label = label_cluster(row, all_profiles=cluster_profiles)
        drivers_in = result_df_fp[result_df_fp["ClusterInt"] == cluster_id]["Driver"].tolist()
        with st.expander(f"Cluster {cluster_id}: {label} — {', '.join(drivers_in)}"):
            st.dataframe(
                pd.DataFrame({"Feature": row.index, "Value": row.values.round(3)}),
                use_container_width=True
            )

    #Feature Discriminability
    st.subheader("Feature Discriminability")
    st.caption("Higher variance = this feature separates drivers more")
    variance = features_df[FEATURE_COLUMNS].var().sort_values(ascending=True)
    fig_var = go.Figure(go.Bar(
        x=variance.values,
        y=[RADAR_LABELS.get(c, c) for c in variance.index],
        orientation="h", marker_color="#E8002D"
    ))
    fig_var.update_layout(
        template="plotly_dark",
        xaxis_title="Variance",
        height=300, margin=dict(t=10)
    )
    st.plotly_chart(fig_var, use_container_width=True)