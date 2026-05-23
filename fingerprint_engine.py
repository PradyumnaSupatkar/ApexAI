import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit as st


FEATURE_COLUMNS = [
    "throttle_aggression",   
    "brake_pressure",         
    "coast_fraction",         
    "corner_entry_speed",     
    "traction_zone_throttle", 
    "high_speed_stability",   
    "brake_release_rate",     
]


@st.cache_data(show_spinner=False)
def extract_driver_features(_session):
    
    laps = _session.laps
    valid_laps = laps[laps["LapTime"].notna()]

    records = []

    for driver in valid_laps["Driver"].unique():
        driver_laps = valid_laps[valid_laps["Driver"] == driver]
        lap_features = []

        for _, lap in driver_laps.iterrows():
            try:
                tel = lap.get_telemetry()
                if tel is None or tel.empty:
                    continue

                speed    = tel["Speed"].to_numpy(dtype=float)
                throttle = tel["Throttle"].to_numpy(dtype=float)
                brake    = tel["Brake"].astype(bool).to_numpy()

                if len(speed) < 80:
                    continue

                #Throttle aggression in corners ──────────────────────
                corner_mask = speed < 200
                throttle_aggression = (
                    throttle[corner_mask].mean()
                    if corner_mask.sum() > 10 else np.nan
                )

                #Brake pressure proxy ────────────────────────────────
                brake_starts = np.where(np.diff(brake.astype(int)) == 1)[0]
                drops = []
                for bs in brake_starts:
                    if bs >= 5:
                        pre = throttle[bs - 5:bs]
                        drops.append(pre.mean() - throttle[bs])
                brake_pressure = np.mean(drops) if drops else np.nan

                #Coast fraction
                coast_mask = (throttle < 5) & (~brake)
                coast_fraction = coast_mask.mean()

                #Corner entry speed
                p20 = np.percentile(speed, 20)
                low_speed_mask = speed <= p20
                corner_entry_speed = (
                    speed[low_speed_mask].mean()
                    if low_speed_mask.sum() > 5 else np.nan
                )

                #Traction zone throttle
                traction_mask = (speed >= 50) & (speed <= 150)
                traction_zone_throttle = (
                    throttle[traction_mask].mean()
                    if traction_mask.sum() > 10 else np.nan
                )

                #High speed stability
                hs_mask = speed > 250
                high_speed_stability = (
                    speed[hs_mask].std()
                    if hs_mask.sum() > 10 else np.nan
                )

                #Brake-to-throttle transition rate
                brake_ends = np.where(np.diff(brake.astype(int)) == -1)[0]
                post_brake = []
                for be in brake_ends:
                    if be + 4 < len(throttle):
                        post_brake.append(throttle[be + 1:be + 4].mean())
                brake_release_rate = np.mean(post_brake) if post_brake else np.nan

                feat = {
                    "throttle_aggression":    throttle_aggression,
                    "brake_pressure":         brake_pressure,
                    "coast_fraction":         coast_fraction,
                    "corner_entry_speed":     corner_entry_speed,
                    "traction_zone_throttle": traction_zone_throttle,
                    "high_speed_stability":   high_speed_stability,
                    "brake_release_rate":     brake_release_rate,
                }

                valid_count = sum(
                    1 for v in feat.values()
                    if v is not None and not np.isnan(v)
                )
                if valid_count >= 5:
                    lap_features.append(feat)

            except Exception:
                continue

        if not lap_features:
            continue

        agg = pd.DataFrame(lap_features).mean().to_dict()
        agg["Driver"] = driver
        records.append(agg)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index("Driver")


def run_fingerprinting(features_df, n_clusters=3):

    if features_df.empty or len(features_df) < n_clusters:
        return None, None, None

    X = features_df[FEATURE_COLUMNS].dropna()
    if len(X) < n_clusters:
        return None, None, None

    drivers = X.index.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    result_df = pd.DataFrame({
        "Driver":  drivers,
        "PC1":     coords[:, 0],
        "PC2":     coords[:, 1],
        "Cluster": clusters.astype(str),
        **{col: X[col].values for col in FEATURE_COLUMNS}
    })

    explained = pca.explained_variance_ratio_ * 100

    result_df["ClusterInt"] = clusters
    cluster_profiles = (
        result_df.groupby("ClusterInt")[FEATURE_COLUMNS]
        .mean()
        .round(3)
    )

    return result_df, explained, cluster_profiles


def label_cluster(profile_row, all_profiles):
    cluster_ids = list(all_profiles.index)

    if len(cluster_ids) < 2:
        return "⚪ Balanced"

    def col_rank(col, ascending=False):
        """ascending=False → rank 1 = highest value (best for that trait)."""
        return all_profiles[col].rank(method="first", ascending=ascending)

    # Combined rank scores per axis
    axis_scores = {
        "Aggression":   col_rank("throttle_aggression")            + col_rank("traction_zone_throttle"),
        "Late Braker":  col_rank("brake_pressure")                 + col_rank("corner_entry_speed"),
        "Conservative": col_rank("coast_fraction")                 + col_rank("throttle_aggression", ascending=True),
        "Smooth":       col_rank("high_speed_stability", ascending=True) + col_rank("brake_release_rate"),
    }


    assigned = {}
    remaining_clusters = list(cluster_ids)
    remaining_axes     = list(axis_scores.keys())

    while remaining_clusters and remaining_axes:
        best_score = 999
        best_pair  = None
        for cid in remaining_clusters:
            for ax in remaining_axes:
                s = axis_scores[ax][cid]
                if s < best_score:
                    best_score = s
                    best_pair  = (cid, ax)
        if best_pair:
            cid, ax = best_pair
            assigned[cid] = ax
            remaining_clusters.remove(cid)
            remaining_axes.remove(ax)

    # Any extra clusters beyond 4 get "Balanced"
    for cid in remaining_clusters:
        assigned[cid] = "Balanced"

    axis = assigned.get(profile_row.name, "Balanced")

    LABELS = {
        "Aggression":   "🔴 Aggressive — Early throttle, high traction aggression",
        "Late Braker":  "🟠 Late Braker — Carries speed deep, hard braking",
        "Conservative": "🟡 Conservative — High coasting, measured throttle",
        "Smooth":       "🟢 Smooth & Consistent — Stable high-speed, clean exits",
        "Balanced":     "⚪ Balanced — No dominant trait",
    }
    return LABELS[axis]