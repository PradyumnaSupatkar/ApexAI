import pandas as pd
import numpy as np


def compute_time_from_distance(df):
    df = df.copy()

    df = df.sort_values("Distance").reset_index(drop=True)

    df["Speed_mps"] = df["Speed"] * (1000 / 3600)

    df["Speed_mps"] = df["Speed_mps"].replace(0, np.nan)
    df["Speed_mps"] = df["Speed_mps"].ffill().bfill()

    df["DeltaDistance"] = df["Distance"].diff().fillna(0)

    df["DeltaTime"] = df["DeltaDistance"] / df["Speed_mps"]
    df["DeltaTime"] = df["DeltaTime"].fillna(0)

    df["CumulativeTime"] = df["DeltaTime"].cumsum()

    return df


def compute_delta(lap1_df, lap2_df):

    if lap1_df.empty or lap2_df.empty:
        return pd.DataFrame({"Distance": [], "Delta": []})

    lap1 = compute_time_from_distance(lap1_df)
    lap2 = compute_time_from_distance(lap2_df)

    lap1 = lap1.drop_duplicates(subset=["Distance"])
    lap2 = lap2.drop_duplicates(subset=["Distance"])

    max_dist = min(lap1["Distance"].max(), lap2["Distance"].max())
    common_distance = np.linspace(0, max_dist, 500)

    lap1_time = np.interp(common_distance, lap1["Distance"], lap1["CumulativeTime"])
    lap2_time = np.interp(common_distance, lap2["Distance"], lap2["CumulativeTime"])

    delta = lap1_time - lap2_time

    return pd.DataFrame({
        "Distance": common_distance,
        "Delta": delta
    })
