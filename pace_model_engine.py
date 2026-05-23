import fastf1
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD"]

TIRE_COLORS = {
    "SOFT":   "#E8002D",
    "MEDIUM": "#FFF200",
    "HARD":   "#FFFFFF",
}

MIN_ROWS_FOR_MODEL = 50

NUMERIC_FEATURES = [
    "TyreLife",
    "StintLapIndex",
    "PrevLap",
    "LapDelta",
    "RollingMean3",
    "RollingStd3",
    "StintProgress",
]


def get_all_feature_names() -> list:
    return NUMERIC_FEATURES + [f"Compound_{c}" for c in COMPOUND_ORDER]


#LapFormat
def format_laptime(seconds: float) -> str:
    """95.437 → '1:35.437'"""
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)) or seconds < 0:
        return "N/A"
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    secs = int(remainder)
    millis = int(round((remainder - secs) * 1000))
    if millis >= 1000:
        secs += 1
        millis -= 1000
    return f"{minutes}:{secs:02d}.{millis:03d}"



def encode_features(df: pd.DataFrame) -> np.ndarray:
    
    num = df[NUMERIC_FEATURES].to_numpy(dtype=np.float64)

    
    compound_arr = df["Compound"].to_numpy()
    ohe = np.zeros((len(df), len(COMPOUND_ORDER)), dtype=np.float64)
    for i, c in enumerate(COMPOUND_ORDER):
        ohe[:, i] = (compound_arr == c).astype(np.float64)

    return np.hstack([num, ohe])



def _clean_session_laps(session) -> pd.DataFrame:
    laps = session.laps.copy()
    laps = laps[laps["LapTime"].notna()].copy()
    if laps.empty:
        return pd.DataFrame()

    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

    # Remove pit laps
    pit_mask = laps["PitInTime"].notna() | laps["PitOutTime"].notna()
    laps = laps[~pit_mask].copy()

    # Remove outlaps
    first_per_stint = laps.groupby(["Driver", "Stint"])["LapNumber"].min().to_dict()
    outlap_mask = laps.apply(
        lambda r: r["LapNumber"] == first_per_stint.get((r["Driver"], r["Stint"]), -1),
        axis=1
    )
    laps = laps[~outlap_mask].copy()

    # Dry compounds only
    laps = laps[laps["Compound"].isin(COMPOUND_ORDER)].copy()
    if laps.empty:
        return pd.DataFrame()

    # Per-driver SC/VSC filter
    clean_frames = []
    for drv in laps["Driver"].unique():
        dl = laps[laps["Driver"] == drv].copy()
        med = dl["LapTime_s"].median()
        dl = dl[(dl["LapTime_s"] <= med * 1.10) & (dl["LapTime_s"] >= med * 0.94)]
        if not dl.empty:
            clean_frames.append(dl)

    if not clean_frames:
        return pd.DataFrame()

    laps = (
        pd.concat(clean_frames, ignore_index=True)
        .sort_values(["Driver", "LapNumber"])
        .reset_index(drop=True)
    )

    # Feature Engineering
    laps["StintLapIndex"] = laps.groupby(["Driver", "Stint"]).cumcount()
    laps["PrevLap"]       = laps.groupby("Driver")["LapTime_s"].shift(1)
    laps["LapDelta"]      = laps.groupby("Driver")["LapTime_s"].diff()
    laps["RollingMean3"]  = laps.groupby("Driver")["LapTime_s"].transform(
        lambda x: x.rolling(3, min_periods=2).mean()
    )
    laps["RollingStd3"]   = laps.groupby("Driver")["LapTime_s"].transform(
        lambda x: x.rolling(3, min_periods=2).std().fillna(0)
    )
    stint_max = laps.groupby(["Driver", "Stint"])["TyreLife"].transform("max")
    laps["StintProgress"] = laps["TyreLife"] / stint_max.replace(0, np.nan)

    keep = ["Driver", "LapNumber", "Stint", "TyreLife", "Compound",
            "LapTime_s"] + NUMERIC_FEATURES
    laps = laps[[c for c in keep if c in laps.columns]].dropna()
    return laps.reset_index(drop=True)


#Season dataset builder
@st.cache_data(show_spinner=True)
def build_season_dataset(season: int, gp_list: list) -> pd.DataFrame:
    all_frames = []
    for gp_name in gp_list:
        try:
            sess = fastf1.get_session(season, gp_name, "R")
            sess.load(laps=True, telemetry=False, weather=False, messages=False)
            frame = _clean_session_laps(sess)
            if not frame.empty:
                frame["Season"] = season
                frame["Event"]  = gp_name
                all_frames.append(frame)
        except Exception:
            pass
    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True).reset_index(drop=True)


#Model training
def train_global_model(train_df: pd.DataFrame):
    if len(train_df) < MIN_ROWS_FOR_MODEL:
        return None

    X = encode_features(train_df)                          
    y = train_df["LapTime_s"].to_numpy(dtype=np.float64)  

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y) 
    return rf


#Prediction
def predict_for_driver(model, test_df: pd.DataFrame, driver_code: str):
    driver_df = (
        test_df[test_df["Driver"] == driver_code]
        .reset_index(drop=True)
        .copy()
    )
    if driver_df.empty or len(driver_df) < 3:
        return None, None

    X = encode_features(driver_df)
    predictions = model.predict(X)

    driver_df["Predicted_s"] = predictions
    driver_df["PredictedSmooth"] = (
        pd.Series(predictions)
        .rolling(3, center=True, min_periods=1)
        .mean()
        .values
    )

    mae = mean_absolute_error(driver_df["LapTime_s"], predictions)
    r2  = r2_score(driver_df["LapTime_s"], predictions) if len(driver_df) > 1 else float("nan")

    return driver_df, {"mae": mae, "r2": r2, "n_laps": len(driver_df)}


#Feature importance
def get_feature_importance(model) -> pd.DataFrame:
    names = get_all_feature_names()
    imps  = model.feature_importances_
    return (
        pd.DataFrame({"Feature": names[:len(imps)], "Importance": imps[:len(names)]})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
