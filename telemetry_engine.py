import fastf1
import pandas as pd
import streamlit as st


def get_driver_lap(session, driver_code, lap_number):

    laps = session.laps
    result = laps[
        (laps["Driver"] == driver_code) &
        (laps["LapNumber"] == lap_number) &
        (laps["LapTime"].notna())
    ]

    if result.empty:
        return None

    return result.iloc[0]


@st.cache_data(show_spinner=False)
def get_telemetry_for_lap(_session, driver_code, lap_number):

    laps = _session.laps
    result = laps[
        (laps["Driver"] == driver_code) &
        (laps["LapNumber"] == lap_number) &
        (laps["LapTime"].notna())
    ]

    if result.empty:
        return pd.DataFrame()

    lap = result.iloc[0]
    tel = lap.get_telemetry()

    if tel is None or tel.empty:
        return pd.DataFrame()

    df = pd.DataFrame({
        "Time": tel["Time"],
        "Speed": tel["Speed"],
        "Throttle": tel["Throttle"],
        "Brake": tel["Brake"],
        "Distance": tel["Distance"]
    })

    df = df.dropna(subset=["Speed", "Distance"])
    return df
