import fastf1
import pandas as pd
import streamlit as st
import os

#Enable FastF1 cache to avoid re-downloading data on every Streamlit rerun
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".fastf1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


@st.cache_resource(show_spinner=False)
def load_session(year=2023, gp="Bahrain", session_type="R"):

    session = fastf1.get_session(year, gp, session_type)
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    return session


@st.cache_resource(show_spinner=False)
def load_quali_session(year=2023, gp="Bahrain"):

    session = fastf1.get_session(year, gp, "Q")
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    return session


@st.cache_data(show_spinner=False)
def get_available_drivers(_session):

    laps = _session.laps
    valid = laps[laps["LapTime"].notna()]
    return sorted(valid["Driver"].unique().tolist())


@st.cache_data(show_spinner=False)
def get_fastest_lap_number(_session, driver_code):

    laps = _session.laps
    driver_laps = laps[
        (laps["Driver"] == driver_code) & laps["LapTime"].notna()
    ]
    if driver_laps.empty:
        return None
    return int(driver_laps.loc[driver_laps["LapTime"].idxmin(), "LapNumber"])


def get_laps(session):
    return session.laps
