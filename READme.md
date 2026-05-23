# ApexAI

ApexAI is a motorsports telemetry analytics and ML pace forecasting platform built using Formula 1 telemetry data.  
It combines driver telemetry comparison, lap delta analysis, machine learning based pace prediction, and unsupervised driver fingerprinting in an interactive Streamlit dashboard.

## Overview

ApexAI transforms raw racing telemetry into practical performance insights. The platform allows users to compare drivers, analyze lap-level behavior, forecast race pace, and group drivers based on driving-style patterns extracted from telemetry data.

The project is built to demonstrate applied machine learning, feature engineering, data visualization, and domain-specific analytics using real motorsports data.

## Key Features

### Telemetry and Delta Analysis
- Compare two drivers across the same clean race lap.
- Visualize speed, throttle, and brake traces over distance.
- Compute lap delta to identify where one driver gained or lost time.
- Filter noisy laps by removing pit-in, pit-out, and abnormal lap records.

### ML Race Pace Forecasting
- Trains a Random Forest regression model on historical race lap data.
- Predicts lap-time behavior for a selected driver and race session.
- Displays actual vs predicted lap times with compound-based visualization.
- Reports model performance using MAE and R² score.

### Driver Fingerprinting
- Extracts behavioral driving-style features from raw telemetry.
- Uses K-Means clustering to group similar driver behavior patterns.
- Applies PCA for two-dimensional cluster visualization.
- Displays radar charts for driver-specific telemetry profiles.

## Machine Learning Techniques Used

- Feature engineering from telemetry and lap timing data
- Time-series style lap analysis
- Random Forest regression for lap-time and pace forecasting
- K-Means clustering for unsupervised driver behavior grouping
- PCA for cluster visualization
- Model evaluation using MAE and R² score

## Tech Stack

- Python
- Streamlit
- FastF1
- Pandas
- NumPy
- Scikit-learn
- Plotly

