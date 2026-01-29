import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="India AQI Prediction", layout="wide")
st.title("🇮🇳 IN India AQI Prediction (ML + CPCB + Heatmap)")

# ---------------- LOAD CSV DIRECTLY ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("air_quality.csv")
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# ---------------- CPCB AQI (PM2.5) ----------------
def pm25_to_aqi(pm):
    if pm <= 30: return pm * 50 / 30
    if pm <= 60: return 50 + (pm - 30) * 50 / 30
    if pm <= 90: return 100 + (pm - 60) * 100 / 30
    if pm <= 120: return 200 + (pm - 90) * 100 / 30
    if pm <= 250: return 300 + (pm - 120) * 100 / 130
    return 400 + (pm - 250) * 100 / 130

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")
city = st.sidebar.selectbox("🏙 Select City", sorted(df["city"].unique()))
future_days = st.sidebar.slider("📅 Predict Days Ahead", 1, 7, 3)

city_df = df[df["city"] == city].sort_values("date").reset_index(drop=True)

# ---------------- METRICS ----------------
latest_pm = city_df["PM2.5"].iloc[-1]
latest_aqi = int(pm25_to_aqi(latest_pm))

col1, col2 = st.columns(2)
col1.metric("Latest PM2.5", round(latest_pm, 1))
col2.metric("Latest AQI (CPCB)", latest_aqi)

# ---------------- HISTORICAL TREND ----------------
st.subheader(f"📈 Pollution Trend — {city}")
st.line_chart(
    city_df.set_index("date")[["PM2.5", "PM10", "NO2", "SO2", "CO"]]
)

# ---------------- ML PREDICTION (LINEAR REGRESSION) ----------------
st.subheader("🤖 ML Prediction")

city_df["day_index"] = np.arange(len(city_df))
X = city_df[["day_index"]]
y = city_df["PM2.5"]

model = LinearRegression()
model.fit(X, y)

future_index = city_df["day_index"].iloc[-1] + future_days
pm25_pred = model.predict([[future_index]])[0]
aqi_pred = int(pm25_to_aqi(pm25_pred))

st.success(f"Predicted PM2.5 after {future_days} days: **{pm25_pred:.1f}**")
st.success(f"Predicted AQI (CPCB): **{aqi_pred}**")

# ---------------- INDIA AQI HEATMAP ----------------
st.subheader("🗺 India AQI Heatmap (City-wise Average)")

heat_df = df.groupby("city")["PM2.5"].mean().reset_index()
heat_df["AQI"] = heat_df["PM2.5"].apply(pm25_to_aqi)

# Approx city coordinates (lightweight, no API)
city_coords = {
    "Delhi": [28.61, 77.21],
    "Noida": [28.53, 77.39],
    "Ghaziabad": [28.67, 77.42],
    "Gurugram": [28.46, 77.03],
    "Great Noida": [28.47, 77.50],
    "Bhiwadi": [28.21, 76.86],
    "Hapur": [28.73, 77.78],
    "Muzaffarnagar": [29.47, 77.70],
    "Hajipur": [25.69, 85.21],
    "Byrnihat": [26.04, 91.88]
}

heat_df["lat"] = heat_df["city"].map(lambda x: city_coords[x][0])
heat_df["lon"] = heat_df["city"].map(lambda x: city_coords[x][1])

st.map(heat_df[["lat", "lon", "AQI"]])

# ---------------- BAR COMPARISON ----------------
st.subheader("📊 AQI Comparison Across Cities")

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(heat_df["city"], heat_df["AQI"])
ax.set_xlabel("AQI")
ax.set_title("Average AQI by City")
st.pyplot(fig)

# ---------------- DOWNLOAD ----------------
output = city_df.copy()
output["Predicted_PM2.5"] = pm25_pred
output["Predicted_AQI"] = aqi_pred

st.download_button(
    "📥 Download City Prediction CSV",
    data=output.to_csv(index=False),
    file_name=f"{city}_AQI_prediction.csv",
    mime="text/csv"
)
