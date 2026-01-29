import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(page_title="India AQI Prediction", layout="wide")
st.title("🇮🇳 India AQI Prediction (ML + CPCB + Heatmap)")

# ---------------- CPCB AQI FUNCTION ----------------
def pm25_to_aqi(pm):
    if pm <= 30: return pm * 50 / 30
    if pm <= 60: return 50 + (pm - 30) * 50 / 30
    if pm <= 90: return 100 + (pm - 60) * 100 / 30
    if pm <= 120: return 200 + (pm - 90) * 100 / 30
    if pm <= 250: return 300 + (pm - 120) * 100 / 130
    return 400 + (pm - 250) * 100 / 130

# ---------------- CITY COORDINATES ----------------
CITY_COORDS = {
    "Delhi": (28.6139, 77.2090),
    "Noida": (28.5355, 77.3910),
    "Ghaziabad": (28.6692, 77.4538),
    "Gurugram": (28.4595, 77.0266),
    "Great Noida": (28.4744, 77.5040),
    "Bhiwadi": (28.2100, 76.8600),
    "Hapur": (28.7306, 77.7800),
    "Muzaffarnagar": (29.4727, 77.7085),
    "Hajipur": (25.6925, 85.2082),
    "Byrnihat": (26.0333, 91.8667),
}

# ---------------- UPLOAD CSV ----------------
file = st.file_uploader("📂 Upload Air Quality CSV", type="csv")

if not file:
    st.info("Upload your air_quality.csv to continue")
    st.stop()

# ---------------- LOAD DATA ----------------
df = pd.read_csv(file)
df.columns = df.columns.str.strip()
df["date"] = pd.to_datetime(df["date"])

# ---------------- SIDEBAR ----------------
city = st.sidebar.selectbox("🏙 Select City", sorted(df["city"].unique()))
days = st.sidebar.slider("📅 Predict next days", 1, 7, 3)

city_df = df[df["city"] == city].sort_values("date").reset_index(drop=True)

# ---------------- METRICS ----------------
latest_pm = city_df["PM2.5"].iloc[-1]
latest_aqi = int(pm25_to_aqi(latest_pm))

col1, col2 = st.columns(2)
col1.metric("Latest PM2.5", round(latest_pm, 1))
col2.metric("Latest AQI", latest_aqi)

# ---------------- HISTORICAL GRAPH ----------------
st.subheader(f"📈 Historical Pollution — {city}")
st.line_chart(
    city_df.set_index("date")[["PM2.5", "PM10", "NO2", "SO2", "CO"]]
)

# ---------------- ML PREDICTION ----------------
st.subheader("🤖 AI Forecast (Linear Regression)")

city_df["t"] = np.arange(len(city_df))
X = city_df[["t"]]
y = city_df["PM2.5"]

model = LinearRegression()
model.fit(X, y)

future_t = np.array([[len(city_df) + days]])
pm25_pred = model.predict(future_t)[0]
aqi_pred = int(pm25_to_aqi(pm25_pred))

st.success(f"Predicted PM2.5 after {days} days: {pm25_pred:.1f}")
st.success(f"Predicted AQI: {aqi_pred}")

# ---------------- INDIA AQI HEATMAP ----------------
st.subheader("🗺 India AQI Heatmap")

heat = df.groupby("city")["PM2.5"].mean().reset_index()
heat["AQI"] = heat["PM2.5"].apply(pm25_to_aqi)

heat["lat"] = heat["city"].map(lambda x: CITY_COORDS.get(x, (None, None))[0])
heat["lon"] = heat["city"].map(lambda x: CITY_COORDS.get(x, (None, None))[1])
heat = heat.dropna()

fig = px.scatter_mapbox(
    heat,
    lat="lat",
    lon="lon",
    size="AQI",
    color="AQI",
    hover_name="city",
    color_continuous_scale="turbo",
    zoom=4,
    height=600
)

fig.update_layout(
    mapbox_style="carto-darkmatter",
    margin={"r":0,"t":0,"l":0,"b":0}
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- DOWNLOAD ----------------
output = city_df.copy()
output["Predicted_PM2.5"] = pm25_pred
output["Predicted_AQI"] = aqi_pred

st.download_button(
    label="📥 Download Prediction CSV",
    data=output.to_csv(index=False),
    file_name=f"{city}_AQI_prediction.csv",
    mime="text/csv"
)
