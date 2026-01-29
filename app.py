import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="India AQI LSTM", layout="wide")
st.title("🇮🇳 India AQI Prediction (LSTM + CPCB)")

# ---------------- AQI (CPCB PM2.5) ----------------
def pm25_to_aqi(pm):
    if pm <= 30: return pm * 50 / 30
    if pm <= 60: return 50 + (pm - 30) * 50 / 30
    if pm <= 90: return 100 + (pm - 60) * 100 / 30
    if pm <= 120: return 200 + (pm - 90) * 100 / 30
    if pm <= 250: return 300 + (pm - 120) * 100 / 130
    return 400 + (pm - 250) * 100 / 130

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

# ---------------- LSTM MODEL ----------------
WINDOW = 5
features = ["PM2.5", "PM10", "NO2", "SO2", "CO"]

if len(city_df) <= WINDOW:
    st.warning("Not enough data for LSTM prediction")
    st.stop()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(city_df[features])

X, y = [], []
for i in range(len(scaled) - WINDOW):
    X.append(scaled[i:i+WINDOW])
    y.append(scaled[i+WINDOW][0])

X, y = np.array(X), np.array(y)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, len(features))),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=25, batch_size=4, verbose=0)

# ---------------- FORECAST ----------------
last_seq = scaled[-WINDOW:].reshape(1, WINDOW, len(features))
pred_scaled = model.predict(last_seq)[0][0]

pm25_pred = scaler.inverse_transform(
    [[pred_scaled, 0, 0, 0, 0]]
)[0][0]

aqi_pred = int(pm25_to_aqi(pm25_pred))

st.subheader("🤖 AI Forecast")
st.success(f"Predicted PM2.5: {pm25_pred:.1f}")
st.success(f"Predicted AQI: {aqi_pred}")

# ---------------- HEATMAP ----------------
st.subheader("🗺 India AQI Comparison")
heat = df.groupby("city")["PM2.5"].mean().reset_index()
heat["AQI"] = heat["PM2.5"].apply(pm25_to_aqi)

fig, ax = plt.subplots(figsize=(8,5))
ax.barh(heat["city"], heat["AQI"])
ax.set_xlabel("AQI")
st.pyplot(fig)

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
