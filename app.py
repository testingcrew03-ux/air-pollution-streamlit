import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

st.set_page_config(page_title="India AQI LSTM", layout="wide")
st.title("🇮🇳 India AQI Prediction System (LSTM)")

# ---------------- CPCB AQI (PM2.5) ----------------
def calculate_aqi_pm25(pm):
    if pm <= 30: return pm * 50 / 30
    elif pm <= 60: return 50 + (pm - 30) * 50 / 30
    elif pm <= 90: return 100 + (pm - 60) * 100 / 30
    elif pm <= 120: return 200 + (pm - 90) * 100 / 30
    elif pm <= 250: return 300 + (pm - 120) * 100 / 130
    else: return 400 + (pm - 250) * 100 / 130

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("📂 Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"])

    city = st.sidebar.selectbox("🏙️ Select City", sorted(df["city"].unique()))
    city_df = df[df["city"] == city].sort_values("date").reset_index(drop=True)

    st.subheader(f"📊 Historical Data — {city}")
    st.line_chart(city_df.set_index("date")[["PM2.5","PM10","NO2","SO2","CO"]])

    # ---------------- AQI ----------------
    city_df["AQI"] = city_df["PM2.5"].apply(calculate_aqi_pm25)
    st.metric("📌 Latest AQI", int(city_df["AQI"].iloc[-1]))

    # ---------------- LSTM SAFETY CHECK ----------------
    WINDOW = 5
    if len(city_df) <= WINDOW:
        st.warning(
            f"⚠️ Not enough data for LSTM (need > {WINDOW} rows, found {len(city_df)})"
        )
        st.stop()

    # ---------------- LSTM DATA ----------------
    features = ["PM2.5","PM10","NO2","SO2","CO"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(city_df[features])

    X, y = [], []
    for i in range(len(scaled) - WINDOW):
        X.append(scaled[i:i+WINDOW])
        y.append(scaled[i+WINDOW][0])

    X, y = np.array(X), np.array(y)

    # ---------------- LSTM MODEL ----------------
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW, len(features))),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=40, batch_size=4, verbose=0)

    # ---------------- PREDICTION ----------------
    last_window = scaled[-WINDOW:]
    pred_scaled = model.predict(last_window.reshape(1, WINDOW, len(features)))
    pm25_pred = scaler.inverse_transform(
        np.hstack([pred_scaled, np.zeros((1,4))])
    )[0][0]

    predicted_aqi = int(calculate_aqi_pm25(pm25_pred))

    st.success(f"🔮 Next Predicted AQI: **{predicted_aqi}**")

    # ---------------- HEATMAP ----------------
    st.subheader("🗺️ Average AQI by City")
    heat = df.groupby("city")["PM2.5"].mean().reset_index()
    heat["AQI"] = heat["PM2.5"].apply(calculate_aqi_pm25)

    fig, ax = plt.subplots()
    ax.barh(heat["city"], heat["AQI"])
    ax.set_xlabel("AQI")
    st.pyplot(fig)

    # ---------------- DOWNLOAD ----------------
    out = city_df.copy()
    out["Predicted_Next_PM2.5"] = pm25_pred
    out["Predicted_Next_AQI"] = predicted_aqi

    st.download_button(
        "⬇️ Download City AQI CSV",
