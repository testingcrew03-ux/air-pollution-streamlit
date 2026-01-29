import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

st.set_page_config(page_title="India AQI LSTM", layout="wide")
st.title("🇮🇳 India AQI Prediction (LSTM)")

# ---------- CPCB AQI ----------
def pm25_aqi(pm):
    if pm <= 30: return pm * 50 / 30
    if pm <= 60: return 50 + (pm - 30) * 50 / 30
    if pm <= 90: return 100 + (pm - 60) * 100 / 30
    if pm <= 120: return 200 + (pm - 90) * 100 / 30
    if pm <= 250: return 300 + (pm - 120) * 100 / 130
    return 400 + (pm - 250) * 100 / 130

# ---------- Upload ----------
file = st.file_uploader("Upload AQI CSV", type="csv")

if file:
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"])

    city = st.sidebar.selectbox("Select City", sorted(df["city"].unique()))
    city_df = df[df["city"] == city].sort_values("date").reset_index(drop=True)

    st.subheader(f"Historical Pollution — {city}")
    st.line_chart(city_df.set_index("date")[["PM2.5","PM10","NO2","SO2","CO"]])

    city_df["AQI"] = city_df["PM2.5"].apply(pm25_aqi)
    st.metric("Latest AQI", int(city_df["AQI"].iloc[-1]))

    WINDOW = 5
    if len(city_df) <= WINDOW:
        st.warning("Not enough data for LSTM")
        st.stop()

    features = ["PM2.5","PM10","NO2","SO2","CO"]
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
    model.fit(X, y, epochs=30, batch_size=4, verbose=0)

    last = scaled[-WINDOW:].reshape(1, WINDOW, len(features))
    pred = model.predict(last)[0][0]

    pm25_pred = scaler.inverse_transform(
        np.hstack([[pred, 0, 0, 0, 0]])
    )[0]

    aqi_pred = int(pm25_aqi(pm25_pred))
    st.success(f"Predicted Next AQI: {aqi_pred}")

    # ---------- Heatmap ----------
    st.subheader("India AQI Heatmap")
    heat = df.groupby("city")["PM2.5"].mean().reset_index()
    heat["AQI"] = heat["PM2.5"].apply(pm25_aqi)

    fig, ax = plt.subplots()
    ax.barh(heat["city"], heat["AQI"])
    ax.set_xlabel("AQI")
    st.pyplot(fig)

    # ---------- Download ----------
    output = city_df.copy()
    output["Predicted_PM2.5"] = pm25_pred
    output["Predicted_AQI"] = aqi_pred

    st.download_button(
        "Download AQI CSV",
        output.to_csv(index=False),
        file_name=f"{city}_AQI.csv",
        mime="text/csv"
    )
