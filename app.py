import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="India AQI LSTM Predictor",
    layout="wide"
)

st.title("🇮🇳 India AQI Prediction System (LSTM)")
st.caption("Multi-pollutant Deep Learning | CPCB AQI Standard")

# ---------------- CPCB AQI FUNCTION (PM2.5) ----------------
def calculate_aqi_pm25(pm):
    if pm <= 30: return pm * 50 / 30
    elif pm <= 60: return 50 + (pm - 30) * 50 / 30
    elif pm <= 90: return 100 + (pm - 60) * 100 / 30
    elif pm <= 120: return 200 + (pm - 90) * 100 / 30
    elif pm <= 250: return 300 + (pm - 120) * 100 / 130
    else: return 400 + (pm - 250) * 100 / 130

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload Air Quality CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ---------------- VALIDATION ----------------
    required_cols = {
        "date","city","PM2.5","PM10","NO2","SO2","CO"
    }

    if not required_cols.issubset(df.columns):
        st.error("❌ CSV columns do not match required format")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # ---------------- CITY SELECTION ----------------
    city = st.sidebar.selectbox("🏙️ Select City", sorted(df["city"].unique()))
    city_df = df[df["city"] == city].reset_index(drop=True)

    st.subheader(f"📊 Historical Pollution — {city}")
    st.line_chart(
        city_df.set_index("date")[["PM2.5","PM10","NO2","SO2","CO"]]
    )

    # ---------------- AQI DISPLAY ----------------
    city_df["AQI"] = city_df["PM2.5"].apply(calculate_aqi_pm25)
    latest_aqi = int(city_df["AQI"].iloc[-1])

    st.metric("📌 Current CPCB AQI", latest_aqi)

    # ---------------- LSTM PREPARATION ----------------
    features = ["PM2.5","PM10","NO2","SO2","CO"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(city_df[features])

    window = 5
    X, y = [], []

    for i in range(len(scaled) - window):
        X.append(scaled[i:i+window])
        y.append(scaled[i+window][0])  # Predict PM2.5

    X, y = np.array(X), np.array(y)

    # ---------------- LSTM MODEL ----------------
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=50, batch_size=4, verbose=0)

    # ---------------- PREDICTION ----------------
    last_window = scaled[-window:]
    prediction_scaled = model.predict(last_window.reshape(1, window, len(features)))
    pm25_pred = scaler.inverse_transform(
        np.hstack([prediction_scaled, np.zeros((1,4))])
    )[0][0]

    predicted_aqi = int(calculate_aqi_pm25(pm25_pred))

    st.success(f"🔮 Next AQI Prediction: **{predicted_aqi}**")

    # ---------------- HEATMAP ----------------
    st.subheader("🗺️ India AQI Heatmap")

    heat_df = df.groupby("city")["PM2.5"].mean().reset_index()
    heat_df["AQI"] = heat_df["PM2.5"].apply(calculate_aqi_pm25)

    fig, ax = plt.subplots()
    ax.barh(heat_df["city"], heat_df["AQI"])
    ax.set_xlabel("AQI")
    ax.set_title("Average AQI by City")
    st.pyplot(fig)

    # ---------------- DOWNLOAD ----------------
    output = city_df.copy()
    output["Predicted_Next_PM2.5"] = pm25_pred
    output["Predicted_Next_AQI"] = predicted_aqi

    st.download_button(
        "⬇️ Download City AQI Data",
        output.to_csv(index=False),
        file_name=f"{city}_AQI_Predictions.csv"
    )
