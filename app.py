import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="India AQI AI Platform",
    layout="wide"
)

st.markdown("""
<style>
body { background-color: #0E1117; color:white; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌍 Controls")

city = st.sidebar.selectbox(
    "Select City",
    sorted(df["City"].unique())
)

days = st.sidebar.slider(
    "Predict Next Days",
    3, 14, 7
)

city_df = df[df["City"] == city].copy()

# ---------------- HEADER ----------------
st.title(f"📍 Air Quality AI Forecast — {city}")

latest = city_df.iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("PM2.5", round(latest["PM2.5"], 1))
col2.metric("PM10", round(latest["PM10"], 1))
col3.metric("NO2", round(latest["NO2"], 1))

st.caption(f"Last Updated: {latest['Date'].date()}")

# ---------------- HISTORICAL CHART ----------------
st.subheader("📈 Historical Pollutant Trend")

fig = px.line(
    city_df,
    x="Date",
    y=["PM2.5", "PM10", "NO2"],
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- MULTI-INPUT LSTM ----------------
st.subheader("🤖 AI Forecast (Multi-Pollutant LSTM)")

features = city_df[["PM2.5", "PM10", "NO2"]].values

scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)

SEQ_LEN = 5
X, y = [], []

for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i+SEQ_LEN])
    y.append(scaled[i+SEQ_LEN, 0])  # Predict PM2.5

X = np.array(X)
y = np.array(y)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 3)),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=60, batch_size=8, verbose=0)

# ---------------- FUTURE PREDICTION ----------------
last_seq = scaled[-SEQ_LEN:]
future_preds = []

for _ in range(days):
    pred = model.predict(last_seq.reshape(1, SEQ_LEN, 3), verbose=0)[0][0]
    future_preds.append(pred)

    next_row = last_seq[-1].copy()
    next_row[0] = pred  # PM2.5
    last_seq = np.vstack([last_seq[1:], next_row])

future_pm25 = scaler.inverse_transform(
    np.column_stack([
        future_preds,
        [last_seq[-1][1]] * days,
        [last_seq[-1][2]] * days
    ])
)[:, 0]

future_dates = pd.date_range(
    city_df["Date"].max() + pd.Timedelta(days=1),
    periods=days
)

pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted PM2.5": future_pm25
})

st.dataframe(pred_df, use_container_width=True)

st.download_button(
    "📥 Download Prediction CSV",
    pred_df.to_csv(index=False),
    "pm25_lstm_predictions.csv",
    "text/csv"
)

# ---------------- FORECAST CHART ----------------
fig2 = px.line(
    pred_df,
    x="Date",
    y="Predicted PM2.5",
    markers=True,
    title="AI Forecasted PM2.5"
)

st.plotly_chart(fig2, use_container_width=True)
