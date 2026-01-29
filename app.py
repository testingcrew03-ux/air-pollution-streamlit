import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="India Air Quality Predictor",
    page_icon="🌍",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("air_quality.csv")

    # 🔧 Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Required columns (lowercase)
    required_cols = ["city", "date", "pm2.5", "pm10", "no2", "so2", "co"]

    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ Missing column in CSV: {col}")
            st.stop()

    # Clean values
    df["city"] = df["city"].astype(str).str.strip().str.title()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date"])
    return df


df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎛 Controls")

cities = sorted(df["city"].unique())
selected_city = st.sidebar.selectbox("Select City", cities)

pollutants = ["pm2.5", "pm10", "no2", "so2", "co"]
selected_pollutant = st.sidebar.selectbox(
    "Select Pollutant",
    pollutants,
    format_func=lambda x: x.upper()
)

predict_days = st.sidebar.slider("Predict Next Days", 1, 14, 7)

# ---------------- FILTER DATA ----------------
city_df = df[df["city"] == selected_city].sort_values("date")

latest_value = city_df[selected_pollutant].iloc[-1]
last_updated = city_df["date"].max().date()

# ---------------- HEADER ----------------
st.markdown(
    f"""
    <h2>📍 Current Air Quality in {selected_city}</h2>
    <p style="color:gray">ML-powered • Health-aware • India-wide</p>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

col1.metric("Pollutant", selected_pollutant.upper())
col2.metric("Latest Value", f"{latest_value:.1f}")
col3.metric("Last Updated", str(last_updated))

# ---------------- HEALTH ALERT ----------------
def health_alert(val):
    if val <= 50:
        return "🟢 Good – Safe"
    elif val <= 100:
        return "🟡 Moderate"
    elif val <= 200:
        return "🟠 Poor"
    elif val <= 300:
        return "🔴 Very Poor"
    else:
        return "⚫ Severe"

st.warning(f"🚨 Health Advisory: **{health_alert(latest_value)}**")

# ---------------- HISTORICAL TREND ----------------
st.subheader("📊 Historical Trend")
st.line_chart(city_df.set_index("date")[selected_pollutant])

# ---------------- ML PREDICTION ----------------
st.subheader("🤖 ML Prediction")

city_df = city_df.copy()
city_df["day_index"] = np.arange(len(city_df))

X = city_df[["day_index"]]
y = city_df[selected_pollutant]

model = LinearRegression()
model.fit(X, y)

future_index = np.arange(len(city_df), len(city_df) + predict_days)
future_dates = [
    city_df["date"].max() + timedelta(days=i + 1)
    for i in range(predict_days)
]

predictions = model.predict(future_index.reshape(-1, 1))

prediction_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Value": predictions
})

# ---------------- PREDICTION CHART ----------------
st.line_chart(prediction_df.set_index("Date"))

# ---------------- DOWNLOAD ----------------
st.subheader("📥 Download Predictions")

csv = prediction_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Prediction CSV",
    csv,
    f"{selected_city}_{selected_pollutant}_prediction.csv",
    "text/csv"
)

# ---------------- TABLE ----------------
with st.expander("📄 View Prediction Table"):
    st.dataframe(prediction_df)

# ---------------- FOOTER ----------------
st.markdown(
    "<hr><center>🌱 India Air Quality Predictor • Streamlit + ML</center>",
    unsafe_allow_html=True
)
