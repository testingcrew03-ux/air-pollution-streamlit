import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Air Pollution Dashboard",
    layout="wide",
    page_icon="🌍"
)

# ---------------- TITLE ----------------
st.title("🌍 AI Air Pollution Monitoring & Prediction System")
st.caption("India • ML-powered • Health-aware • Real-time insights")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("air_quality.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.header("🎛 Controls")

city = st.sidebar.selectbox(
    "Select City",
    sorted(df["city"].unique())
)

pollutant = st.sidebar.selectbox(
    "Select Pollutant",
    ["PM2.5", "PM10", "NO2", "SO2", "CO"]
)

predict_days = st.sidebar.slider(
    "Predict Next Days",
    3, 30, 7
)

# ---------------- FILTER DATA ----------------
city_df = df[df["city"] == city].sort_values("date")

latest_value = city_df[pollutant].iloc[-1]

# ---------------- CURRENT STATUS ----------------
st.subheader(f"📍 Current Air Quality in {city}")

col1, col2, col3 = st.columns(3)
col1.metric("Pollutant", pollutant)
col2.metric("Latest Value", round(latest_value, 2))
col3.metric("Last Updated", city_df["date"].max().date())

# ---------------- HEALTH ALERTS ----------------
def health_status(value, pollutant):
    if pollutant in ["PM2.5", "PM10"]:
        if value <= 50:
            return "🟢 Good – Safe to go outdoors"
        elif value <= 100:
            return "🟡 Moderate – Sensitive people be careful"
        else:
            return "🔴 Poor – Avoid outdoor exposure"
    else:
        if value <= 40:
            return "🟢 Normal"
        elif value <= 80:
            return "🟡 Elevated"
        else:
            return "🔴 High – Health risk"

st.info(health_status(latest_value, pollutant))

# ---------------- HISTORICAL TREND ----------------
st.subheader("📈 Historical Trend")
st.line_chart(
    city_df.set_index("date")[[pollutant]],
    use_container_width=True
)

# ---------------- ML PREDICTION ----------------
st.subheader(f"🤖 AI Prediction – Next {predict_days} Days")

# Prepare ML data
city_df = city_df.reset_index(drop=True)
city_df["day_number"] = np.arange(len(city_df))

X = city_df[["day_number"]]
y = city_df[pollutant]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future
last_day = city_df["day_number"].iloc[-1]
future_X = np.array([[last_day + i] for i in range(1, predict_days + 1)])
predictions = model.predict(future_X)

future_dates = [
    city_df["date"].max() + timedelta(days=i)
    for i in range(1, predict_days + 1)
]

prediction_df = pd.DataFrame({
    "date": future_dates,
    "Predicted_" + pollutant: predictions
})

# ---------------- DISPLAY PREDICTION ----------------
st.line_chart(
    prediction_df.set_index("date"),
    use_container_width=True
)

st.dataframe(prediction_df, use_container_width=True)

# ---------------- DOWNLOAD BUTTON ----------------
csv = prediction_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Prediction as CSV",
    data=csv,
    file_name=f"{city}_{pollutant}_prediction.csv",
    mime="text/csv"
)

# ---------------- MODEL INFO ----------------
st.caption("🔍 Model: Linear Regression | Data-driven | Auto-trained")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "💡 **Tip:** Predictions improve with more historical data. "
    "Add daily readings for higher accuracy."
)
