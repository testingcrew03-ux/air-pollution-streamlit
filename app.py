import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="India Air Quality Predictor",
    page_icon="🌍",
    layout="wide"
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("air_quality.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Ensure required columns
    required_cols = ["City", "Date", "PM2.5", "PM10", "NO2", "SO2", "CO"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column in CSV: {col}")
            st.stop()

    # Clean data
    df["City"] = df["City"].astype(str).str.strip().str.title()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.dropna(subset=["Date"])
    return df


df = load_data()

# -------------------- SIDEBAR --------------------
st.sidebar.title("🎛 Controls")

cities = sorted(df["City"].unique())
selected_city = st.sidebar.selectbox("Select City", cities)

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO"]
selected_pollutant = st.sidebar.selectbox("Select Pollutant", pollutants)

predict_days = st.sidebar.slider("Predict Next Days", 1, 14, 7)

# -------------------- FILTER DATA --------------------
city_df = df[df["City"] == selected_city].sort_values("Date")

if city_df.empty:
    st.error("No data available for this city.")
    st.stop()

latest_value = city_df[selected_pollutant].iloc[-1]
last_updated = city_df["Date"].max().date()

# -------------------- HEADER --------------------
st.markdown(
    f"""
    <h2>📍 Current Air Quality in {selected_city}</h2>
    <p style="color:gray">ML-powered • Health-aware • India-wide</p>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

col1.metric("Pollutant", selected_pollutant)
col2.metric("Latest Value", f"{latest_value:.1f}")
col3.metric("Last Updated", str(last_updated))

# -------------------- HEALTH ALERTS --------------------
def health_alert(value):
    if value <= 50:
        return "🟢 Good – Safe for everyone"
    elif value <= 100:
        return "🟡 Moderate – Sensitive people take care"
    elif value <= 200:
        return "🟠 Poor – Limit outdoor activity"
    elif value <= 300:
        return "🔴 Very Poor – Health risk"
    else:
        return "⚫ Severe – Emergency conditions"

st.warning(f"🚨 Health Advisory: **{health_alert(latest_value)}**")

# -------------------- HISTORICAL CHART --------------------
st.subheader("📊 Historical Trend")

st.line_chart(
    city_df.set_index("Date")[selected_pollutant]
)

# -------------------- ML PREDICTION --------------------
st.subheader("🤖 ML Prediction")

# Prepare training data
train_df = city_df.copy()
train_df["DayIndex"] = np.arange(len(train_df))

X = train_df[["DayIndex"]]
y = train_df[selected_pollutant]

model = LinearRegression()
model.fit(X, y)

# Future dates
future_days = np.arange(len(train_df), len(train_df) + predict_days)
future_dates = [
    train_df["Date"].max() + timedelta(days=i + 1)
    for i in range(predict_days)
]

predictions = model.predict(future_days.reshape(-1, 1))

prediction_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Value": predictions
})

# -------------------- PREDICTION CHART --------------------
st.line_chart(
    prediction_df.set_index("Date")["Predicted_Value"]
)

# -------------------- DOWNLOAD CSV --------------------
st.subheader("📥 Download Predictions")

csv = prediction_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Prediction as CSV",
    data=csv,
    file_name=f"{selected_city}_{selected_pollutant}_prediction.csv",
    mime="text/csv"
)

# -------------------- DATA TABLE --------------------
with st.expander("📄 View Prediction Table"):
    st.dataframe(prediction_df)

# -------------------- FOOTER --------------------
st.markdown(
    """
    <hr>
    <center>
    🌱 Built with Streamlit • Machine Learning • India Air Quality Data
    </center>
    """,
    unsafe_allow_html=True
)
