import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Air Pollution Dashboard",
    page_icon="🌍",
    layout="wide"
)

# -------------------- DARK MODE CSS --------------------
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
[data-testid="stMetric"] { background-color: #1e222a; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("air_quality.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Controls")

city = st.sidebar.selectbox("🏙️ Select City", df["city"].unique())
pollutant = st.sidebar.selectbox(
    "🧪 Select Pollutant",
    ["PM2.5", "PM10", "NO2", "SO2", "CO"]
)

predict_days = st.sidebar.slider("📅 Predict next days", 3, 15, 7)

# -------------------- FILTER DATA --------------------
df_city = df[df["city"] == city].sort_values("date")

# -------------------- TITLE --------------------
st.title("🌍 AI-Based Air Pollution Monitoring")
st.subheader(f"📍 {city} — {pollutant}")

# -------------------- CURRENT METRICS --------------------
latest_value = df_city[pollutant].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Latest Value", f"{latest_value}")
col2.metric("Max", f"{df_city[pollutant].max()}")
col3.metric("Min", f"{df_city[pollutant].min()}")

# -------------------- HEALTH ALERTS --------------------
st.subheader("🚨 Health Advisory")

if pollutant == "PM2.5":
    if latest_value <= 50:
        st.success("✅ Air quality is Good")
    elif latest_value <= 100:
        st.warning("⚠️ Moderate – Sensitive groups be careful")
    else:
        st.error("🚨 Unhealthy – Avoid outdoor activities")

elif pollutant == "PM10":
    if latest_value <= 100:
        st.success("✅ Acceptable air quality")
    elif latest_value <= 250:
        st.warning("⚠️ Moderate pollution")
    else:
        st.error("🚨 Very unhealthy air")

# -------------------- HISTORICAL TREND --------------------
st.subheader("📈 Historical Trend")
st.line_chart(df_city.set_index("date")[pollutant])

# -------------------- ML PREDICTION --------------------
st.subheader("🤖 AI Prediction (Improved Accuracy)")

df_city = df_city.copy()
df_city["day_number"] = np.arange(len(df_city))

X = df_city[["day_number"]]
y = df_city[pollutant]

model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("lr", LinearRegression())
])

model.fit(X, y)

future_days = np.arange(len(df_city), len(df_city) + predict_days).reshape(-1, 1)
predictions = model.predict(future_days)

future_dates = pd.date_range(
    start=df_city["date"].max() + pd.Timedelta(days=1),
    periods=predict_days
)

prediction_df = pd.DataFrame({
    "date": future_dates,
    "Predicted Value": predictions
})

st.line_chart(prediction_df.set_index("date"))

# -------------------- DOWNLOAD CSV --------------------
st.subheader("📥 Download Prediction")

csv = prediction_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Prediction CSV",
    csv,
    file_name=f"{city}_{pollutant}_prediction.csv",
    mime="text/csv"
)

# -------------------- CITY COMPARISON --------------------
st.subheader("📊 City Comparison")

city2 = st.selectbox(
    "Compare with another city",
    [c for c in df["city"].unique() if c != city]
)

df_city2 = df[df["city"] == city2].sort_values("date")

col1, col2 = st.columns(2)

col1.write(f"### {city}")
col1.line_chart(df_city.set_index("date")[pollutant])

col2.write(f"### {city2}")
col2.line_chart(df_city2.set_index("date")[pollutant])

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("🚀 AI Air Pollution Dashboard | Built with Streamlit & ML")
