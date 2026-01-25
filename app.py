import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Air Pollution Monitoring System",
    page_icon="🌍",
    layout="wide"
)

# ---------------- CUSTOM DARK UI ----------------
st.markdown("""
<style>
body { background-color: #0E1117; color: white; }
.metric-container { background-color: #1c1f26; }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("air_quality.csv")

df = load_data()

# Fix column names
df.columns = df.columns.str.strip()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌍 Air Pollution Dashboard")
city = st.sidebar.selectbox("🏙️ Select City", df["City"].unique())
pollutant = st.sidebar.selectbox(
    "🧪 Select Pollutant",
    ["PM2.5", "PM10", "NO2", "SO2", "CO"]
)

# ---------------- FILTER DATA ----------------
city_df = df[df["City"] == city]
current_value = city_df[pollutant].values[0]

# ---------------- METRICS ----------------
st.markdown("## 📊 Current Pollution Status")

c1, c2, c3 = st.columns(3)
c1.metric("🏙️ City", city)
c2.metric("🧪 Pollutant", pollutant)
c3.metric("📍 Current Value", current_value)

# ---------------- HEALTH STATUS ----------------
st.markdown("### 🚨 Health Advisory")

if current_value <= 50:
    st.success("🟢 Good – Safe for everyone")
elif current_value <= 100:
    st.warning("🟡 Moderate – Sensitive groups take care")
else:
    st.error("🔴 Poor – Health risk for all")

# ---------------- BAR CHART ----------------
st.markdown("### 📈 Pollution Comparison (All Cities)")
st.bar_chart(df.set_index("City")[pollutant])

# ---------------- MACHINE LEARNING ----------------
st.markdown("## 🔮 ML-Based Pollution Prediction")

# Create synthetic time steps for ML
X = np.arange(len(df)).reshape(-1, 1)
y = df[pollutant].values

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
accuracy = r2_score(y, y_pred)

st.metric("📊 Model Accuracy (R²)", round(accuracy, 2))

# Predict next value
future_step = np.array([[len(df) + 1]])
future_prediction = model.predict(future_step)[0]

st.success(f"📌 Predicted Future {pollutant}: **{round(future_prediction, 2)}**")

# ---------------- HEALTH ALERT (PREDICTED) ----------------
st.markdown("### 🚨 Predicted Health Alert")

if future_prediction > 150:
    st.error("🚨 Severe pollution expected!")
elif future_prediction > 100:
    st.warning("⚠️ Pollution may become unhealthy.")
else:
    st.success("✅ Pollution levels expected to remain safe.")

# ---------------- DOWNLOAD CSV ----------------
st.markdown("### 📥 Download Prediction")

download_df = pd.DataFrame({
    "City": [city],
    "Pollutant": [pollutant],
    "Current Value": [current_value],
    "Predicted Value": [future_prediction]
})

csv = download_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Prediction CSV",
    csv,
    "prediction.csv",
    "text/csv"
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "👨‍💻 **AI-Based Air Pollution Monitoring System**  \n"
    "Built using Streamlit & Machine Learning"
)

