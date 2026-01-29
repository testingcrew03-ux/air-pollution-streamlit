import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="India AQI Intelligence Platform",
    layout="wide",
    page_icon="🌏"
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("air_quality.csv")
    df.columns = df.columns.str.lower().str.strip()
    df["city"] = df["city"].str.title()
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# ================= CPCB AQI =================
def calculate_aqi(pm25):
    if pm25 <= 30: return pm25 * (50/30)
    elif pm25 <= 60: return 50 + (pm25-30)*(50/30)
    elif pm25 <= 90: return 100 + (pm25-60)*(100/30)
    elif pm25 <= 120: return 200 + (pm25-90)*(100/30)
    elif pm25 <= 250: return 300 + (pm25-120)*(100/130)
    else: return 400 + (pm25-250)*(100/130)

df["aqi"] = df["pm2.5"].apply(calculate_aqi)

# ================= SIDEBAR =================
st.sidebar.title("🎛 Controls")

cities = sorted(df["city"].unique())
city = st.sidebar.selectbox("Select City", cities)

pollutant_map = {
    "PM2.5": "pm2.5",
    "PM10": "pm10",
    "NO2": "no2",
    "SO2": "so2",
    "CO": "co"
}

pollutant_label = st.sidebar.selectbox("Select Pollutant", pollutant_map.keys())
pollutant = pollutant_map[pollutant_label]

days = st.sidebar.slider("Predict Next Days", 3, 14, 7)

# ================= FILTER =================
city_df = df[df["city"] == city].sort_values("date")

latest_value = city_df[pollutant].iloc[-1]
latest_aqi = calculate_aqi(latest_value)

# ================= HEADER =================
st.title(f"📍 Air Quality in {city}")

col1, col2, col3 = st.columns(3)
col1.metric("Pollutant", pollutant_label)
col2.metric("Latest Value", f"{latest_value:.1f}")
col3.metric("AQI", f"{latest_aqi:.0f}")

# ================= HEALTH ALERT =================
def health_alert(aqi):
    if aqi <= 50: return "🟢 Good"
    elif aqi <= 100: return "🟡 Moderate"
    elif aqi <= 200: return "🟠 Poor"
    elif aqi <= 300: return "🔴 Very Poor"
    else: return "⚫ Severe"

st.warning(f"🚨 Health Status: **{health_alert(latest_aqi)}**")

# ================= HISTORICAL =================
st.subheader("📈 Historical Trend")
st.line_chart(city_df.set_index("date")[pollutant])

# ================= ML MODEL =================
st.subheader("🤖 AI Forecast")

city_df["day_index"] = np.arange(len(city_df))
X = city_df[["day_index"]]
y = city_df[pollutant]

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)
accuracy = r2_score(y, y_pred)

future_days = np.arange(len(city_df), len(city_df)+days)
future_X = poly.transform(future_days.reshape(-1,1))
forecast = model.predict(future_X)

future_dates = [city_df["date"].max() + timedelta(days=i+1) for i in range(days)]

pred_df = pd.DataFrame({
    "Date": future_dates,
    f"Predicted {pollutant_label}": forecast
})

st.metric("Model Accuracy (R²)", f"{accuracy:.3f}")
st.line_chart(pred_df.set_index("Date"))

# ================= DOWNLOAD =================
st.download_button(
    "📥 Download Prediction CSV",
    pred_df.to_csv(index=False),
    file_name=f"{city}_{pollutant_label}_forecast.csv"
)

# ================= CITY COMPARISON =================
st.subheader("🏙 City AQI Comparison")

latest_df = df.sort_values("date").groupby("city").tail(1)
fig = px.bar(
    latest_df,
    x="city",
    y="aqi",
    color="aqi",
    title="Latest AQI Across Cities"
)
st.plotly_chart(fig, use_container_width=True)

# ================= INDIA HEATMAP =================
st.subheader("🗺 India AQI Heatmap")

city_coords = {
    "Delhi": (28.61,77.21),
    "Mumbai": (19.07,72.87),
    "Kolkata": (22.57,88.36),
    "Chennai": (13.08,80.27),
    "Bengaluru": (12.97,77.59),
    "Ahmedabad": (23.03,72.58),
    "Noida": (28.54,77.39),
    "Ghaziabad": (28.67,77.45),
    "Gurugram": (28.46,77.03),
    "Bhiwadi": (28.21,76.86),
    "Hapur": (28.73,77.78),
    "Muzaffarnagar": (29.47,77.70),
    "Hajipur": (25.69,85.21),
    "Byrnihat": (26.05,91.87)
}

map_df = latest_df[latest_df["city"].isin(city_coords)].copy()
map_df["lat"] = map_df["city"].map(lambda x: city_coords[x][0])
map_df["lon"] = map_df["city"].map(lambda x: city_coords[x][1])

fig_map = px.scatter_mapbox(
    map_df,
    lat="lat",
    lon="lon",
    size="aqi",
    color="aqi",
    zoom=4,
    mapbox_style="carto-darkmatter"
)
st.plotly_chart(fig_map, use_container_width=True)

# ================= FOOTER =================
st.markdown(
    "<hr><center>🇮🇳 India AQI Intelligence Platform • ML Powered</center>",
    unsafe_allow_html=True
)
