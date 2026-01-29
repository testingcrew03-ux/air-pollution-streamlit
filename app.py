import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

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

days = st.sidebar.slider("Predict Next Days", 3, 14, 7)

city_df = df[df["City"] == city].copy()

# ---------------- HEADER ----------------
st.title(f"📍 Air Quality AI Forecast — {city}")

latest = city_df.iloc[-1]

c1, c2, c3 = st.columns(3)
c1.metric("PM2.5", round(latest["PM2.5"], 1))
c2.metric("PM10", round(latest["PM10"], 1))
c3.metric("NO2", round(latest["NO2"], 1))

st.caption(f"Last Updated: {latest['Date'].date()}")

# ---------------- HISTORICAL ----------------
st.subheader("📈 Historical Trend")

fig = px.line(
    city_df,
    x="Date",
    y=["PM2.5", "PM10", "NO2"],
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- FEATURE ENGINEERING ----------------
st.subheader("🤖 AI Forecast")

df_ml = city_df.copy()
df_ml["PM2.5_lag1"] = df_ml["PM2.5"].shift(1)
df_ml["PM2.5_lag2"] = df_ml["PM2.5"].shift(2)
df_ml["PM2.5_lag3"] = df_ml["PM2.5"].shift(3)

df_ml.dropna(inplace=True)

X = df_ml[[
    "PM10", "NO2",
    "PM2.5_lag1",
    "PM2.5_lag2",
    "PM2.5_lag3"
]]

y = df_ml["PM2.5"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_scaled, y)

# ---------------- PREDICTION ----------------
last_row = df_ml.iloc[-1].copy()
predictions = []

for _ in range(days):
    features = np.array([[
        last_row["PM10"],
        last_row["NO2"],
        last_row["PM2.5"],
        last_row["PM2.5_lag1"],
        last_row["PM2.5_lag2"]
    ]])

    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    predictions.append(pred)

    last_row["PM2.5_lag2"] = last_row["PM2.5_lag1"]
    last_row["PM2.5_lag1"] = last_row["PM2.5"]
    last_row["PM2.5"] = pred

future_dates = pd.date_range(
    city_df["Date"].max() + pd.Timedelta(days=1),
    periods=days
)

pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted PM2.5": predictions
})

st.dataframe(pred_df, use_container_width=True)

st.download_button(
    "📥 Download Prediction CSV",
    pred_df.to_csv(index=False),
    "pm25_predictions.csv",
    "text/csv"
)

# ---------------- FORECAST CHART ----------------
fig2 = px.line(
    pred_df,
    x="Date",
    y="Predicted PM2.5",
    markers=True,
    title="AI PM2.5 Forecast"
)

st.plotly_chart(fig2, use_container_width=True)
