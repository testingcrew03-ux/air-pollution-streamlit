import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Air Pollution Monitoring", layout="wide")

st.title("🌍 Air Pollution Monitoring System")
st.write("Visualizing and predicting air pollution levels using data & ML")

# ------------------ LOAD DATA ------------------
df = pd.read_csv("air_quality.csv")

# Clean column names (VERY IMPORTANT)
df.columns = df.columns.str.strip().str.lower()

# ------------------ SIDEBAR ------------------
st.sidebar.header("🔧 Controls")

city = st.sidebar.selectbox("Select City", df["city"].unique())

pollutant = st.sidebar.selectbox(
    "Select Pollutant",
    [col for col in df.columns if col not in ["city"]]
)

compare = st.sidebar.checkbox("Compare with another city")

# ------------------ FILTER DATA ------------------
filtered_df = df[df["city"] == city].reset_index(drop=True)

# ------------------ LATEST VALUE ------------------
latest_value = filtered_df[pollutant].iloc[-1]

st.subheader(f"📍 Latest {pollutant.upper()} level in {city}: {latest_value}")

# Air Quality Status
if latest_value <= 50:
    st.success("🟢 Air Quality is GOOD")
elif latest_value <= 100:
    st.warning("🟡 Air Quality is MODERATE")
else:
    st.error("🔴 Air Quality is POOR")

# ------------------ LINE CHART ------------------
st.subheader("📈 Pollution Trend")
st.line_chart(filtered_df[pollutant])

# ------------------ CITY COMPARISON ------------------
if compare:
    city2 = st.sidebar.selectbox(
        "Select Second City",
        df["city"].unique()
    )

    filtered_df2 = df[df["city"] == city2].reset_index(drop=True)

    st.subheader("📊 City Comparison")

    col1, col2 = st.columns(2)

    col1.subheader(city)
    col1.line_chart(filtered_df[pollutant])

    col2.subheader(city2)
    col2.line_chart(filtered_df2[pollutant])

# ------------------ ML PREDICTION ------------------
st.markdown("## 🔮 Air Quality Prediction (Next 7 Days)")

df_city = filtered_df.copy()
df_city["day_number"] = np.arange(len(df_city))

X = df_city[["day_number"]]
y = df_city[pollutant]

model = LinearRegression()
model.fit(X, y)

future_days = 7
last_day = df_city["day_number"].iloc[-1]

future_X = np.array([[last_day + i] for i in range(1, future_days + 1)])
predictions = model.predict(future_X)

future_df = pd.DataFrame({
    "Day": [f"Day {i}" for i in range(1, future_days + 1)],
    "Predicted Value": predictions
})

st.line_chart(predictions)
st.dataframe(future_df)

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("Developed using Streamlit & Machine Learning")
