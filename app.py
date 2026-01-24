import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


st.set_page_config(page_title="Air Pollution Monitoring")

st.title("🌍 Air Pollution Monitoring System")
st.write("This app displays air quality data.")
st.markdown("---")
st.markdown("### 🌍 Select Parameters")


# Load data
df = pd.read_csv("air_quality.csv")

# Show data
st.subheader("📊 Air Quality Data")
st.dataframe(df)
# City filter
st.subheader("🏙️ Filter by City")

city = st.selectbox(
    "Select a city",
    df["City"].unique()
)
compare = st.checkbox("🔁 Compare with another city")

if compare:
    city2 = st.selectbox(
        "Select Second City",
        df["city"].unique(),
        index=1
    )

pollutant = st.selectbox(
    "Select Pollutant",
    ["PM2.5", "PM10"]
)


# Filter data for selected city
filtered_df = df[df["City"] == city]
# Get latest pollution value
latest_value = filtered_df[pollutant].iloc[-1]



# ================= CITY COMPARISON SECTION =================
if compare:
    filtered_df2 = df[df["city"] == city2]

    st.markdown("### 📊 City Comparison")

    col1, col2 = st.columns(2)

    col1.subheader(city)
    col1.line_chart(filtered_df.set_index("date")[pollutant])

    col2.subheader(city2)
    col2.line_chart(filtered_df2.set_index("date")[pollutant])

# ================= AIR QUALITY STATUS SECTION =================
if pollutant == "PM2.5":
    if latest_value <= 50:
        st.success("🟢 Good Air Quality")
    elif latest_value <= 100:
        st.warning("🟡 Moderate Air Quality")
    else:
        st.error("🔴 Poor Air Quality")

elif pollutant == "PM10":
    if latest_value <= 100:
        st.success("🟢 Good Air Quality")
    elif latest_value <= 250:
        st.warning("🟡 Moderate Air Quality")
    else:
        st.error("🔴 Poor Air Quality")
st.markdown("## 🤖 Air Quality Prediction (Next 7 Days)")

# Use selected city data
df_city = filtered_df.copy()
df_city = df_city.reset_index(drop=True)


# Convert date to numbers for ML
df_city["day_number"] = np.arange(len(df_city))

X = df_city[["day_number"]]
y = df_city[pollutant]

# Train ML model
model = LinearRegression()
model.fit(X, y)

# Predict future values
future_days = 7
last_day = df_city["day_number"].iloc[-1]

future_X = np.array(
    [[last_day + i] for i in range(1, future_days + 1)]
)

predictions = model.predict(future_X)

# Create future dates
future_dates = pd.date_range(
    start=df_city["date"].max() + pd.Timedelta(days=1),
    periods=future_days
)

prediction_df = pd.DataFrame({
    "date": future_dates,
    "Predicted Value": predictions
})

st.line_chart(prediction_df.set_index("date"))




if latest_value <= 50:
    st.info("✅ Safe to go outdoors. Enjoy your day!")
elif latest_value <= 100:
    st.info("⚠️ Sensitive people should reduce prolonged outdoor exertion.")
else:
    st.warning("🚫 Avoid outdoor activities. Wear a mask if going outside.")
st.markdown("### 📈 Pollution Trend")

if "date" in filtered_df.columns:
    filtered_df["date"] = pd.to_datetime(filtered_df["date"])
    filtered_df = filtered_df.sort_values("date")

    st.line_chart(
        filtered_df.set_index("date")[pollutant]
    )
else:
    st.line_chart(filtered_df[pollutant])





st.write(f"Showing data for: **{city}**")
st.dataframe(filtered_df)
# Pollution level chart
# Pollution level chart
st.subheader("📈 Pollution Levels")

chart_data = filtered_df[["PM2.5", "PM10"]]
st.line_chart(chart_data)
st.subheader("🚦 Air Quality Status")

pm25 = filtered_df["PM2.5"].values[0]

if pm25 <= 50:
    st.success("🟢 Good Air Quality")
elif pm25 <= 100:
    st.warning("🟡 Moderate Air Quality")
else:
    st.error("🔴 Poor Air Quality")
st.markdown("---")
st.caption("🌍 Air Pollution Monitoring System | Built using Streamlit")


