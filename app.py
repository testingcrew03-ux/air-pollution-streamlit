import streamlit as st
import pandas as pd

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
pollutant = st.selectbox(
    "Select Pollutant",
    ["PM2.5", "PM10"]
)


filtered_df = df[df["City"] == city]
latest_value = filtered_df[pollutant].values[0]
st.markdown("---")
st.markdown("### 📊 Air Quality Overview")

col1, col2, col3 = st.columns(3)

col1.metric("City", city)
col2.metric("Pollutant", pollutant)
col3.metric("Value", latest_value)
st.markdown("### 🚦 Air Quality Status")

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
        st.markdown("### 🩺 Health Advisory")

if latest_value <= 50:
    st.info("✅ Safe to go outdoors. Enjoy your day!")
elif latest_value <= 100:
    st.info("⚠️ Sensitive people should reduce prolonged outdoor exertion.")
else:
    st.warning("🚫 Avoid outdoor activities. Wear a mask if going outside.")





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


