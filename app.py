import streamlit as st
import pandas as pd

st.set_page_config(page_title="Air Pollution Monitoring")

st.title("🌍 Air Pollution Monitoring System")
st.write("This app displays air quality data.")

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

filtered_df = df[df["City"] == city]


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


