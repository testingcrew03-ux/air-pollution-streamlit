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
    df["city"].unique()
)

filtered_df = df[df["city"] == city]

st.write(f"Showing data for: **{city}**")
st.dataframe(filtered_df)
