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
# Pollution level chart
st.subheader("📈 Pollution Levels Over Time")

if "date" in filtered_df.columns and "pm2_5" in filtered_df.columns:
    filtered_df["date"] = pd.to_datetime(filtered_df["date"])
    filtered_df = filtered_df.sort_values("date")

    st.line_chart(
        filtered_df.set_index("date")["pm2_5"]
    )
else:
    st.warning("Required columns (date, pm2_5) not found in dataset.")

