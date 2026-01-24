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
