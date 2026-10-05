import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Page Configuration
st.set_page_config(
    page_title="Sales Forecasting System",
    layout="wide"
)


# Title
st.title("📈 Sales Forecasting System")
st.write("Forecast comparison using ARIMA, SARIMA, Prophet, and Random Forest")


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("final_forecast_results.csv")
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    return df


df = load_data()


# Sidebar
st.sidebar.header("Filter")

selected_model = st.sidebar.selectbox(
    "Select Forecast Model",
    df["model_name"].unique()
)


# Filter data
model_data = df[
    df["model_name"] == selected_model
]


# Overview
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Forecast Records",
        len(df)
    )

with col2:
    st.metric(
        "Number of Models",
        df["model_name"].nunique()
    )

with col3:
    best_model = (
        df.groupby("model_name")["mape"]
        .mean()
        .idxmin()
    )

    st.metric(
        "Best Model",
        best_model
    )


# Model Performance
st.subheader("Model Performance")

performance = (
    df.groupby("model_name")
    [["mae","mape"]]
    .mean()
    .reset_index()
)

st.dataframe(
    performance,
    use_container_width=True
)


# Forecast Graph
st.subheader(
    f"{selected_model} Forecast"
)

fig = px.line(
    model_data,
    x="forecast_date",
    y="predicted_demand",
    title=f"{selected_model} Predicted Demand"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# Raw Data
st.subheader("Forecast Data")

st.dataframe(
    model_data,
    use_container_width=True
)
