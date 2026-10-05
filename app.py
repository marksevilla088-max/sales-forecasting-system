import streamlit as st
import pandas as pd
import plotly.express as px



st.set_page_config(
    page_title="Sales Forecasting System",
    layout="wide"
)


st.title("📈 Sales Forecasting System")

st.write(
    "Forecast comparison using ARIMA, SARIMA, Prophet, and Random Forest"
)




@st.cache_data
def load_forecast():

    df = pd.read_csv(
        "final_forecast_results.csv"
    )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"]
    )

    return df



@st.cache_data
def load_actual():

    actual = pd.read_csv(
        "daily_sales_clean.csv"
    )

    actual["date"] = pd.to_datetime(
        actual["date"]
    )

    return actual



df = load_forecast()

actual_df = load_actual()




st.sidebar.header("Filter")


selected_model = st.sidebar.selectbox(
    "Select Forecast Model",
    df["model_name"].unique()
)



model_data = df[
    df["model_name"] == selected_model
]






col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Forecast Records",
        len(df)
    )


with col2:

    st.metric(
        "Models Tested",
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


with col4:

    avg_mape = (
        df.groupby("model_name")["mape"]
        .mean()
        .min()
    )


    st.metric(
        "Lowest MAPE",
        f"{avg_mape:.2f}%"
    )





st.subheader("📊 Model Performance")


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




st.subheader(
    "Model Accuracy Comparison"
)


fig_mae = px.bar(
    performance,
    x="model_name",
    y="mae",
    title="Comparison of MAE"
)


st.plotly_chart(
    fig_mae,
    use_container_width=True
)



fig_mape = px.bar(
    performance,
    x="model_name",
    y="mape",
    title="Comparison of MAPE (%)"
)


st.plotly_chart(
    fig_mape,
    use_container_width=True
)






st.subheader(
    f"{selected_model}: Actual vs Forecast"
)


comparison = model_data.merge(
    actual_df,
    left_on="forecast_date",
    right_on="date",
    how="left"
)



fig_compare = px.line(
    comparison,
    x="forecast_date",
    y=[
        "daily_demand",
        "predicted_demand"
    ],
    title="Actual Demand vs Predicted Demand"
)


st.plotly_chart(
    fig_compare,
    use_container_width=True
)







st.subheader(
    "Forecast Error Analysis"
)


comparison["error"] = (
    comparison["daily_demand"]
    -
    comparison["predicted_demand"]
)


fig_error = px.line(
    comparison,
    x="forecast_date",
    y="error",
    title="Forecast Error"
)


st.plotly_chart(
    fig_error,
    use_container_width=True
)







st.subheader(
    f"{selected_model} Forecast Trend"
)


fig_forecast = px.line(
    model_data,
    x="forecast_date",
    y="predicted_demand",
    title=f"{selected_model} Predicted Demand"
)


st.plotly_chart(
    fig_forecast,
    use_container_width=True
)






st.subheader(
    "Download Forecast Result"
)


csv = model_data.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download Selected Forecast CSV",
    data=csv,
    file_name=f"{selected_model}_forecast.csv",
    mime="text/csv"
)







st.subheader(
    "Forecast Data"
)


st.dataframe(
    model_data,
    use_container_width=True
)
