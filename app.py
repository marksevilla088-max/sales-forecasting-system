import streamlit as st
import pandas as pd
import plotly.express as px


# ===============================
# PAGE CONFIGURATION
# ===============================

st.set_page_config(
    page_title="Sales Forecasting System",
    layout="wide"
)


# ===============================
# TITLE
# ===============================

st.title("📈 Sales Forecasting System")

st.write(
    "Forecast comparison using ARIMA, SARIMA, Prophet, and Random Forest"
)



# ===============================
# LOAD DATA
# ===============================

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



# ===============================
# SIDEBAR FILTER
# ===============================

st.sidebar.header("Forecast Model Selection")


selected_model = st.sidebar.selectbox(
    "Select Forecast Model",
    df["model_name"].unique()
)



model_data = df[
    df["model_name"] == selected_model
]



# ===============================
# SUMMARY CARDS
# ===============================


col1, col2, col3 = st.columns(3)



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
        "Recommended Model",
        best_model
    )



# ===============================
# MODEL PERFORMANCE
# ===============================

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



# ===============================
# MODEL RANKING
# ===============================


st.subheader("🏆 Model Ranking")


ranking = performance.sort_values(
    by="mape"
).reset_index(drop=True)


ranking.insert(
    0,
    "Rank",
    range(1,len(ranking)+1)
)



st.dataframe(
    ranking,
    use_container_width=True
)



# ===============================
# RECOMMENDATION
# ===============================


st.subheader("🤖 Forecast Recommendation")


best_row = ranking.iloc[0]


st.success(
    f"""
Recommended Model: {best_row['model_name']}

The model achieved the lowest forecasting error
with MAPE of {best_row['mape']:.2f}%.
"""
)



# ===============================
# ACCURACY COMPARISON
# ===============================


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



# ===============================
# SELECTED MODEL PERFORMANCE
# ===============================


st.subheader(
    f"{selected_model} Performance Summary"
)



selected_performance = performance[
    performance["model_name"] == selected_model
]



st.dataframe(
    selected_performance,
    use_container_width=True
)



# ===============================
# ACTUAL VS FORECAST
# ===============================


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



# ===============================
# MONTHLY SALES TREND
# ===============================


st.subheader(
    "📅 Monthly Actual Sales Trend"
)



monthly_sales = actual_df.copy()



monthly_sales["month"] = (
    monthly_sales["date"]
    .dt.to_period("M")
    .astype(str)
)



monthly_sales = (
    monthly_sales
    .groupby("month")
    ["daily_demand"]
    .sum()
    .reset_index()
)



fig_month = px.line(
    monthly_sales,
    x="month",
    y="daily_demand",
    title="Monthly Sales Demand"
)



st.plotly_chart(
    fig_month,
    use_container_width=True
)



# ===============================
# ERROR ANALYSIS
# ===============================


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




# ===============================
# FORECAST TREND
# ===============================


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



# ===============================
# DOWNLOAD SECTION
# ===============================


st.subheader(
    "Download Forecast Results"
)



# Selected model

selected_csv = model_data.to_csv(
    index=False
).encode("utf-8")



st.download_button(
    label="Download Selected Forecast CSV",
    data=selected_csv,
    file_name=f"{selected_model}_forecast.csv",
    mime="text/csv"
)



# All models

all_csv = df.to_csv(
    index=False
).encode("utf-8")



st.download_button(
    label="Download All Forecast Results",
    data=all_csv,
    file_name="all_forecast_results.csv",
    mime="text/csv"
)



# ===============================
# DATA TABLE
# ===============================


st.subheader(
    "Forecast Data"
)



st.dataframe(
    model_data,
    use_container_width=True
)
