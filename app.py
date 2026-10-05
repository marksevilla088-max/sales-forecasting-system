import streamlit as st
import pandas as pd
import plotly.express as px


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Sales Forecasting System",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("📈 Sales Forecasting System")

st.write(
    "Forecast comparison using ARIMA, SARIMA, Prophet, and Random Forest"
)



# ==========================================
# LOAD DATA
# ==========================================

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



# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header(
    "Forecast Model Selection"
)


selected_model = st.sidebar.selectbox(
    "Select Forecast Model",
    df["model_name"].unique()
)


st.sidebar.divider()


st.sidebar.subheader(
    "System Information"
)


st.sidebar.info(
"""
Dataset:
Sta. Cruz Laguna Sales Dataset

Forecast Models:
• ARIMA
• SARIMA
• Prophet
• Random Forest

Evaluation Metrics:
• MAE
• MAPE

Platform:
Python + Streamlit
"""
)



model_data = df[
    df["model_name"] == selected_model
]



# ==========================================
# DATASET OVERVIEW
# ==========================================

st.subheader(
    "📁 Dataset Overview"
)


dataset_col1, dataset_col2, dataset_col3 = st.columns(3)


with dataset_col1:

    st.metric(
        "Historical Sales Records",
        len(actual_df)
    )


with dataset_col2:

    st.metric(
        "Forecast Records",
        len(df)
    )


with dataset_col3:

    st.metric(
        "Models Evaluated",
        df["model_name"].nunique()
    )



# ==========================================
# SUMMARY CARDS
# ==========================================

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
        "Recommended Model",
        best_model
    )



with col4:

    lowest_mape = (
        df.groupby("model_name")["mape"]
        .mean()
        .min()
    )


    st.metric(
        "Lowest MAPE",
        f"{lowest_mape:.2f}%"
    )



# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.subheader(
    "📊 Model Performance"
)



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



# ==========================================
# MODEL RANKING
# ==========================================


st.subheader(
    "🏆 Model Ranking"
)



ranking = performance.sort_values(
    by="mape"
).reset_index(drop=True)



ranking.insert(
    0,
    "Rank",
    range(1,len(ranking)+1)
)



def performance_level(mape):

    if mape < 10:
        return "Excellent"

    elif mape < 20:
        return "Good"

    elif mape < 50:
        return "Acceptable"

    else:
        return "Poor"



ranking["Performance"] = (
    ranking["mape"]
    .apply(performance_level)
)



st.dataframe(
    ranking,
    use_container_width=True
)



# ==========================================
# RECOMMENDATION
# ==========================================


st.subheader(
    "🤖 Forecast Recommendation"
)



best_row = ranking.iloc[0]



st.success(
f"""
Recommended Model: {best_row['model_name']}

Performance Level:
{best_row['Performance']}

The model achieved the lowest forecasting error
with MAPE of {best_row['mape']:.2f}%.
"""
)



# ==========================================
# END PART 1
# ==========================================

# ==========================================
# STATISTICAL VS MACHINE LEARNING COMPARISON
# ==========================================


st.subheader(
    "🧠 Forecasting Approach Comparison"
)


def classify_model(model):

    if model in ["ARIMA", "SARIMA"]:
        return "Statistical Model"

    else:
        return "Machine Learning Model"



comparison_type = performance.copy()


comparison_type["Approach"] = (
    comparison_type["model_name"]
    .apply(classify_model)
)



approach_summary = (
    comparison_type
    .groupby("Approach")
    [["mae","mape"]]
    .mean()
    .reset_index()
)



st.dataframe(
    approach_summary,
    use_container_width=True
)



# ==========================================
# ACTUAL SALES SUMMARY
# ==========================================


st.subheader(
    "📦 Historical Sales Summary"
)



sales_col1, sales_col2, sales_col3, sales_col4 = st.columns(4)



with sales_col1:

    st.metric(
        "Total Demand",
        f"{actual_df['daily_demand'].sum():,.0f}"
    )



with sales_col2:

    st.metric(
        "Average Daily Demand",
        f"{actual_df['daily_demand'].mean():,.2f}"
    )



with sales_col3:

    highest_date = (
        actual_df
        .loc[
            actual_df["daily_demand"].idxmax(),
            "date"
        ]
    )


    st.metric(
        "Highest Demand Date",
        str(highest_date.date())
    )



with sales_col4:

    st.metric(
        "Maximum Demand",
        f"{actual_df['daily_demand'].max():,.0f}"
    )




# ==========================================
# ACTUAL VS FORECAST
# ==========================================


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




# ==========================================
# MONTHLY TREND
# ==========================================


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




# ==========================================
# ERROR ANALYSIS
# ==========================================


st.subheader(
    "📉 Forecast Error Analysis"
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




# ==========================================
# FORECAST TREND
# ==========================================


st.subheader(
    f"📈 {selected_model} Forecast Trend"
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




# ==========================================
# FORECAST SUMMARY
# ==========================================


st.subheader(
    "📌 Forecast Summary"
)



summary1, summary2, summary3 = st.columns(3)



with summary1:

    st.metric(
        "Average Forecast",
        f"{model_data['predicted_demand'].mean():,.2f}"
    )


with summary2:

    st.metric(
        "Highest Forecast",
        f"{model_data['predicted_demand'].max():,.2f}"
    )


with summary3:

    st.metric(
        "Lowest Forecast",
        f"{model_data['predicted_demand'].min():,.2f}"
    )




# ==========================================
# DOWNLOAD SECTION
# ==========================================


st.subheader(
    "⬇️ Download Forecast Results"
)



selected_csv = model_data.to_csv(
    index=False
).encode("utf-8")



st.download_button(
    label="Download Selected Forecast CSV",
    data=selected_csv,
    file_name=f"{selected_model}_forecast.csv",
    mime="text/csv"
)



all_csv = df.to_csv(
    index=False
).encode("utf-8")



st.download_button(
    label="Download All Forecast Results",
    data=all_csv,
    file_name="all_forecast_results.csv",
    mime="text/csv"
)



ranking_csv = ranking.to_csv(
    index=False
).encode("utf-8")



st.download_button(
    label="Download Model Ranking Report",
    data=ranking_csv,
    file_name="model_ranking_report.csv",
    mime="text/csv"
)




# ==========================================
# FINAL DATA TABLE
# ==========================================


st.subheader(
    "📋 Forecast Data"
)



st.dataframe(
    model_data,
    use_container_width=True
)
