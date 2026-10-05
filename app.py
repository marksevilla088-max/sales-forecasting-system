import streamlit as st
import pandas as pd
import plotly.express as px


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide"
)


# =====================================================
# CUSTOM CSS - PROFESSIONAL THESIS UI
# =====================================================

st.markdown(
    """
    <style>

    /* Main title */
    h1 {
        font-size: 42px;
        font-weight: 700;
    }


    /* Section headers */
    h2 {
        font-size: 28px;
        font-weight: 600;
    }


    h3 {
        font-size: 22px;
        font-weight: 600;
    }


    /* Metric cards */
    div[data-testid="metric-container"] {

        background-color: #111827;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 12px;

    }


    /* Dataframe */
    .stDataFrame {

        border-radius: 10px;

    }


    /* Sidebar */

    section[data-testid="stSidebar"] {

        background-color:#0f172a;

    }


    /* Buttons */

    .stDownloadButton button {

        border-radius:8px;
        font-weight:600;

    }


    </style>
    """,
    unsafe_allow_html=True
)



# =====================================================
# TITLE
# =====================================================

st.title(
    "Sales Forecasting System"
)


st.write(
    "A comparative forecasting system using ARIMA, SARIMA, Prophet, and Random Forest models."
)


st.divider()



# =====================================================
# LOAD DATA
# =====================================================


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



# =====================================================
# SIDEBAR
# =====================================================


st.sidebar.title(
    "Forecast Model Selection"
)



selected_model = st.sidebar.selectbox(
    "Choose Forecasting Model",
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
ARIMA
SARIMA
Prophet
Random Forest


Evaluation Metrics:
MAE
MAPE


Platform:
Python + Streamlit
"""
)



model_data = df[
    df["model_name"] == selected_model
]



# =====================================================
# DATASET OVERVIEW
# =====================================================


st.header(
    "Dataset Overview"
)



historical_records = len(actual_df)

forecast_records = len(df)

number_models = df["model_name"].nunique()



col1,col2,col3 = st.columns(3)



with col1:

    st.metric(
        "Historical Sales Records",
        historical_records
    )


with col2:

    st.metric(
        "Forecast Records",
        forecast_records
    )


with col3:

    st.metric(
        "Models Evaluated",
        number_models
    )



st.divider()



# =====================================================
# MODEL PERFORMANCE
# =====================================================


st.header(
    "Model Performance Evaluation"
)



performance = (

    df.groupby("model_name")
    [["mae","mape"]]
    .mean()
    .reset_index()

)



performance["mape"] = performance["mape"].round(4)

performance["mae"] = performance["mae"].round(4)



st.dataframe(
    performance,
    use_container_width=True,
    hide_index=True
)


# =====================================================
# MODEL RANKING
# =====================================================


st.header(
    "Model Ranking"
)


ranking = (

    performance
    .sort_values(
        by="mape"
    )
    .reset_index(drop=True)

)



ranking.insert(
    0,
    "Rank",
    range(1, len(ranking)+1)
)



def classify_performance(mape):

    if mape < 10:
        return "Excellent"

    elif mape < 20:
        return "Good"

    elif mape < 50:
        return "Acceptable"

    else:
        return "Poor"



ranking["Performance Level"] = (
    ranking["mape"]
    .apply(classify_performance)
)



st.dataframe(
    ranking,
    use_container_width=True,
    hide_index=True
)



st.divider()



# =====================================================
# RECOMMENDED MODEL
# =====================================================


st.header(
    "Forecast Recommendation"
)



best_model_row = ranking.iloc[0]



recommendation_col1, recommendation_col2 = st.columns(2)



with recommendation_col1:


    st.metric(
        "Recommended Forecast Model",
        best_model_row["model_name"]
    )


with recommendation_col2:


    st.metric(
        "Lowest MAPE",
        f"{best_model_row['mape']:.2f}%"
    )



st.success(

f"""
The recommended forecasting model is **{best_model_row['model_name']}**.

It achieved the lowest forecasting error among the evaluated models,
with a Mean Absolute Percentage Error (MAPE) of
**{best_model_row['mape']:.2f}%**.
"""

)



st.divider()



# =====================================================
# FORECASTING APPROACH COMPARISON
# =====================================================


st.header(
    "Forecasting Approach Comparison"
)



approach_df = performance.copy()



approach_df["approach"] = approach_df[
    "model_name"
].apply(

lambda x:

"Machine Learning Model"
if x == "Random Forest"

else "Statistical Model"

)



approach_comparison = (

approach_df
.groupby("approach")
[["mae","mape"]]
.mean()
.reset_index()

)



st.dataframe(
    approach_comparison,
    use_container_width=True,
    hide_index=True
)



fig_approach = px.bar(

    approach_comparison,

    x="approach",

    y="mape",

    title="Average MAPE Comparison by Forecasting Approach"

)



st.plotly_chart(
    fig_approach,
    use_container_width=True
)




fig_approach_mae = px.bar(

    approach_comparison,

    x="approach",

    y="mae",

    title="Average MAE Comparison by Forecasting Approach"

)



st.plotly_chart(
    fig_approach_mae,
    use_container_width=True
)




st.divider()



# =====================================================
# HISTORICAL SALES SUMMARY
# =====================================================


st.header(
    "Historical Sales Summary"
)



total_sales = (

actual_df["daily_demand"]
.sum()

)



average_daily = (

actual_df["daily_demand"]
.mean()

)



max_sales = (

actual_df["daily_demand"]
.max()

)



highest_date = (

actual_df.loc[
    actual_df["daily_demand"].idxmax(),
    "date"
]

)



summary1, summary2, summary3, summary4 = st.columns(4)



with summary1:

    st.metric(
        "Total Demand",
        f"{total_sales:,.0f}"
    )



with summary2:

    st.metric(
        "Average Daily Demand",
        f"{average_daily:,.2f}"
    )



with summary3:

    st.metric(
        "Highest Demand",
        f"{max_sales:,.0f}"
    )



with summary4:

    st.metric(
        "Highest Demand Date",
        highest_date.strftime("%Y-%m-%d")
    )



st.divider()



# =====================================================
# ACTUAL VS FORECAST
# =====================================================


st.header(
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

    title="Actual Demand Compared with Forecasted Demand"

)



st.plotly_chart(

    fig_compare,

    use_container_width=True

)


