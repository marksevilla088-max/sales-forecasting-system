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

# PROFESSIONAL UI STYLE

# ==========================================



st.markdown(
    """
    <div style="
        padding-top:60px;
        padding-bottom:25px;
    ">

        <h1 style="
            font-size:52px;
            font-weight:750;
            margin-bottom:15px;
        ">
            Sales Forecasting System
        </h1>

        <p style="
            font-size:19px;
            color:#b8c1d1;
            line-height:1.6;
        ">
            A comparative forecasting system using ARIMA, SARIMA,
            Prophet, and Random Forest models.
        </p>

    </div>

    <hr style="
        border:0;
        border-top:1px solid #30363d;
        margin-top:30px;
        margin-bottom:45px;
    ">
    """,
    unsafe_allow_html=True
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

    "Choose Forecasting Model",

    df["model_name"].unique()

)







st.sidebar.divider()







st.sidebar.subheader(

    "System Information"

)







st.sidebar.info(

"""

Dataset



Sta. Cruz Laguna Sales Dataset





Forecasting Models



ARIMA

SARIMA

Prophet

Random Forest





Evaluation Metrics



MAE

MAPE





Development Platform



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

    "Dataset Overview"

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







st.divider()







# ==========================================

# SUMMARY CARDS

# ==========================================





st.subheader(

    "Forecast Summary"

)







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







st.divider()







# ==========================================

# MODEL PERFORMANCE

# ==========================================





st.subheader(

    "Forecast Model Evaluation"

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

    "Model Performance Ranking"

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







ranking["Performance Level"] = (

    ranking["mape"]

    .apply(performance_level)

)







st.dataframe(

    ranking,

    use_container_width=True

)



# ==========================================

# FORECAST RECOMMENDATION

# ==========================================





st.divider()





st.subheader(

    "Forecast Recommendation"

)







best_row = ranking.iloc[0]







rec_col1, rec_col2 = st.columns(2)







with rec_col1:



    st.metric(

        "Recommended Forecast Model",

        best_row["model_name"]

    )







with rec_col2:



    st.metric(

        "Lowest MAPE",

        f"{best_row['mape']:.2f}%"

    )







st.success(

f"""

The recommended forecasting model is {best_row['model_name']}.



It achieved the lowest forecasting error among the evaluated models,

with a Mean Absolute Percentage Error (MAPE) of {best_row['mape']:.2f}%.

"""

)







# ==========================================

# FORECASTING APPROACH COMPARISON

# ==========================================





st.subheader(

    "Forecasting Approach Comparison"

)







approach_df = pd.DataFrame({



    "Approach":[

        "Machine Learning Model",

        "Statistical Model"

    ],



    "mae":[



        df[

            df["model_name"]=="Random Forest"

        ]["mae"].mean(),



        df[

            df["model_name"].isin(

                ["ARIMA","SARIMA"]

            )

        ]["mae"].mean()



    ],



    "mape":[



        df[

            df["model_name"]=="Random Forest"

        ]["mape"].mean(),



        df[

            df["model_name"].isin(

                ["ARIMA","SARIMA"]

            )

        ]["mape"].mean()



    ]



})







st.dataframe(

    approach_df,

    use_container_width=True

)







fig_approach = px.bar(



    approach_df,



    x="Approach",



    y="mape",



    title="Average MAPE Comparison by Forecasting Approach"



)







st.plotly_chart(

    fig_approach,

    use_container_width=True

)









fig_approach_mae = px.bar(



    approach_df,



    x="Approach",



    y="mae",



    title="Average MAE Comparison by Forecasting Approach"



)







st.plotly_chart(

    fig_approach_mae,

    use_container_width=True

)









# ==========================================

# HISTORICAL SALES SUMMARY

# ==========================================





st.divider()





st.subheader(

    "Historical Sales Summary"

)







hist_col1, hist_col2, hist_col3, hist_col4 = st.columns(4)







with hist_col1:



    st.metric(

        "Total Demand",

        f"{actual_df['daily_demand'].sum():,.0f}"

    )







with hist_col2:



    st.metric(

        "Average Daily Demand",

        f"{actual_df['daily_demand'].mean():,.2f}"

    )







with hist_col3:



    st.metric(

        "Highest Demand",

        f"{actual_df['daily_demand'].max():,.0f}"

    )







with hist_col4:



    highest_date = actual_df.loc[

        actual_df["daily_demand"].idxmax()

    ]["date"]





    st.metric(

        "Highest Demand Date",

        str(highest_date.date())

    )











# ==========================================

# ACTUAL VS FORECAST

# ==========================================





st.divider()





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



    title="Actual Demand Compared with Forecasted Demand"



)







st.plotly_chart(



    fig_compare,



    use_container_width=True



)









# ==========================================

# MONTHLY SALES TREND

# ==========================================





st.subheader(

    "Monthly Historical Sales Trend"

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









# ==========================================

# FORECAST TREND

# ==========================================





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











# ==========================================

# FORECAST SUMMARY

# ==========================================





st.subheader(

    "Forecast Summary"

)







sum1,sum2,sum3 = st.columns(3)







with sum1:



    st.metric(



        "Average Forecast",



        f"{model_data['predicted_demand'].mean():,.2f}"



    )





with sum2:



    st.metric(



        "Highest Forecast",



        f"{model_data['predicted_demand'].max():,.2f}"



    )





with sum3:



    st.metric(



        "Lowest Forecast",



        f"{model_data['predicted_demand'].min():,.2f}"



    )









# ==========================================

# DOWNLOAD SECTION

# ==========================================





st.divider()





st.subheader(

    "Download Forecast Results"

)







selected_csv = model_data.to_csv(



    index=False



).encode("utf-8")







st.download_button(



    "Download Selected Forecast CSV",



    selected_csv,



    file_name=f"{selected_model}_forecast.csv",



    mime="text/csv"



)









all_csv = df.to_csv(



    index=False



).encode("utf-8")







st.download_button(



    "Download All Forecast Results",



    all_csv,



    file_name="all_forecast_results.csv",



    mime="text/csv"



)









ranking_csv = ranking.to_csv(



    index=False



).encode("utf-8")







st.download_button(



    "Download Model Ranking Report",



    ranking_csv,



    file_name="model_ranking_report.csv",



    mime="text/csv"



)







# ==========================================

# FINAL DATA TABLE

# ==========================================





st.subheader(

    "Forecast Data"

)







st.dataframe(



    model_data,



    use_container_width=True



)
