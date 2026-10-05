
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
