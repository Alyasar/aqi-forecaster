import pandas as pd
import streamlit as st

st.set_page_config(page_title="Delhi AQI Forecaster", page_icon="🌫️", layout="wide")

# India's National AQI categories for PM2.5 (µg/m³)
CATEGORIES = [
    (30, "Good", "#00b050"),
    (60, "Satisfactory", "#92d050"),
    (90, "Moderate", "#ffd700"),
    (120, "Poor", "#ff9900"),
    (250, "Very Poor", "#ff0000"),
    (float("inf"), "Severe", "#a00000"),
]

def category(value):
    for limit, name, color in CATEGORIES:
        if value <= limit:
            return name, color

@st.cache_data
def load_data():
    preds = pd.read_csv("reports/test_predictions.csv", index_col=0, parse_dates=True)
    table = pd.read_csv("reports/evaluation_table.csv")
    return preds, table

preds, table = load_data()

st.title("🌫️ Delhi PM2.5 Forecaster")
st.write(
    "Forecasts PM2.5 at the **Shadipur (CPCB)** station **24 hours ahead**, "
    "using current pollution and the next day's weather. "
    "All forecasts below are out-of-sample: the model never saw these days during training."
)

day = st.date_input(
    "Pick a day",
    value=pd.Timestamp("2025-11-15"),
    min_value=preds.index.min().date(),
    max_value=preds.index.max().date(),
)

selected = preds.loc[str(day)]
if selected.empty:
    st.warning("No data for this day (the sensor was offline).")
    st.stop()

col1, col2 = st.columns(2)
for col, label, value in [
    (col1, "Forecast (made 24h earlier)", selected["predicted"].mean()),
    (col2, "What actually happened", selected["actual"].mean()),
]:
    name, color = category(value)
    col.metric(label, f"{value:.0f} µg/m³")
    col.markdown(
        f"<span style='background:{color};padding:4px 12px;border-radius:6px;"
        f"color:black;font-weight:600'>{name}</span>",
        unsafe_allow_html=True,
    )

st.subheader("Hour by hour")
st.line_chart(selected[["actual", "predicted", "baseline"]])
st.caption("Baseline = 'PM2.5 in 24 hours will be the same as now'.")

st.subheader("How accurate is the model?")
st.dataframe(table, hide_index=True)
st.caption(
    "Test period: Jul–Dec 2025. The model retrains monthly. Lower MAE/RMSE and higher R² are better. "
    "It matches the baseline on calm days and is more accurate when pollution changes sharply."
)