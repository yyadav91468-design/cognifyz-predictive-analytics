
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_sales_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "daily_sales_features.csv"
PREDICTIONS_PATH = ROOT / "reports" / "test_predictions.csv"
COMPARISON_PATH = ROOT / "reports" / "model_comparison.csv"
IMPORTANCE_PATH = ROOT / "reports" / "feature_importance.csv"

st.set_page_config(
    page_title="Sales Intelligence | Cognifyz",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python -m src.train_model` first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run `python -m src.preprocess_data` first."
        )
    data = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return data.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_predictions():
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["Date"])
    return predictions.sort_values("Date").reset_index(drop=True)


@st.cache_data
def load_comparison():
    if not COMPARISON_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(COMPARISON_PATH)


@st.cache_data
def load_importance():
    if not IMPORTANCE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(IMPORTANCE_PATH)


st.sidebar.title("📊 Sales Intelligence")
st.sidebar.caption("Cognifyz Internship · Task 3")
page = st.sidebar.radio(
    "Navigate",
    [
        "Business Overview",
        "Model Evaluation",
        "Sales Prediction",
        "Project Information",
    ],
)

try:
    artifact = load_artifact()
    data = load_data()
    predictions = load_predictions()
    comparison = load_comparison()
    importance = load_importance()
except Exception as exc:
    st.error(str(exc))
    st.info(
        "From the project root, activate your virtual environment and run "
        "`python -m src.preprocess_data` followed by `python -m src.train_model`."
    )
    st.stop()


model = artifact["model"]
features = artifact["feature_columns"]
model_name = artifact.get("model_name", "Trained model")
currency = artifact.get("currency", "GBP")
metrics = artifact.get("test_metrics", {})

st.title("Sales Intelligence Dashboard")
st.caption(
    "Daily sales analysis, held-out model evaluation, and next-day sales estimation"
)

if page == "Business Overview":
    st.subheader("Business Overview")

    first_col, second_col, third_col, fourth_col = st.columns(4)

    first_col.metric("Model", model_name)
    second_col.metric("Daily records", f"{len(data):,}")

    if not data.empty:
        second_col.caption(
            f"From {data['Date'].min():%d %b %Y} to {data['Date'].max():%d %b %Y}"
        )

    if "sales_lag_0" in data.columns:
        modelled_sales_total = data["sales_lag_0"].sum()
        third_col.metric(
            "Sales in modelling rows",
            f"£{modelled_sales_total:,.0f}" if currency == "GBP"
            else f"{modelled_sales_total:,.0f} {currency}",
        )
    else:
        third_col.metric("Sales in modelling rows", "N/A")

    fourth_col.metric("Test R²", f"{metrics.get('R2', 0):.3f}")

    st.markdown("### Daily sales history")

    if "sales_lag_0" in data.columns:
        chart = px.line(
            data,
            x="Date",
            y="sales_lag_0",
            title="Recorded daily sales in the prepared modelling dataset",
            labels={
                "Date": "Date",
                "sales_lag_0": f"Daily sales ({currency})",
            },
        )
        chart.update_layout(hovermode="x unified")
        st.plotly_chart(chart, width="stretch")

    left, right = st.columns(2)

    with left:
        st.markdown("### Sales by month")
        if {"month", "sales_lag_0"}.issubset(data.columns):
            monthly = (
                data.groupby("month", as_index=False)["sales_lag_0"]
                .sum()
                .sort_values("month")
            )
            monthly["Month"] = monthly["month"].map(
                lambda m: pd.Timestamp(2000, int(m), 1).strftime("%b")
            )
            fig = px.bar(
                monthly,
                x="Month",
                y="sales_lag_0",
                labels={
                    "Month": "Month",
                    "sales_lag_0": f"Sales ({currency})",
                },
            )
            st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("### Weekday sales pattern")
        if {"day_of_week", "sales_lag_0"}.issubset(data.columns):
            weekday = (
                data.groupby("day_of_week", as_index=False)["sales_lag_0"]
                .mean()
                .sort_values("day_of_week")
            )
            weekday_names = {
                0: "Monday", 1: "Tuesday", 2: "Wednesday",
                3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday",
            }
            weekday["Weekday"] = weekday["day_of_week"].map(weekday_names)
            fig = px.bar(
                weekday,
                x="Weekday",
                y="sales_lag_0",
                labels={
                    "Weekday": "Day",
                    "sales_lag_0": f"Average sales ({currency})",
                },
            )
            st.plotly_chart(fig, width="stretch")

    st.caption(
        "Note: totals and charts use the prepared modelling rows after lag and "
        "rolling-window exclusions; they are not totals for every raw transaction."
    )


elif page == "Model Evaluation":
    st.subheader("Model Evaluation")
    st.caption("Metrics are from the chronological held-out test period.")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics.get('MAE', 0):,.2f} {currency}")
    c2.metric("RMSE", f"{metrics.get('RMSE', 0):,.2f} {currency}")
    c3.metric("R²", f"{metrics.get('R2', 0):.4f}")

    st.markdown(f"**Selected model:** {model_name}")

    if not predictions.empty and {
        "Date", "Actual_Sales", "Predicted_Sales"
    }.issubset(predictions.columns):
        st.markdown("### Actual vs predicted sales")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=predictions["Date"],
                y=predictions["Actual_Sales"],
                mode="lines",
                name="Actual sales",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=predictions["Date"],
                y=predictions["Predicted_Sales"],
                mode="lines",
                name="Predicted sales",
            )
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=f"Sales ({currency})",
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

        if "Absolute_Error" in predictions.columns:
            st.markdown("### Largest prediction errors")
            largest_errors = predictions.nlargest(10, "Absolute_Error")
            st.dataframe(
                largest_errors[
                    ["Date", "Actual_Sales", "Predicted_Sales", "Absolute_Error"]
                ],
                width="stretch",
                hide_index=True,
            )

        with st.expander("View all test predictions"):
            st.dataframe(predictions, width="stretch", hide_index=True)

    if not comparison.empty:
        st.markdown("### Model comparison")
        st.dataframe(comparison, width="stretch", hide_index=True)

        if {"Model", "RMSE"}.issubset(comparison.columns):
            fig = px.bar(
                comparison.sort_values("RMSE"),
                x="Model",
                y="RMSE",
                title="Test RMSE by model (lower is better)",
                labels={"RMSE": f"RMSE ({currency})"},
            )
            st.plotly_chart(fig, width="stretch")

    if not importance.empty:
        st.markdown("### Feature importance")
        importance.columns = [
            str(column).strip() for column in importance.columns
        ]
        name_col = next(
            (c for c in importance.columns
             if c.lower() in {"feature", "feature_name"}),
            None,
        )
        score_col = next(
            (c for c in importance.columns
             if c.lower() in {"importance", "feature_importance"}),
            None,
        )

        if name_col and score_col:
            ranked = importance.sort_values(score_col, ascending=True)
            fig = px.bar(
                ranked,
                x=score_col,
                y=name_col,
                orientation="h",
                title="Relative feature importance",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.dataframe(importance, width="stretch")


elif page == "Sales Prediction":
    st.subheader("Estimate Next-Day Sales")
    st.write(
        "The form starts with the latest row in the prepared dataset. "
        "Adjust the numeric features to explore a hypothetical prediction."
    )
    st.warning(
        "This is an experimental estimate based on historical data, not a "
        "guaranteed forecast. Keep inputs realistic and interpret results cautiously."
    )

    latest = data.iloc[-1]
    st.caption(f"Default feature values are based on {latest['Date']:%d %b %Y}.")

    with st.form("prediction_form"):
        values = {}

        st.markdown("#### Calendar features")
        calendar_columns = st.columns(3)

        for index, feature in enumerate(features):
            default_value = float(latest[feature])
            with calendar_columns[index % 3]:
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=default_value,
                    step=1.0 if feature in {
                        "day_of_week", "day_of_month", "month",
                        "day_of_year", "is_weekend",
                    } else 100.0,
                    format="%.2f",
                    help=f"Model input feature: {feature}",
                )

        submitted = st.form_submit_button(
            "Predict next-day sales",
            type="primary",
            width="stretch",
        )

    if submitted:
        try:
            input_frame = pd.DataFrame(
                [[values[name] for name in features]],
                columns=features,
            )
            predicted_sales = float(model.predict(input_frame)[0])

            st.markdown("### Prediction result")
            st.metric(
                "Estimated next-day sales",
                f"£{predicted_sales:,.2f}" if currency == "GBP"
                else f"{predicted_sales:,.2f} {currency}",
            )

            if predicted_sales < 0:
                st.info(
                    "The model produced a negative estimate. Actual sales "
                    "cannot be negative, and this result highlights model uncertainty."
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


else:
    st.subheader("Project Information")

    st.markdown(
        """
        **Project:** Sales Prediction & Business Intelligence Dashboard

        **Internship:** Cognifyz Technologies — Level 3, Advanced

        **Dataset:** UCI Online Retail transaction dataset

        **Technology stack**
        - Python, pandas, NumPy
        - scikit-learn for model training and evaluation
        - FastAPI for the prediction API
        - Streamlit and Plotly for interactive analytics
        - Joblib for model persistence

        **Workflow**
        1. Clean transaction data and remove invalid sales.
        2. Aggregate transactions into daily sales.
        3. Create calendar, lag, and rolling-window features.
        4. Train and compare regression models chronologically.
        5. Save the selected model and test predictions.
        6. Present historical analysis and predictions in this dashboard.

        **Evaluation caution:** A modest R² score indicates that much of the
        day-to-day sales variation remains unexplained. This is a baseline
        predictive analytics project and should not be treated as a production
        financial forecasting system.
        """
    )

    st.markdown("### Model details")
    st.json(
        {
            "model": model_name,
            "currency": currency,
            "feature_count": len(features),
            "features": features,
            "test_metrics": metrics,
        }
    )

st.sidebar.divider()
st.sidebar.caption("Built with Python · Streamlit · Plotly · scikit-learn")
