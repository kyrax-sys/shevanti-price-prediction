from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from src.xgboost_model import DATA_PATH, train_and_evaluate

st.set_page_config(page_title="Shevanti Price Forecast", page_icon="🌼", layout="wide")
st.title("🌼 Shevanti Price Forecast")
st.caption("Next-day price estimate from recent price history. No weather or festival inputs are used.")

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

@st.cache_data(show_spinner="Training XGBoost and preparing the forecast…")
def load_results():
    raw, dataset, train, test, model, metrics, final_model, forecast, explanation = train_and_evaluate()
    return raw, dataset, train, test, metrics, forecast, explanation

try:
    raw, dataset, train, test, metrics, forecast, explanation = load_results()
except Exception as exc:
    st.error(f"Could not prepare the forecast: {exc}")
    st.stop()

test_residuals = (test["target"] - test["pred_xgb"]).abs()
error_band = float(test_residuals.quantile(0.90))
left, right = st.columns(2)
left.metric("Next-day XGBoost estimate", f"{forecast:,.2f}", help=f"Based on data through {raw.date.max().date()}")
right.metric("Typical 90% test-error band", f"±{error_band:,.2f}", help="Approximate range from the 90th percentile of absolute errors on the held-out test period; not a guaranteed interval.")
st.info(f"Latest available observation: **{raw.date.max().date()}** — {raw.price.iloc[-1]:,.2f}. The forecast is for the next calendar day and uses historical prices only.")

st.subheader("Recent price history")
fig, ax = plt.subplots(figsize=(10, 3.2))
recent = raw.tail(180)
ax.plot(recent["date"], recent["price"], color="#287a58", linewidth=1.6)
ax.set_ylabel("Price")
ax.set_xlabel("")
ax.grid(alpha=.2)
fig.tight_layout()
st.pyplot(fig, clear_figure=True)

st.subheader("Why this forecast?")
st.caption("These are XGBoost contribution scores (SHAP contributions from the model). Positive scores push the estimate up; negative scores push it down. They describe model behavior, not market causes.")
top = explanation.head(6).sort_values("contribution")
fig, ax = plt.subplots(figsize=(8, 3.2))
colors = ["#bd5b50" if v < 0 else "#287a58" for v in top["contribution"]]
ax.barh(top["feature"], top["contribution"], color=colors)
ax.axvline(0, color="#777", linewidth=.8)
ax.set_xlabel("Contribution to estimate")
fig.tight_layout()
st.pyplot(fig, clear_figure=True)

st.subheader("Held-out model comparison")
st.caption("Chronological 80/20 split: the earliest 80% of feature-ready records train the model; the latest 20% are used once for out-of-sample evaluation.")
st.dataframe(metrics.round(2), use_container_width=True, hide_index=True)
with st.expander("Explore held-out forecasts"):
    chart = test.set_index("date")[["target", "pred_xgb"]].rename(columns={"target": "Actual next-day price", "pred_xgb": "XGBoost estimate"})
    st.line_chart(chart, height=280)
    st.dataframe(test[["date", "target", "pred_xgb"]].rename(columns={"target": "actual", "pred_xgb": "xgboost_estimate"}), use_container_width=True, hide_index=True)

st.caption("Dataset: data/processed/Flower_price_2010-2025 · Forecasts require recent observed prices and should be treated as estimates.")
