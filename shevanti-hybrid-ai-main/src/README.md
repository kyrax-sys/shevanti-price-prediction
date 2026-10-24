## Source

`shared_utils.py` contains shared CSV loading/validation, lag and rolling-price
features, chronological 80/20 splitting, and common evaluation metrics.

`xgboost_model.py` trains and evaluates the XGBoost component. From the project
root, run `python src/xgboost_model.py`. It reads
`data/processed/Flower_price_2010-2025`, holds out the latest 20% of
feature-ready records, and writes test predictions and metric tables under
`results/tables/`. It then refits on all labeled data for the app's next-day
estimate.

The agreed split ratio is chronological 80/20 for both XGBoost and ARIMA. The
ARIMA owner's notebook/model is separate from the XGBoost code.
