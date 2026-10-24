"""Train and evaluate the XGBoost price-history model."""

from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor

try:
    from .shared_utils import FEATURE_COLUMNS, evaluate, get_splits, load_and_clean_data, make_features
except ImportError:  # direct execution: python src/xgboost_model.py
    from shared_utils import FEATURE_COLUMNS, evaluate, get_splits, load_and_clean_data, make_features

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "Flower_price_2010-2025"


def train_and_evaluate(data_path=DATA_PATH):
    raw = load_and_clean_data(data_path)
    features = make_features(raw["price"])
    dataset = pd.concat([raw[["date", "price", "target"]], features], axis=1)
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    train, test = get_splits(dataset, train_ratio=0.8)

    model = XGBRegressor(
        n_estimators=500, learning_rate=0.04, max_depth=4,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        objective="reg:squarederror", eval_metric="mae",
        random_state=42, n_jobs=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train["target"])
    test = test.copy()
    test["pred_naive"] = test["price"]
    test["pred_ma7"] = test["rolling_mean_7"]
    test["pred_xgb"] = model.predict(test[FEATURE_COLUMNS])
    metrics = pd.DataFrame([
        evaluate(test, "pred_naive", "Naive (today)", verbose=False),
        evaluate(test, "pred_ma7", "7-day average", verbose=False),
        evaluate(test, "pred_xgb", "XGBoost", verbose=False),
    ])

    # Refit on all available labeled history for the live next-day forecast.
    final_model = XGBRegressor(
        n_estimators=500, learning_rate=0.04, max_depth=4,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        objective="reg:squarederror", eval_metric="mae",
        random_state=42, n_jobs=-1,
    ).fit(dataset[FEATURE_COLUMNS], dataset["target"])
    last_features = make_features(raw["price"]).iloc[[-1]][FEATURE_COLUMNS]
    forecast = float(final_model.predict(last_features)[0])
    contributions = final_model.get_booster().predict(
        __import__("xgboost").DMatrix(last_features), pred_contribs=True
    )[0][:-1]
    explanation = pd.DataFrame({"feature": FEATURE_COLUMNS, "contribution": contributions})
    explanation["abs_contribution"] = explanation["contribution"].abs()
    explanation = explanation.sort_values("abs_contribution", ascending=False)
    return raw, dataset, train, test, model, metrics, final_model, forecast, explanation


if __name__ == "__main__":
    raw, dataset, train, test, model, metrics, _, forecast, explanation = train_and_evaluate()
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    test[["date", "target", "pred_xgb"]].rename(columns={"target": "actual"}).to_csv(out / "predictions_kiran_xgb.csv", index=False)
    metrics.to_csv(out / "xgb_metrics.csv", index=False)
    print(f"Data: {raw.date.min().date()} to {raw.date.max().date()} ({len(raw):,} rows)")
    print(f"Chronological split: {len(train):,} train rows ({len(train)/len(dataset):.0%}), {len(test):,} test rows ({len(test)/len(dataset):.0%})")
    print(metrics.to_string(index=False))
    print(f"Next-day forecast after {raw.date.max().date()}: {forecast:.2f}")
