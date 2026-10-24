"""Shared data and evaluation helpers for the Shevanti forecasting project."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_and_clean_data(path):
    """Load the project's DATE/PRICES file (CSV extension is optional)."""
    df = pd.read_csv(path)
    normalized = {str(c).strip().lower(): c for c in df.columns}
    date_col = normalized.get("date") or df.columns[0]
    price_col = normalized.get("prices") or normalized.get("price")
    if price_col is None:
        price_col = df.columns[1] if len(df.columns) > 1 else None
    if date_col is None or price_col is None:
        raise ValueError("Expected one date column and one price column.")

    df = df[[date_col, price_col]].copy()
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"]).sort_values("date")
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found. Resolve them before training.")
    if (df["price"] <= 0).any():
        raise ValueError("Prices must be positive.")
    df = df.reset_index(drop=True)
    df["target"] = df["price"].shift(-1)
    return df


FEATURE_COLUMNS = [
    "lag_1", "lag_2", "lag_3", "lag_7", "rolling_mean_7",
    "rolling_mean_14", "rolling_std_7", "rolling_min_7", "rolling_max_7",
    "pct_change_1", "pct_change_7", "trend_7",
]


def make_features(prices):
    """Create features using current and past prices only."""
    p = pd.Series(prices, dtype=float).reset_index(drop=True)
    f = pd.DataFrame(index=p.index)
    # At index t, the target is price[t+1], so lag_1 is today's observed price.
    for lag in (1, 2, 3, 7):
        f[f"lag_{lag}"] = p.shift(lag - 1)
    f["rolling_mean_7"] = p.rolling(7).mean()
    f["rolling_mean_14"] = p.rolling(14).mean()
    f["rolling_std_7"] = p.rolling(7).std()
    f["rolling_min_7"] = p.rolling(7).min()
    f["rolling_max_7"] = p.rolling(7).max()
    f["pct_change_1"] = p.pct_change(1)
    f["pct_change_7"] = p.pct_change(7)
    f["trend_7"] = f["rolling_mean_7"] - f["rolling_mean_14"]
    return f


def get_splits(df, train_ratio=0.8):
    """Chronological train/test split; the final 20% is held out for testing."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    split_index = int(len(df) * train_ratio)
    if split_index < 1 or split_index >= len(df):
        raise ValueError("Not enough rows for the requested split.")
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def evaluate(df_slice, pred_col, label, verbose=True):
    y_true = df_slice["target"].to_numpy()
    y_pred = df_slice[pred_col].to_numpy()
    today = df_slice["price"].to_numpy()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100
    dir_acc = np.mean(np.sign(y_true - today) == np.sign(y_pred - today)) * 100
    result = {"model": label, "MAE": mae, "RMSE": rmse, "sMAPE": smape, "DirAcc": dir_acc}
    if verbose:
        print(f"{label:20s} MAE={mae:7.2f} RMSE={rmse:7.2f} sMAPE={smape:6.2f}% DirAcc={dir_acc:5.1f}%")
    return result
