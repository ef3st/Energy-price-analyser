from __future__ import annotations
import pandas as pd
import numpy as np

from .ensemble import HierarchicalForecaster, HierarchicalConfig
from .utils import ensure_datetime_index

def rolling_backtest(
    df: pd.DataFrame,
    cfg: HierarchicalConfig,
    start_train_end: str,
    step_hours: int = 24,
    horizon_hours: int = 48,
) -> pd.DataFrame:
    """
    Rolling-origin backtest for short-term accuracy.
    """
    df = ensure_datetime_index(df)
    df = df.set_index("datetime")

    t = pd.Timestamp(start_train_end)
    end = df.index.max() - pd.Timedelta(hours=horizon_hours)

    rows = []
    while t <= end:
        train = df.loc[:t].reset_index()
        test_idx = pd.date_range(t + pd.Timedelta(hours=1), periods=horizon_hours, freq="h")
        test = df.loc[test_idx].reset_index()

        model = HierarchicalForecaster(cfg).fit(train)
        pred = model.predict(test[["datetime"]])

        merged = pred.merge(test[["datetime", "price"]], on="datetime", how="left")
        merged["train_end"] = t
        rows.append(merged)

        t += pd.Timedelta(hours=step_hours)

    return pd.concat(rows, ignore_index=True)

def compute_metrics(bt: pd.DataFrame) -> dict:
    e = bt["price"] - bt["y_hat"]
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return {"MAE": mae, "RMSE": rmse}
