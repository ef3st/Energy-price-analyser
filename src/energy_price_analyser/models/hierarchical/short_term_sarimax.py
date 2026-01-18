from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from .utils import ensure_datetime_index, add_time_features

@dataclass
class ShortTermSarimaxConfig:
    """Short-term residual dynamics."""
    order: tuple = (2, 0, 2)
    seasonal_order: tuple = (1, 0, 1, 24)  # daily seasonality
    exog_cols: tuple = (
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    )

class ShortTermSarimax:
    """
    Short-term model for residuals after removing slow components.
    This is the workhorse for H+1..H+168.
    """

    def __init__(self, cfg: ShortTermSarimaxConfig = ShortTermSarimaxConfig()):
        self.cfg = cfg
        self._fit_res = None
        self._last_dt = None

    def fit(self, df: pd.DataFrame, residual_series: pd.Series) -> "ShortTermSarimax":
        df = ensure_datetime_index(df)
        df = add_time_features(df)

        r = residual_series.reindex(df["datetime"]).astype(float).values
        X = df[list(self.cfg.exog_cols)].astype(float).values

        mod = SARIMAX(
            endog=r,
            exog=X,
            order=self.cfg.order,
            seasonal_order=self.cfg.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._fit_res = mod.fit(disp=False)
        self._last_dt = df["datetime"].iloc[-1]
        return self

    def predict(self, future_df: pd.DataFrame) -> pd.Series:
        if self._fit_res is None:
            raise RuntimeError("ShortTermSarimax is not fitted.")

        future_df = ensure_datetime_index(future_df)
        future_df = add_time_features(future_df)
        Xf = future_df[list(self.cfg.exog_cols)].astype(float).values

        rhat = self._fit_res.get_forecast(steps=len(future_df), exog=Xf).predicted_mean
        return pd.Series(rhat, index=future_df["datetime"], name="resid_short")
