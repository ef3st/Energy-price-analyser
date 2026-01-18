from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

from .utils import ensure_datetime_index, add_time_features

@dataclass
class AnnualBaselineConfig:
    """Configuration for the annual baseline model."""
    order: tuple = (1, 0, 1)  # Parsimonious
    seasonal_order: tuple = (0, 0, 0, 0)  # Usually none for baseline
    exog_cols: tuple = (
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "year_sin_1", "year_cos_1",
        "year_sin_2", "year_cos_2",
        "is_weekend",
    )

class AnnualBaselineModel:
    """
    Smooth annual baseline model to forecast the expected price shape for next year.

    Notes:
    - Intentionally low-variance (parsimonious).
    - Uses calendar/Fourier features, not spike-sensitive.
    """

    def __init__(self, cfg: AnnualBaselineConfig = AnnualBaselineConfig()):
        self.cfg = cfg
        self._fit_res = None

    def fit(self, df: pd.DataFrame) -> "AnnualBaselineModel":
        df = ensure_datetime_index(df)
        df = add_time_features(df)

        y = df["price"].astype(float).values
        X = df[list(self.cfg.exog_cols)].astype(float).values

        mod = SARIMAX(
            endog=y,
            exog=X,
            order=self.cfg.order,
            seasonal_order=self.cfg.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._fit_res = mod.fit(disp=False)
        return self

    def predict(self, future_df: pd.DataFrame) -> pd.Series:
        """Return baseline forecast for provided future timestamps."""
        if self._fit_res is None:
            raise RuntimeError("AnnualBaselineModel is not fitted.")

        future_df = ensure_datetime_index(future_df)
        future_df = add_time_features(future_df)
        Xf = future_df[list(self.cfg.exog_cols)].astype(float).values

        yhat = self._fit_res.get_forecast(steps=len(future_df), exog=Xf).predicted_mean
        return pd.Series(yhat, index=future_df["datetime"], name="baseline_year")
