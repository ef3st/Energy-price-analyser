from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

from .utils import ensure_datetime_index

@dataclass
class MonthlyCorrectorConfig:
    """Model monthly mean residuals."""
    order: tuple = (1, 0, 0)  # AR(1) on monthly residuals is often enough

class MonthlyCorrector:
    """
    Monthly correction model.
    Learns a model on aggregated monthly residuals and expands it back to hourly resolution.
    """

    def __init__(self, cfg: MonthlyCorrectorConfig = MonthlyCorrectorConfig()):
        self.cfg = cfg
        self._fit_res = None
        self._last_month = None

    def fit(self, df: pd.DataFrame, baseline: pd.Series) -> "MonthlyCorrector":
        df = ensure_datetime_index(df)
        base = baseline.reindex(df["datetime"]).values
        resid = df["price"].values - base

        tmp = pd.DataFrame({"datetime": df["datetime"], "resid": resid})
        tmp["month"] = tmp["datetime"].dt.to_period("M")
        monthly = tmp.groupby("month")["resid"].mean().astype(float)

        self._last_month = monthly.index.max()
        mod = SARIMAX(monthly.values, order=self.cfg.order, seasonal_order=(0, 0, 0, 0))
        self._fit_res = mod.fit(disp=False)
        self._monthly_index = monthly.index
        return self

    def predict(self, future_df: pd.DataFrame) -> pd.Series:
        if self._fit_res is None:
            raise RuntimeError("MonthlyCorrector is not fitted.")

        future_df = ensure_datetime_index(future_df)
        future_months = future_df["datetime"].dt.to_period("M").unique()
        steps = len(future_months)

        monthly_hat = self._fit_res.get_forecast(steps=steps).predicted_mean
        monthly_hat = pd.Series(monthly_hat, index=future_months)

        # Expand to hourly: map each datetime's month to the monthly value
        out = future_df["datetime"].dt.to_period("M").map(monthly_hat).astype(float)
        return pd.Series(out.values, index=future_df["datetime"], name="corr_month")
