from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from .utils import ensure_datetime_index

@dataclass
class WeeklyCorrectorConfig:
    """Model weekly mean residuals."""
    order: tuple = (1, 0, 0)

class WeeklyCorrector:
    """
    Weekly correction model.
    Learns aggregated weekly residuals (ISO weeks) and expands back to hourly resolution.
    """

    def __init__(self, cfg: WeeklyCorrectorConfig = WeeklyCorrectorConfig()):
        self.cfg = cfg
        self._fit_res = None

    @staticmethod
    def _iso_week(period_dt: pd.Series) -> pd.PeriodIndex:
        # ISO week as Year-Week period
        iso = period_dt.dt.isocalendar()
        return (iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)).astype("period[W]")

    def fit(self, df: pd.DataFrame, baseline_plus_month: pd.Series) -> "WeeklyCorrector":
        df = ensure_datetime_index(df)
        base = baseline_plus_month.reindex(df["datetime"]).values
        resid = df["price"].values - base

        tmp = pd.DataFrame({"datetime": df["datetime"], "resid": resid})
        tmp["week"] = tmp["datetime"].dt.to_period("W")  # acceptable weekly grouping
        weekly = tmp.groupby("week")["resid"].mean().astype(float)

        mod = SARIMAX(weekly.values, order=self.cfg.order, seasonal_order=(0, 0, 0, 0))
        self._fit_res = mod.fit(disp=False)
        self._weekly_index = weekly.index
        return self

    def predict(self, future_df: pd.DataFrame) -> pd.Series:
        if self._fit_res is None:
            raise RuntimeError("WeeklyCorrector is not fitted.")

        future_df = ensure_datetime_index(future_df)
        future_weeks = future_df["datetime"].dt.to_period("W").unique()
        steps = len(future_weeks)

        weekly_hat = self._fit_res.get_forecast(steps=steps).predicted_mean
        weekly_hat = pd.Series(weekly_hat, index=future_weeks)

        out = future_df["datetime"].dt.to_period("W").map(weekly_hat).astype(float)
        return pd.Series(out.values, index=future_df["datetime"], name="corr_week")
