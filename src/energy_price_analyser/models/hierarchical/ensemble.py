from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

from .annual_baseline import AnnualBaselineModel, AnnualBaselineConfig
from .monthly_corrector import MonthlyCorrector, MonthlyCorrectorConfig
from .weekly_corrector import WeeklyCorrector, WeeklyCorrectorConfig
from .short_term_sarimax import ShortTermSarimax, ShortTermSarimaxConfig
from .spike_model import SpikeModel, SpikeModelConfig
from .utils import ensure_datetime_index

@dataclass
class HierarchicalConfig:
    annual: AnnualBaselineConfig = field(default_factory=AnnualBaselineConfig)
    monthly: MonthlyCorrectorConfig = field(default_factory=MonthlyCorrectorConfig)
    weekly: WeeklyCorrectorConfig = field(default_factory=WeeklyCorrectorConfig)
    short_term: ShortTermSarimaxConfig = field(default_factory=ShortTermSarimaxConfig)
    spike: SpikeModelConfig = field(default_factory=SpikeModelConfig)

    use_weekly: bool = True
    use_spikes: bool = True

class HierarchicalForecaster:
    """
    Hierarchical forecaster:
    y = baseline_year + corr_month + corr_week + resid_short + spike_adj
    """

    def __init__(self, cfg: HierarchicalConfig = HierarchicalConfig()):
        self.cfg = cfg
        self.annual = AnnualBaselineModel(cfg.annual)
        self.monthly = MonthlyCorrector(cfg.monthly)
        self.weekly = WeeklyCorrector(cfg.weekly)
        self.short = ShortTermSarimax(cfg.short_term)
        self.spike = SpikeModel(cfg.spike)

        self._trained_components = {}

    def fit(self, df: pd.DataFrame) -> "HierarchicalForecaster":
        df = ensure_datetime_index(df)

        # Fit annual baseline
        self.annual.fit(df)
        base = self.annual.predict(df[["datetime"]])

        # Fit monthly correction
        self.monthly.fit(df, baseline=base)
        cm = self.monthly.predict(df[["datetime"]])
        base_m = base + cm

        # Fit weekly correction (optional)
        if self.cfg.use_weekly:
            self.weekly.fit(df, baseline_plus_month=base_m)
            cw = self.weekly.predict(df[["datetime"]])
        else:
            cw = 0.0

        # Residual for short-term model
        if isinstance(cw, pd.Series):
            slow = base_m + cw
        else:
            slow = base_m

        resid = df.set_index("datetime")["price"].astype(float) - slow.reindex(df["datetime"]).values
        resid = pd.Series(resid.values, index=df["datetime"], name="resid_lvl2")

        # Fit short-term residual dynamics
        self.short.fit(df, residual_series=resid)

        # Fit spike model on (in-sample) residuals of short-term (or lvl2 residuals)
        if self.cfg.use_spikes:
            self.spike.fit(resid)

        self._trained_components = {
            "annual": True,
            "monthly": True,
            "weekly": self.cfg.use_weekly,
            "short": True,
            "spike": self.cfg.use_spikes,
        }
        return self

    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame:
        future_df = ensure_datetime_index(future_df)

        base = self.annual.predict(future_df[["datetime"]])
        cm = self.monthly.predict(future_df[["datetime"]])
        out = pd.DataFrame(index=future_df["datetime"])
        out["baseline_year"] = base.values
        out["corr_month"] = cm.values

        y = base + cm

        if self.cfg.use_weekly:
            cw = self.weekly.predict(future_df[["datetime"]])
            out["corr_week"] = cw.values
            y = y + cw
        else:
            out["corr_week"] = 0.0

        rhat = self.short.predict(future_df[["datetime"]])
        out["resid_short"] = rhat.values
        y = y + rhat

        if self.cfg.use_spikes:
            adj = self.spike.predict_adjustment(rhat)
            out["spike_adj"] = adj.values
            y = y + adj
        else:
            out["spike_adj"] = 0.0

        out["y_hat"] = y.values
        return out.reset_index(names=["datetime"])
