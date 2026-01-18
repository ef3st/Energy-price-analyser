from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class SpikeModelConfig:
    """Spike detection config."""
    q_threshold: float = 0.99  # define spikes on training residual distribution

class SpikeModel:
    """
    Spike model interface.

    Recommended evolution:
    - proba model: logistic regression / gradient boosting
    - severity model: quantile regression or EVT (POT+GPD)
    """

    def __init__(self, cfg: SpikeModelConfig = SpikeModelConfig()):
        self.cfg = cfg
        self._thr = None
        self._pos_mean = None

    def fit(self, residuals: pd.Series) -> "SpikeModel":
        r = residuals.dropna().astype(float).values
        self._thr = np.quantile(r, self.cfg.q_threshold)
        pos = r[r > self._thr]
        self._pos_mean = float(pos.mean()) if len(pos) else 0.0
        return self

    def predict_adjustment(self, residual_forecast: pd.Series) -> pd.Series:
        """
        Very simple spike adjustment:
        - probability proxy: 1 if residual_forecast > threshold else 0
        - severity: mean exceedance observed in training
        """
        if self._thr is None:
            raise RuntimeError("SpikeModel is not fitted.")

        proba = (residual_forecast.values > self._thr).astype(float)
        adj = proba * self._pos_mean
        return pd.Series(adj, index=residual_forecast.index, name="spike_adj")
