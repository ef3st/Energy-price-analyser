import numpy as np
import pandas as pd

def ensure_datetime_index(df: pd.DataFrame, dt_col: str = "datetime") -> pd.DataFrame:
    """Ensure df has a timezone-aware datetime column and is sorted."""
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col])
    out = out.sort_values(dt_col)
    return out

def add_time_features(df: pd.DataFrame, dt_col="datetime") -> pd.DataFrame:
    """Add basic calendar features for hourly energy price modelling."""
    out = df.copy()
    dt = pd.to_datetime(out[dt_col])
    out["hour"] = dt.dt.hour
    out["dow"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    # Smooth cycles
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)

    # Annual cycle (1-2 harmonics are usually enough)
    doy = dt.dt.dayofyear
    out["year_sin_1"] = np.sin(2 * np.pi * doy / 365.25)
    out["year_cos_1"] = np.cos(2 * np.pi * doy / 365.25)
    out["year_sin_2"] = np.sin(4 * np.pi * doy / 365.25)
    out["year_cos_2"] = np.cos(4 * np.pi * doy / 365.25)
    return out
