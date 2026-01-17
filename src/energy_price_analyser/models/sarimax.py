import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json
from pathlib import Path


from tools.logger import get_logger, log_performance
EXOG_COLS = [
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_2",
    "month_3",
    "month_4",
    "month_5",
    "month_6",
    "month_7",
    "month_8",
    "month_9",
    "month_10",
    "month_11",
    "month_12",
    "is_holiday",
    "is_preholiday",
    # "year_sin_1", "year_cos_1",
    # "year_sin_2", "year_cos_2",
    # "is_dst",
]

FIT_RESULT_DIR = Path(r"C:\Users\utente\Desktop\CARRIER\NE\energy-price-analyser\fit_results\SARIMAX")




class SarimaxModel:
    def __init__(self, order, seasonal_order, train_frac: float = 1, load_from: str = None, exog_cols=EXOG_COLS):
        """
        Wrapper class for a SARIMAX (Seasonal ARIMA with eXogenous variables) model,
        tailored for hourly electricity price time series.

        Parameters
        ----------
        order : tuple(int, int, int)
            Non-seasonal ARIMA order (p, d, q):

            - p (autoregressive order):
                Number of lagged observations y_{t-1}, ..., y_{t-p} used to model
                short-term temporal persistence in the price series.

            - d (order of differencing):
                Number of non-seasonal differences applied to the series to achieve
                stationarity. For electricity prices, d=1 is typically used to model
                price variations rather than absolute levels.

            - q (moving average order):
                Number of lagged forecast errors ε_{t-1}, ..., ε_{t-q} included to
                capture short-lived shocks and corrective dynamics.

        seasonal_order : tuple(int, int, int, int)
            Seasonal ARIMA order (P, D, Q, s):

            - P (seasonal autoregressive order):
                Number of lagged seasonal observations y_{t-s}, y_{t-2s}, ... used
                to model persistence across seasonal cycles.

            - D (seasonal differencing order):
                Number of seasonal differences applied to remove seasonal
                non-stationarity. With hourly data, D=1 removes daily level effects
                by differencing y_t - y_{t-s}.

            - Q (seasonal moving average order):
                Number of lagged seasonal forecast errors ε_{t-s}, ε_{t-2s}, ...
                capturing recurring seasonal shocks.

            - s (seasonal period):
                Length of the seasonal cycle. For hourly electricity prices,
                s=24 corresponds to the daily cycle.

        Notes
        -----
        The combination (p, d, q) × (P, D, Q, s) defines the full stochastic structure
        of the SARIMAX model, describing both short-term dynamics and recurring
        seasonal behavior. A common and well-established specification for
        electricity spot prices is:

            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 24)

        which captures hourly persistence, daily seasonality, and transient price
        shocks while maintaining a parsimonious parameterization.
        

        Examples
        --------
        Basic usage with hourly electricity price data:

        >>> import pandas as pd
        >>> from sarimax_model import SarimaxModel
        >>>
        >>> # DataFrame with columns: ['datetime', 'price']
        >>> df = pd.read_csv("prices.csv", parse_dates=["datetime"])
        >>>
        >>> model = SarimaxModel(
        ...     order=(1, 1, 1),
        ...     seasonal_order=(1, 1, 1, 24)
        ... )
        >>>
        >>> model.fit(df)
        >>>
        >>> # Forecast next 24 hours
        >>> y_hat, conf_int = model.forecast(steps=24)
        >>>
        >>> print(y_hat.head())
        >>> print(conf_int.head())

        The forecast returns the expected price level and the corresponding
        confidence intervals, reconstructed automatically from the differenced
        model.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.train_frac = train_frac

        self.model = None
        self.fitted_model = None

        self.exog_cols = exog_cols#["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        self.last_datetime = None  # serve per costruire exog future
        self.freq = "h"           # es. "H"
        if load_from is not None:
            loaded = self.load(load_from)
            self.__dict__.update(loaded.__dict__)
        
        self.logger = get_logger("enprice")

    @staticmethod
    def _ensure_pandas(data):
        if isinstance(data, pd.DataFrame):
            return data
        try:
            import polars as pl
            if isinstance(data, pl.DataFrame):
                return data.to_pandas()
        except Exception:
            pass
        raise ValueError("Input data must be a pandas DataFrame or polars DataFrame.")

    def _add_time_exog(self, df: pd.DataFrame, annual_K: int = 2) -> pd.DataFrame:
        df = df.copy()

        if "datetime" not in df.columns:
            raise ValueError("Missing required column: 'datetime'")
        if "price" not in df.columns:
            raise ValueError("Missing required column: 'price'")

        df["datetime"] = pd.to_datetime(df["datetime"])

        # --- Basic time components ---
        df["hour"] = df["datetime"].dt.hour
        df["dow"] = df["datetime"].dt.dayofweek  # Monday=0
        df["month"] = df["datetime"].dt.month

        # --- Week structure ---
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
        #NOTE DOW sin/cos (smooth weekly cycle)
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
        
        # Day-of-week dummies (drop one to avoid multicollinearity)
        dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=True)
        df = pd.concat([df, dow_dummies], axis=1)

        # Month dummies
        month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
        df = pd.concat([df, month_dummies], axis=1)

        # --- Daylight Saving Time (Europe/Rome) robust across pandas versions ---
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Europe/Rome")

        def _is_dst(ts: pd.Timestamp) -> int:
            ts = pd.Timestamp(ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=tz)
            else:
                ts = ts.astimezone(tz)
            return int(bool(ts.dst()))

        df["is_dst"] = df["datetime"].apply(_is_dst).astype(int)

        # --- Holidays / pre-holidays placeholders ---
        if "is_holiday" not in df.columns:
            df["is_holiday"] = 0
        if "is_preholiday" not in df.columns:
            df["is_preholiday"] = 0
            

        # --- Yearly Fourier (continuous annual seasonality) ---
        dt = df["datetime"]
        doy = dt.dt.dayofyear.astype(float)
        frac_day = (dt.dt.hour + dt.dt.minute / 60.0 + dt.dt.second / 3600.0) / 24.0
        t_year = doy + frac_day

        period = 365.25
        annual_K = int(max(1, annual_K))

        for k in range(1, annual_K + 1):
            df[f"year_sin_{k}"] = np.sin(2.0 * np.pi * k * t_year / period)
            df[f"year_cos_{k}"] = np.cos(2.0 * np.pi * k * t_year / period)

        # Optional: if you want single names when K=1
        if annual_K == 1:
            df.rename(
                columns={"year_sin_1": "year_sin", "year_cos_1": "year_cos"},
                inplace=True,
            )


        return df



    def _make_future_exog(self, start_dt: pd.Timestamp, steps: int, freq: str = "h") -> pd.DataFrame:
        idx = pd.date_range(start=start_dt, periods=steps, freq=freq)
        tmp = pd.DataFrame({"datetime": idx})
        tmp = self._add_time_exog(tmp.assign(price=0.0))  # price dummy solo per riusare la funzione
    
        # Allinea colonne a quelle del training; crea le mancanti e le mette a 0
        tmp = tmp.reindex(columns=self.exog_cols, fill_value=0)
        # bool -> int
        bool_cols = tmp.select_dtypes(include=["bool"]).columns
        tmp[bool_cols] = tmp[bool_cols].astype(int)

        # forza tutto a numerico
        tmp = tmp.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    
        return tmp
    
    
    @log_performance("fit_sarimax")
    def fit(self, data):
        data = self._ensure_pandas(data)
        data = self._add_time_exog(data)

        # idealmente dati orari regolari
        data = data.sort_values("datetime").reset_index(drop=True)
        self.last_datetime = data["datetime"].iloc[-1]

        y = data["price"]
        X = data[self.exog_cols].copy()
        # Convert booleans to int
        bool_cols = X.select_dtypes(include=["bool"]).columns
        X[bool_cols] = X[bool_cols].astype(int)
        
        # Force all columns to numeric (dummies can become object in some pipelines)
        X = X.apply(pd.to_numeric, errors="coerce")
        
        # Handle missing values (choose one strategy)
        X = X.fillna(0.0)
        
        y = pd.to_numeric(data["price"], errors="coerce").astype(float)
        y = y.ffill().bfill()

        train_size = int(len(data) * self.train_frac)
        y_train, X_train = y.iloc[:train_size], X.iloc[:train_size]

        self.model = SARIMAX(
            y_train,
            exog=X_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.fitted_model = self.model.fit(disp=False)
        return self

    def summary(self):
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before summarizing.")
        return self.fitted_model.summary()

    def forecast(self, steps: int, freq: str = "h", alpha: float = 0.05):
        """
        Forecast out-of-sample for 'steps' periods ahead.
        Builds exog future (calendar features) automatically.
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting.")
        if self.last_datetime is None:
            raise ValueError("Missing last_datetime; fit the model first.")

        start_dt = self.last_datetime + pd.tseries.frequencies.to_offset(freq)
        X_future = self._make_future_exog(start_dt=start_dt, steps=steps, freq=freq)

        self.logger.debug(f"types: {X_future.dtypes}")
        self.logger.debug(f"head: {X_future.head()}")
        pred = self.fitted_model.get_forecast(steps=steps, exog=X_future)
        y_hat = pred.predicted_mean
        conf = pred.conf_int(alpha=alpha)
        return y_hat, conf




    def save(self, path: str | Path):
        """
        Save fitted SARIMAX results + minimal metadata required to forecast later.
        Creates:
          - <path>.pkl  (statsmodels results)
          - <path>.json (metadata)
        """
        if self.fitted_model is None:
            raise ValueError("Nothing to save: model is not fitted (fitted_model is None).")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 1) Save statsmodels results (pickle)
        results_path = path.with_suffix(".pkl")
        self.fitted_model.save(str(results_path))

        # 2) Save metadata required to rebuild future exog and timeline
        meta = {
            "order": list(self.order),
            "seasonal_order": list(self.seasonal_order),
            "exog_cols": list(self.exog_cols),
            "last_datetime": None if self.last_datetime is None else str(pd.Timestamp(self.last_datetime)),
            "freq": self.freq,  # optional, but useful
            "train_frac": self.train_frac,

            # environment info (helps debugging incompatibilities)
            "versions": {
                "python": None,
                "pandas": pd.__version__,
                # "statsmodels": statsmodels_version,
                "numpy": np.__version__,
            },
        }

        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


    @classmethod
    def load(cls, path: str | Path):
        """
        Load fitted SARIMAX results + metadata.
        """
        path = Path(path)
        results_path = path.with_suffix(".pkl")
        meta_path = path.with_suffix(".json")

        if not results_path.exists():
            raise FileNotFoundError(f"Missing results file: {results_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        obj = cls(
            order=tuple(meta["order"]),
            seasonal_order=tuple(meta["seasonal_order"]),
            train_frac=float(meta.get("train_frac", 1.0)),
            load_from=None,
        )

        obj.exog_cols = meta["exog_cols"]
        obj.last_datetime = None if meta.get("last_datetime") is None else pd.to_datetime(meta["last_datetime"])
        obj.freq = meta.get("freq", None)

        obj.fitted_model = SARIMAXResults.load(str(results_path))
        obj.model = None  # not needed for forecast; results contain everything needed

        return obj