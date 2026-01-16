import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SarimaxModel:
    def __init__(self, order, seasonal_order, train_frac: float = 1):
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

        self.exog_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        self.last_datetime = None  # serve per costruire exog future
        self.freq = None           # es. "H"

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

    def _add_time_exog(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "datetime" not in df.columns:
            raise ValueError("Missing required column: 'datetime'")
        if "price" not in df.columns:
            raise ValueError("Missing required column: 'price'")

        df["datetime"] = pd.to_datetime(df["datetime"])
        df["hour"] = df["datetime"].dt.hour
        df["dow"] = df["datetime"].dt.dayofweek

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"]  = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"]  = np.cos(2 * np.pi * df["dow"] / 7)
        return df

    def _make_future_exog(self, start_dt: pd.Timestamp, steps: int, freq: str = "H") -> pd.DataFrame:
        idx = pd.date_range(start=start_dt, periods=steps, freq=freq)
        tmp = pd.DataFrame({"datetime": idx})
        tmp = self._add_time_exog(tmp.assign(price=0.0))  # price dummy solo per riusare la funzione
        return tmp[self.exog_cols]

    def fit(self, data):
        data = self._ensure_pandas(data)
        data = self._add_time_exog(data)

        # idealmente dati orari regolari
        data = data.sort_values("datetime").reset_index(drop=True)
        self.last_datetime = data["datetime"].iloc[-1]

        y = data["price"]
        X = data[self.exog_cols]

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

    def forecast(self, steps: int, freq: str = "H", alpha: float = 0.05):
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

        pred = self.fitted_model.get_forecast(steps=steps, exog=X_future)
        y_hat = pred.predicted_mean
        conf = pred.conf_int(alpha=alpha)
        return y_hat, conf
