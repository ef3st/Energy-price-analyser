import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from models.sarimax import SarimaxModel
from tools.logger import setup_logger, get_logger


# ----------------------------------------------------------------------
# Exogenous feature sets to test (grid over combinations)
# ----------------------------------------------------------------------
EXOG_SETS = [
    # 1) Weekly structure
    # ["is_weekend"],
    ["dow_sin", "dow_cos"],
    ["is_weekend", "dow_sin", "dow_cos"],

    # 2) Weekly + DST
    # ["is_dst"],
    # ["is_weekend", "is_dst"],
    ["dow_sin", "dow_cos", "is_dst"],
    # ["is_weekend", "dow_sin", "dow_cos", "is_dst"],

    # 3) Annual seasonality (Fourier) – 1 harmonic
    # ["year_sin_1", "year_cos_1"],
    # ["dow_sin", "dow_cos", "year_sin_1", "year_cos_1"],
    # ["is_weekend", "dow_sin", "dow_cos", "year_sin_1", "year_cos_1"],
    # ["is_dst", "year_sin_1", "year_cos_1"],
    # ["is_weekend", "dow_sin", "dow_cos", "year_sin_1", "year_cos_1", "is_dst"],

    # 4) Annual seasonality (Fourier) – 2 harmonics (use only if multi-year training)
    # ["year_sin_1", "year_cos_1", "year_sin_2", "year_cos_2"],
    # ["dow_sin", "dow_cos", "year_sin_1", "year_cos_1", "year_sin_2", "year_cos_2"],
    # ["is_weekend", "dow_sin", "dow_cos", "year_sin_1", "year_cos_1", "year_sin_2", "year_cos_2"],
    # ["is_weekend", "dow_sin", "dow_cos", "year_sin_1", "year_cos_1", "year_sin_2", "year_cos_2", "is_dst"],

    # 5) Calendar effects (only if clean and binary)
    # ["is_holiday"],
    # ["is_preholiday"],
    # ["is_holiday", "is_preholiday"],
    # ["is_weekend", "is_holiday"],
    # ["is_weekend", "is_holiday", "is_preholiday"],
    # ["dow_sin", "dow_cos", "is_holiday", "is_preholiday"],
    # ["is_weekend", "dow_sin", "dow_cos", "is_holiday", "is_preholiday"],

    # 6) Monthly dummies (alternative to Fourier; never together)
    # ["month_2", "month_3", "month_4", "month_5", "month_6", "month_7",
    #  "month_8", "month_9", "month_10", "month_11", "month_12"],
    # ["is_weekend", "month_2", "month_3", "month_4", "month_5", "month_6", "month_7",
    #  "month_8", "month_9", "month_10", "month_11", "month_12"],
    # ["dow_sin", "dow_cos", "month_2", "month_3", "month_4", "month_5", "month_6", "month_7",
    #  "month_8", "month_9", "month_10", "month_11", "month_12"],
    # ["is_weekend", "dow_sin", "dow_cos", "month_2", "month_3", "month_4", "month_5", "month_6", "month_7",
    #  "month_8", "month_9", "month_10", "month_11", "month_12"],
]


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def set_logger():
    setup_logger("enprice", log_level="DEBUG")
    return get_logger("enprice")


def get_dataset(year: int = 2018) -> pl.DataFrame:
    df = pl.read_csv(f"data/row_{year}.csv", separator=";")
    df = df.with_columns(pl.col("€/MWh").str.replace_all(",", ".").cast(pl.Float64))
    df = df.rename({"€/MWh": "price"})

    tz = "Europe/Rome"
    df = df.with_columns(
        pl.col("Data").str.strptime(pl.Date, "%d/%m/%Y").alias("date")
    ).with_columns(
        (
            # 00:00 of the day, interpreted as local time (Europe/Rome)
            pl.col("date").cast(pl.Datetime).dt.replace_time_zone(tz)
            +
            # Hour: 1->01:00, 24->00:00 next day, 25->01:00 next day, etc.
            pl.duration(hours=pl.col("Ora"))
        ).alias("datetime")
    )
    return df


def plot_data(df: pl.DataFrame):
    sns.lineplot(data=df.to_pandas(), x="datetime", y="price")
    plt.xlabel("Datetime")
    plt.ylabel("Price (€/MWh)")
    plt.show()


def sanitize_exog_cols(df_pd: pd.DataFrame, exog_cols: list[str]) -> list[str]:
    """
    Remove missing / all-NaN / constant exog columns to reduce singularities and crashes.
    """
    keep = []
    for c in exog_cols:
        if c not in df_pd.columns:
            continue
        s = df_pd[c]
        if s.isna().all():
            continue
        # treat constant (including all zeros/ones) as useless for estimation
        if s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return keep


def analyse(exog_cols: list[str], df_pd: pd.DataFrame) -> dict:
    """
    Fit a SARIMAX for a given set of exogenous columns and return lightweight metrics.

    IMPORTANT:
    - Do not return the model object from joblib workers (memory-heavy).
    - Always pass exog_cols into SarimaxModel so the grid search is real.
    """
    # exog_cols = sanitize_exog_cols(df_pd, exog_cols)

    try:
        print(f"Fitting SARIMAX with exog: {exog_cols}")
        mod = SarimaxModel(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 24),
            train_frac=1,
            exog_cols=exog_cols,
        ).fit(df_pd)

        res = getattr(mod, "fitted_model", None)
        if res is None:
            # fallback: if your class stores results under a different attribute
            res = getattr(mod, "results", None)

        if res is None:
            raise RuntimeError("Could not find fitted results on model (expected .fitted_model or .results).")
        
        mod.summary()  # for logging purposes

        return {
            "ok": True,
            "exog_cols": exog_cols,
            "aic": float(res.aic),
            "bic": float(res.bic),
            "hqic": float(res.hqic),
            "llf": float(res.llf),
        }

    except Exception as e:
        # Return failure as a record (do not crash the entire Parallel call)
        return {
            "ok": False,
            "exog_cols": exog_cols,
            "error": repr(e),
        }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    logger = set_logger()
    logger.info("Starting Energy Price Analyser")

    # Load 2018 dataset
    df_2018_pl = get_dataset(2024)
    logger.info("Dataset 2018 loaded.")

    # Convert once to pandas to avoid re-serializing polars objects in joblib workers
    df_2018 = df_2018_pl.to_pandas()
    df_2018["datetime"] = pd.to_datetime(df_2018["datetime"])

    # Optional: basic sanity check for nulls
    null_rows = df_2018_pl.filter(pl.any_horizontal(pl.all().is_null()))
    if len(null_rows) > 0:
        logger.warning("Found rows with nulls in 2018 dataset (showing polars output):")
        print(null_rows)

    # -----------------------------
    # Grid search over EXOG_SETS
    # -----------------------------
    logger.info("Running exogenous grid search (lightweight scoring)...")

    # SARIMAX is RAM-bound; using all cores is a common OOM cause.
    n_jobs = -1

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(analyse)(exog_cols, df_2018) for exog_cols in EXOG_SETS
    )

    df_res = pd.DataFrame(results)

    # Show failures (if any)
    failures = df_res[~df_res["ok"]]
    if not failures.empty:
        logger.warning("Some configurations failed. Showing first 10 errors:")
        print(failures[["exog_cols", "error"]].head(10))

    # Keep only successful fits
    df_ok = df_res[df_res["ok"]].copy()
    if df_ok.empty:
        raise RuntimeError("All exogenous configurations failed. Check exog columns and model settings.")

    df_ok = df_ok.sort_values("aic", ascending=True).reset_index(drop=True)

    logger.info("Top 10 configurations by AIC:")
    print(df_ok[["aic", "bic", "hqic", "llf", "exog_cols"]].head(10).to_string(index=False))

    # Pick top-K to refit for forecasting (do NOT forecast every config)
    TOP_K = 3
    top_exog_sets = df_ok["exog_cols"].head(TOP_K).tolist()

    # -----------------------------
    # Forecast on 2019 for top-K
    # -----------------------------
    df_2019 = get_dataset(2025).to_pandas()
    df_2019["datetime"] = pd.to_datetime(df_2019["datetime"])
    steps = len(df_2019)

    logger.info(f"Forecasting {steps} hours (2019) for top {TOP_K} configs...")

    for rank, exog_cols in enumerate(top_exog_sets, start=1):
        logger.info(f"[{rank}/{TOP_K}] Fitting final model with exog: {exog_cols}")

        model = SarimaxModel(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 24),
            train_frac=1,
            exog_cols=exog_cols,
        ).fit(df_2018)

        # Forecast
        y_hat, conf = model.forecast(steps=steps)

        df_fc = pd.DataFrame(
            {
                "y_true": df_2019["price"].to_numpy(),
                "y_hat": np.asarray(y_hat),
                "lower": conf.iloc[:, 0].to_numpy(),
                "upper": conf.iloc[:, 1].to_numpy(),
            },
            index=pd.to_datetime(df_2019["datetime"]),
        )
        df_fc.index = pd.to_datetime(df_fc.index)

        df_fc["error"] = df_fc["y_hat"] - df_fc["y_true"]
        df_fc["abs_error"] = df_fc["error"].abs()
        df_fc["month"] = df_fc.index.month
        df_fc["hour"] = df_fc.index.hour

        # Plot forecast error over time
        plt.figure(figsize=(14, 5))
        sns.lineplot(x=df_fc.index, y=df_fc["error"], linewidth=0.8)
        plt.axhline(0, linestyle="--")
        plt.title(f"Forecast Error (Prediction − Actual) – 2019 | Rank {rank} | exog={exog_cols}")
        plt.ylabel("€ / MWh")
        plt.xlabel("Time")
        plt.tight_layout()
        plt.show()

        # Categorize forecast quality
        conditions = [
            df_fc["abs_error"] < 1.0,
            (df_fc["y_true"] >= df_fc["lower"]) & (df_fc["y_true"] <= df_fc["upper"]),
        ]
        choices = ["Centered |err| < 1€", "In CI (not centered)"]
        df_fc["category"] = np.select(conditions, choices, default="Out of CI")

        monthly_counts = (
            df_fc.groupby(["month", "category"])
            .size()
            .reset_index(name="count")
        )

        plt.figure(figsize=(14, 6))
        sns.barplot(data=monthly_counts, x="month", y="count", hue="category")
        plt.xlabel("Month")
        plt.ylabel("Number of hourly forecasts")
        plt.title(f"Monthly Forecast Quality – 2019 | Rank {rank}")
        plt.legend(title="Forecast category")
        plt.tight_layout()
        plt.show()

        # Print next 24 hours forecast from the end of training
        print(f"\n=== Next 24 hours forecast | Rank {rank} | exog={exog_cols} ===")
        for dt, y, (low, high) in zip(
            pd.date_range(start=model.last_datetime + pd.Timedelta(hours=1), periods=24, freq="h"),
            y_hat[:24],
            conf.values[:24],
        ):
            print(f"{dt}: {y:.2f} €/MWh (CI: {low:.2f} - {high:.2f})")

    logger.info("Done.")


if __name__ == "__main__":
    main()
