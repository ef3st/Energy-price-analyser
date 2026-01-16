import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from models.sarimax import SarimaxModel

plt.show()


def splitter(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    year_cols = df.select(pl.col(r"^€/MWh/\d{4}$")).columns

    dfs_by_year: dict[str, pl.DataFrame] = {}

    for col in year_cols:
        year = col.split("/")[-1]

        dt_str = pl.col("").cast(pl.Utf8).str.strip_chars().str.replace("_", " ")

        # costruisci "YYYY/<dd>/<mm> <HH>:00"
        full = pl.concat_str([pl.lit(f"{year}/"), dt_str, pl.lit(":00")])

        # parse "normale" (00-23)
        dt_normal = full.str.strptime(
            pl.Datetime, format="%Y/%d/%m %H:%M", strict=False
        )

        # gestisci "24:00" => "00:00" + 1 day
        dt_24 = full.str.replace(r" 24:00$", " 00:00").str.strptime(
            pl.Datetime, format="%Y/%d/%m %H:%M", strict=False
        ) + pl.duration(days=1)

        out = (
            df.select(["", col])
            .rename({col: "price"})
            .with_columns(
                pl.when(dt_str.is_null() | (dt_str == ""))
                .then(None)
                .when(full.str.contains(r" 24:00$"))
                .then(dt_24)
                .otherwise(dt_normal)
                .alias("datetime")
            )
        )
        print("----")
        print(out.filter(pl.any_horizontal(pl.all().is_null())))
        print("----")
        dfs_by_year[year] = out
        out.write_csv(f"{year}.csv")

    return dfs_by_year


def merge_with_year(dfs_by_year: dict[str, pl.DataFrame]) -> pl.DataFrame:
    dfs = []

    for year, df in dfs_by_year.items():
        df = (
            df.with_columns(pl.lit(int(year)).alias("year"))
            .group_by("day")
            .agg(pl.col("price").mean())
        )
        dfs.append(df)

    return pl.concat(dfs)


def main():

    df = pl.read_csv("data/row_2018.csv", separator=";")
    df = df.with_columns(pl.col("€/MWh").str.replace_all(",", ".").cast(pl.Float64))
    df = df.rename({"€/MWh": "price"})

    tz = "Europe/Rome"

    df = df.with_columns(
        pl.col("Data").str.strptime(pl.Date, "%d/%m/%Y").alias("date")
    ).with_columns(
        (
            # 00:00 del giorno, interpretato come ora locale (Europe/Rome)
            pl.col("date").cast(pl.Datetime).dt.replace_time_zone(tz)
            +
            # Ora è 1->01:00, 24->00:00 del giorno dopo, 25->01:00 del giorno dopo, ecc.
            pl.duration(hours=pl.col("Ora"))
        ).alias("datetime")
    )
    print(df.filter(pl.any_horizontal(pl.all().is_null())))
    sns.lineplot(
            data=df.to_pandas(), x="datetime", y="price",
        )
    plt.xlabel("Datetime")
    plt.ylabel("Price (€/MWh)")
    plt.show()
    mod = SarimaxModel((1, 1, 1), (1, 1, 1, 24))
    mod.fit(df)
    print(mod.summary())
    y_hat, conf = mod.forecast(steps=24)
    print("Forecast for next 24 hours:")
    import pandas as pd
    for dt, y, (low, high) in zip(
        pd.date_range(
            start=mod.last_datetime + pd.Timedelta(hours=1), periods=24, freq="H"
        ),
        y_hat,
        conf.values,
    ):
        print(f"{dt}: {y:.2f} €/MWh (CI: {low:.2f} - {high:.2f})")
    # df = pl.read_csv("data/demo.csv", separator=";")
    # dfs = splitter(df)
    # print(dfs)
    # fig, axes = plt.subplots(8, 1, figsize=(12, 4))
    # mod: list[SarimaxModel] = []
    # for year, data in dfs.items():
    #     # data = data.group_by("day").agg(
    #     #     pl.col("price").mean()
    #     # )
    #     mod.append(SarimaxModel((1, 1, 1), (1, 1, 1, 24)))
    #     mod[-1].fit(data)
    #     sns.lineplot(
    #         data=data.to_pandas(), x="day", y="price", ax=axes[int(year) - 2018]
    #     )
    #     mod[-1].summary()
    #     axes[int(year) - 2018].set_xlabel("Hour of Day")
    #     axes[int(year) - 2018].set_ylabel("Price (€/MWh)")
    # plt.tight_layout()
    # plt.show()
    # sns.lineplot(data=merge_with_year(dfs).to_pandas(), x="", y="price",)
    # plt.xlabel("Hour of Day")
    # plt.ylabel("Price (€/MWh)")
    # plt.show()


if __name__ == "__main__":
    main()
