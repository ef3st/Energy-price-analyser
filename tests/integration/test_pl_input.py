import pytest

@pytest.mark.integration
def test_pl_import(semester_csv_path):
    import polars as pl
    df = pl.read_csv(semester_csv_path, separator=';')
    print(df)
    assert df.shape[1] == 9
    assert df.shape[0] > 0
    assert list(df.columns) == ['',"€/MWh/2018", "€/MWh/2019", "€/MWh/2020", "€/MWh/2021", "€/MWh/2022", "€/MWh/2023", "€/MWh/2024", "€/MWh/2025"]