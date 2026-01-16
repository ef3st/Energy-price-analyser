import pytest
import polars as pl
from pathlib import Path


@pytest.fixture(scope="session")
def one_year_csv_path() -> Path:
    return Path(__file__).parent / "data" / "one_year_2018.csv"


@pytest.fixture(scope="session")
def semester_csv_path() -> Path:
    return Path(__file__).parent / "data" / "I_semester_2018-2025.csv"
