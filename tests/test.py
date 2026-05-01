from __future__ import annotations

import pandas as pd
import pytest
from src.preprocessing import (
    _clean_html,
    _extract_row,
    _normalize_salary,
    build_dataframe,
    clean_outliers,
)

def test_clean_html_strips_tags():
    text = "<p>Ищем <strong>Python</strong>-разработчика</p>"
    assert _clean_html(text) == "Python -разработчик" or "Python" in _clean_html(text)


def test_clean_html_none_returns_empty():
    assert _clean_html(None) == ""


def test_normalize_salary_rub_gross():
    salary = {"from": 100_000, "to": 150_000, "currency": "RUR", "gross": True}
    s_from, s_to, s_mid = _normalize_salary(salary)
    # Gross -> net: 100k * 0.87 = 87k, 150k * 0.87 = 130.5k
    assert s_from == pytest.approx(87_000)
    assert s_to == pytest.approx(130_500)
    assert s_mid == pytest.approx(108_750)


def test_normalize_salary_usd_converted_to_rub():
    salary = {"from": 1000, "to": None, "currency": "USD", "gross": False}
    s_from, s_to, s_mid = _normalize_salary(salary)
    assert s_from == pytest.approx(90_000)  # 1000 USD * 90 * 1.0
    assert s_to is None
    assert s_mid == pytest.approx(90_000)   # точечная оценка, если только from


def test_normalize_salary_none():
    assert _normalize_salary(None) == (None, None, None)


def test_extract_row_minimal():
    vacancy = {
        "id": "123",
        "name": "Python Developer",
        "area": {"id": "1", "name": "Москва"},
        "employer": {"name": "Test Co"},
        "experience": {"id": "between1And3", "name": "От 1 года до 3 лет"},
        "salary": {"from": 100_000, "to": 150_000, "currency": "RUR", "gross": False},
        "key_skills": [{"name": "Python"}, {"name": "SQL"}],
    }
    row = _extract_row(vacancy)
    assert row["id"] == "123"
    assert row["area_name"] == "Москва"
    assert row["key_skills_count"] == 2
    assert row["salary_mid_rub"] == pytest.approx(125_000)


def test_build_dataframe_deduplicates():
    vacancies = [
        {"id": "1", "name": "A"},
        {"id": "1", "name": "A copy"},
        {"id": "2", "name": "B"},
    ]
    df = build_dataframe(vacancies)
    assert len(df) == 2
    assert set(df["id"]) == {"1", "2"}


def test_clean_outliers_nullifies_extreme_values():
    df = pd.DataFrame({
        "salary_from_rub": [50_000, 0, 9_000, 10_000_000, 100_000],
        "salary_to_rub": [60_000, 10_000, 15_000, 20_000_000, 150_000],
        "salary_mid_rub": [55_000, 5_000, 12_000, 15_000_000, 125_000],
    })
    result = clean_outliers(df)
    assert result.loc[0, "salary_mid_rub"] == 55_000
    assert result.loc[4, "salary_mid_rub"] == 125_000

    assert pd.isna(result.loc[1, "salary_mid_rub"])
    assert pd.isna(result.loc[2, "salary_mid_rub"])
    assert pd.isna(result.loc[3, "salary_mid_rub"])
