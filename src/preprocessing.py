"""
Предобработка сырых вакансий с hh.ru в плоский DataFrame
1. Плоское извлечение полей из вложенного JSON (salary, area, employer, ...)
2. Приведение зарплаты к единой валюте (RUB) и единому виду (net, среднее по диапазону)
3. Базовая очистка текста (описание вакансии приходит с HTML-тегами)
4. Сохранение в data/processed/vacancies.parquet для дальнейшей работы
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)

# Примерные курсы к рублю... В боевом проекте стоит брать исторические курсы ЦБ
CURRENCY_TO_RUB = {
    "RUR": 1.0,
    "RUB": 1.0,
    "USD": 90.0,
    "EUR": 100.0,
    "KZT": 0.2,
    "BYR": 28.0,
    "UAH": 2.5,
}

# Коэффициент для перевода gross -> net (НДФЛ 13%)
GROSS_TO_NET = 0.87

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def load_raw_vacancies(raw_path: Path) -> list[dict[str, Any]]:
    """Загружает сырые вакансии из JSON-файла"""
    with raw_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Загружено %d сырых вакансий из %s", len(data), raw_path)
    return data


def _clean_html(text: str | None) -> str:
    """Удаляет HTML-теги и схлопывает whitespace в описании вакансии"""
    if not text:
        return ""
    no_tags = HTML_TAG_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", no_tags).strip()


def _normalize_salary(
    salary: dict[str, Any] | None,
) -> tuple[float | None, float | None, float | None]:
    """Приводит зарплату к рублям на руки
    Returns:
        (salary_from_rub, salary_to_rub, salary_mid_rub) — None если не указана

    - Переводим в рубли по курсу (константный, см. CURRENCY_TO_RUB)
    - Если gross=True, домножаем на 0.87 (грубый учёт НДФЛ)
    - salary_mid — среднее арифметическое from и to (если указаны оба)
    """
    if not salary:
        return None, None, None

    currency = salary.get("currency") or "RUR"
    rate = CURRENCY_TO_RUB.get(currency)
    if rate is None:
        logger.warning("Неизвестная валюта %s", currency)
        return None, None, None

    gross = salary.get("gross", True)
    tax_coef = GROSS_TO_NET if gross else 1.0

    def _convert(value: float | None) -> float | None:
        if value is None:
            return None
        return float(value) * rate * tax_coef

    s_from = _convert(salary.get("from"))
    s_to = _convert(salary.get("to"))

    if s_from is not None and s_to is not None:
        s_mid = (s_from + s_to) / 2
    else:
        s_mid = s_from if s_from is not None else s_to

    return s_from, s_to, s_mid


def _extract_row(vacancy: dict[str, Any]) -> dict[str, Any]:
    """Преобразует одну вакансию (вложенный JSON) в плоский dict для DataFrame."""
    salary_from, salary_to, salary_mid = _normalize_salary(vacancy.get("salary"))

    area = vacancy.get("area") or {}
    employer = vacancy.get("employer") or {}
    experience = vacancy.get("experience") or {}
    employment = vacancy.get("employment") or {}
    schedule = vacancy.get("schedule") or {}

    key_skills = vacancy.get("key_skills") or []
    skills_list = [s.get("name", "") for s in key_skills if isinstance(s, dict)]

    prof_roles = vacancy.get("professional_roles") or []
    role_names = [r.get("name", "") for r in prof_roles if isinstance(r, dict)]
    role_ids = [r.get("id", "") for r in prof_roles if isinstance(r, dict)]

    return {
        "id": vacancy.get("id"),
        "name": vacancy.get("name", ""),
        "area_id": area.get("id"),
        "area_name": area.get("name"),
        "employer_name": employer.get("name"),
        "employer_id": employer.get("id"),
        "employer_trusted": employer.get("trusted"),
        "experience": experience.get("id"),
        "experience_name": experience.get("name"),
        "employment": employment.get("id"),
        "schedule": schedule.get("id"),
        "professional_role_ids": role_ids,
        "professional_role_names": role_names,
        "has_test": bool(vacancy.get("has_test", False)),
        "response_letter_required": bool(vacancy.get("response_letter_required", False)),
        "description": _clean_html(vacancy.get("description")),
        "key_skills": skills_list,
        "key_skills_count": len(skills_list),
        "published_at": vacancy.get("published_at"),
        "salary_from_rub": salary_from,
        "salary_to_rub": salary_to,
        "salary_mid_rub": salary_mid,
    }


def build_dataframe(vacancies: list[dict[str, Any]]) -> pd.DataFrame:
    """Строит DataFrame из списка сырых вакансий."""
    rows = [_extract_row(v) for v in vacancies]
    df = pd.DataFrame(rows)

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    before = len(df)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    logger.info("Дубликатов по id удалено %d", before - len(df))

    return df


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Отсекает явные выбросы и мусор в зарплате

    Критерии:
    - Нулевые и отрицательные значения -> NaN (очевидные ошибки ввода)
    - Значения <10_000 RUB/мес -> NaN (скорее всего, указали почасовую ставку
      или зарплату в тыс. рублей)
    - Значения >5_000_000 RUB/мес -> NaN (реально редкая история, обычно ошибка)

    Выбросы не удаляем строки, а зануляем таргет — сами признаки (навыки, описание)
    остаются полезными для анализа рынка
    """
    result = df.copy()
    salary_cols = ["salary_from_rub", "salary_to_rub", "salary_mid_rub"]

    for col in salary_cols:
        if col not in result.columns:
            continue
        mask_bad = (
            (result[col] <= 0)
            | (result[col] < 10_000)
            | (result[col] > 5_000_000)
        )
        n_bad = int(mask_bad.sum())
        if n_bad:
            logger.info("Колонка %s занулено %d выбросов", col, n_bad)
        result.loc[mask_bad, col] = pd.NA

    return result


def run_preprocessing(raw_path: Path, processed_dir: Path) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_vacancies(raw_path)
    if not raw:
        logger.error(
            "Парсер не собрал ни одной вакансии..."
            "проверить логи парсера"
        )
        raise ValueError("No raw vacancies to preprocess")

    df = build_dataframe(raw)
    df = clean_outliers(df)

    n_total = len(df)
    if "salary_mid_rub" in df.columns:
        n_with_salary = int(df["salary_mid_rub"].notna().sum())
    else:
        n_with_salary = 0
    logger.info(
        "Итого %d вакансий, из них с корректной зарплатой %d (%.1f%%)",
        n_total, n_with_salary, 100 * n_with_salary / max(n_total, 1),
    )

    output_path = processed_dir / "vacancies.parquet"
    df.to_parquet(output_path, index=False)
    logger.info("Обработанные данные сохранены %s", output_path)

    return output_path