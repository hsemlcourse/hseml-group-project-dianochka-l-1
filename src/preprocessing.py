"""
Предобработка сырых вакансий с hh.ru в плоский DataFrame.

Шаги:
1. Плоское извлечение полей из вложенного JSON (salary, area, employer, ...)
2. Приведение зарплаты к рублям и к виду net (грубый учёт НДФЛ)
3. Базовая очистка текста (описание приходит с HTML-тегами)
4. Сохранение в data/processed/vacancies.parquet
"""

from __future__ import annotations

import html
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Курсы примерные. На CP2 заменить на исторические курсы ЦБ.
CURRENCY_TO_RUB = {
    "RUR": 1.0,
    "RUB": 1.0,
    "USD": 90.0,
    "EUR": 100.0,
    "KZT": 0.2,
    "BYR": 28.0,
    "UAH": 2.5,
}

# Грубый коэффициент gross -> net (НДФЛ 13%).
GROSS_TO_NET = 0.87

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Границы адекватных значений зарплаты (в рублях/месяц).
SALARY_MIN_RUB = 10_000
SALARY_MAX_RUB = 5_000_000


def load_raw_vacancies(raw_path: Path) -> list[dict[str, Any]]:
    """Загружает сырые вакансии.

    Сначала пробует основной файл (обычно vacancies_all.json).
    Если его нет, он пустой или битый — собирает данные из батчей
    vacancies_batch_*.json в той же папке. Это нужно, когда парсинг упал
    до финального сохранения, но успел записать промежуточные батчи.
    """
    data: list[dict[str, Any]] = []
    if raw_path.exists() and raw_path.stat().st_size > 0:
        try:
            with raw_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                data = loaded
        except json.JSONDecodeError as exc:
            logger.warning("Не смогла прочитать %s: %s. Попробую батчи", raw_path, exc)

    if not data:
        logger.info("Основной файл пустой или отсутствует — собираю данные из батчей")
        data = _load_from_batches(raw_path.parent)

    if not isinstance(data, list):
        raise ValueError(f"Ожидался список вакансий, получили {type(data).__name__}")

    logger.info("Загружено %d сырых вакансий", len(data))
    return data


def _load_from_batches(raw_dir: Path) -> list[dict[str, Any]]:
    """Читает все vacancies_batch_*.json из папки и склеивает в один список,
    убирая дубликаты по id."""
    if not raw_dir.exists():
        return []
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    batch_files = sorted(raw_dir.glob("vacancies_batch_*.json"))
    logger.info("Найдено батчей: %d", len(batch_files))
    for path in batch_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                items = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Битый батч %s, пропускаю: %s", path, exc)
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            vid = item.get("id")
            if vid and vid not in seen:
                seen.add(vid)
                result.append(item)
    return result


def _clean_html(text: str | None) -> str:
    """Удаляет HTML-теги, разворачивает entities и схлопывает пробелы."""
    if not text:
        return ""
    unescaped = html.unescape(str(text))
    no_tags = HTML_TAG_PATTERN.sub(" ", unescaped)
    return WHITESPACE_PATTERN.sub(" ", no_tags).strip()


def _normalize_salary(
    salary: dict[str, Any] | None,
) -> tuple[float | None, float | None, float | None]:
    """Приводит зарплату к рублям на руки.

    Returns:
        (salary_from_rub, salary_to_rub, salary_mid_rub).
        None — если зарплата не указана или валюта неизвестна.
    """
    if not salary:
        return None, None, None

    currency = salary.get("currency") or "RUR"
    rate = CURRENCY_TO_RUB.get(currency)
    if rate is None:
        logger.warning("Неизвестная валюта: %s", currency)
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
    elif s_from is not None:
        s_mid = s_from
    else:
        s_mid = s_to

    return s_from, s_to, s_mid


def _extract_row(vacancy: dict[str, Any]) -> dict[str, Any]:
    """Превращает одну вакансию (вложенный JSON) в плоский dict."""
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
    """Строит DataFrame из списка сырых вакансий и снимает дубли по id."""
    rows = [_extract_row(v) for v in vacancies]
    df = pd.DataFrame(rows)

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)

    before = len(df)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    logger.info("Дубликатов по id удалено: %d", before - len(df))

    return df


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Зануляет явные выбросы в зарплатных колонках.

    Строки не удаляем — текст и навыки остаются полезными для анализа.
    Только сам таргет (salary_*_rub) превращаем в NaN, если значение явно битое.
    """
    result = df.copy()
    salary_cols = ["salary_from_rub", "salary_to_rub", "salary_mid_rub"]

    for col in salary_cols:
        if col not in result.columns:
            continue
        mask_bad = (result[col] < SALARY_MIN_RUB) | (result[col] > SALARY_MAX_RUB)
        n_bad = int(mask_bad.sum())
        if n_bad:
            logger.info("%s: занулено %d выбросов", col, n_bad)
        result.loc[mask_bad, col] = pd.NA

    return result


def run_preprocessing(raw_path: Path, processed_dir: Path) -> Path:
    """Полный цикл обработки: JSON → DataFrame → parquet."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_vacancies(raw_path)
    if not raw:
        raise ValueError("В файле нет ни одной вакансии — нечего обрабатывать")

    df = build_dataframe(raw)
    df = clean_outliers(df)

    n_total = len(df)
    n_with_salary = int(df["salary_mid_rub"].notna().sum()) if "salary_mid_rub" in df else 0
    logger.info(
        "Итого %d вакансий, из них с корректной зарплатой %d (%.1f%%)",
        n_total, n_with_salary, 100 * n_with_salary / max(n_total, 1),
    )

    output_path = processed_dir / "vacancies.parquet"
    df.to_parquet(output_path, index=False)
    logger.info("Обработанные данные сохранены: %s", output_path)

    return output_path