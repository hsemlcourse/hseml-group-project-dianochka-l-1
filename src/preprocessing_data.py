"""
Предобработка готового датасета вакансий .csv
"""

from __future__ import annotations

import ast
import html
import json
import logging
import re
from pathlib import Path
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)

# Регулярки для очистки описаний
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")

SALARY_MIN_RUB = 10_000
SALARY_MAX_RUB = 5_000_000

def _clean_html(text: Any) -> str:
    """Убирает HTML-теги, разворачивает entities и нормализует пробелы"""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    unescaped = html.unescape(str(text))
    no_tags = HTML_TAG_PATTERN.sub(" ", unescaped)
    return WHITESPACE_PATTERN.sub(" ", no_tags).strip()


def _parse_list_field(value: Any) -> list[str]:
    """Парсит строку вида "['Python', 'SQL']" в список... Пустой список на любой сбой"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    try:
        parsed = ast.literal_eval(str(value))
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _parse_json_field(value: Any) -> dict[str, Any]:
    """Парсит JSON-строку (requirements, benefits, source) в dict. Пустой dict на сбой"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return value
    s = str(value)

    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(s)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}

def load_raw_csv(raw_path: Path) -> pd.DataFrame:
    """Читает исходный CSV Mendeley... Сбрасывает мусорную колонку индекса"""
    logger.info("Читаю CSV %s", raw_path)
    df = pd.read_csv(raw_path, sep=";", low_memory=False)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    logger.info("Загружено %d сырых строк", len(df))
    return df


def filter_usable_rows(
    df: pd.DataFrame,
    sources: list[str] | None = None,
    countries: list[str] | None = None,
) -> pd.DataFrame:
    """Оставляет строки, подходящие для обучения модели
    Критерии:
    - is_open == True
    - name не пустой
    - salary > 0
    - data_source принадлежит sources (если sources задан)
    - country_name принадлежит countries (если countries задан)
    """
    n_before = len(df)

    mask_open = df["is_open"] == True  # noqa: E712
    mask_has_name = df["name"].notna()
    mask_has_salary = df["salary"].fillna(0) > 0
    mask = mask_open & mask_has_name & mask_has_salary

    if sources:
        mask_source = df["data_source"].isin(sources)
        mask = mask & mask_source
        logger.info("Фильтр по источникам %s %d подходящих строк", sources, int(mask_source.sum()))

    if countries:
        mask_country = df["country_name"].isin(countries)
        mask = mask & mask_country
        logger.info("Фильтр по странам %s %d подходящих строк", countries, int(mask_country.sum()))

    result = df[mask].copy()
    logger.info(
        "Фильтрация usable %d -> %d строк "
        "(открытых=%d, с именем=%d, с зарплатой=%d)",
        n_before, len(result),
        int(mask_open.sum()), int(mask_has_name.sum()), int(mask_has_salary.sum()),
    )
    return result


def clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Парсит JSON-поля, чистит текст, добавляет производные признаки"""
    result = df.copy()

    # Описание
    result["description"] = result["description"].apply(_clean_html)
    result["description_length"] = result["description"].str.len()

    # Навыки
    result["skills"] = result["raw_skills"].apply(_parse_list_field)
    result["skills_count"] = result["skills"].apply(len)

    result["languages_parsed"] = result["languages"].apply(_parse_list_field)
    result["languages_count"] = result["languages_parsed"].apply(len)

    requirements_parsed = result["requirements"].apply(_parse_json_field)
    result["has_test"] = requirements_parsed.apply(lambda d: bool(d.get("test", False)))

    benefits_parsed = result["benefits"].apply(_parse_json_field)
    result["is_premium"] = benefits_parsed.apply(lambda d: bool(d.get("premium", False)))

    result["last_found_at"] = pd.to_datetime(result["last_found_at"], errors="coerce")
    result["year"] = result["last_found_at"].dt.year
    result["month"] = result["last_found_at"].dt.month

    result["salary_rub"] = result["salary"].astype(float)
    result["has_salary_from"] = result["salary_from"] > 0
    result["has_salary_to"] = result["salary_to"] > 0

    # Убираем исходные строковые версии — они уже распарсены
    to_drop = ["raw_skills", "requirements", "benefits", "languages", "source"]
    result = result.drop(columns=[c for c in to_drop if c in result.columns])

    return result


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляем строки целиком... если таргет явно битый, обучать на ней нечему
    """
    n_before = len(df)
    mask = (df["salary_rub"] >= SALARY_MIN_RUB) & (df["salary_rub"] <= SALARY_MAX_RUB)
    result = df[mask].copy().reset_index(drop=True)
    logger.info(
        "Удалено выбросов по зарплате %d (осталось %d). Границы %d ... %d ₽",
        n_before - len(result), len(result), SALARY_MIN_RUB, SALARY_MAX_RUB,
    )
    return result


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    result = df.drop_duplicates(subset=["id", "data_source"], keep="first").reset_index(drop=True)
    logger.info("Удалено дубликатов по (id, data_source) %d", n_before - len(result))
    return result

def run_preprocessing(
    raw_path: Path,
    processed_dir: Path,
    sources: list[str] | None = None,
    countries: list[str] | None = None,
    save_filtered_csv: bool = True,
) -> Path:
    """
    Args:
        raw_path: путь к CSV
        processed_dir: куда сложить parquet
        sources: список data_source (напр. ["hh"]). None = без фильтра
        countries: список country_name (напр. ["Россия"]). None = без фильтра
        save_filtered_csv: сохранять ли промежуточный CSV после фильтрации

    Returns:
        Путь к финальному parquet

    Raises:
        ValueError: если после фильтрации не осталось ни одной строки
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_csv(raw_path)
    df = filter_usable_rows(df, sources=sources, countries=countries)

    if df.empty:
        raise ValueError(
            "После фильтрации не осталось строк"
        )
    if save_filtered_csv:
        filter_tag = _make_filter_tag(sources, countries)
        filtered_path = raw_path.parent / f"vacancies_filtered_{filter_tag}.csv"
        df.to_csv(filtered_path, index=False)
        logger.info("Отфильтрованный CSV сохранён %s", filtered_path)

    df = clean_and_enrich(df)
    df = filter_outliers(df)
    df = deduplicate(df)

    _log_summary(df)

    output_path = processed_dir / "vacancies.parquet"
    df.to_parquet(output_path, index=False)
    logger.info("Parquet сохранён: %s", output_path)
    return output_path


def _make_filter_tag(sources: list[str] | None, countries: list[str] | None) -> str:
    parts: list[str] = []
    if sources:
        parts.extend(sources)
    if countries:
        parts.extend(_simple_slug(c) for c in countries)
    return "_".join(parts) if parts else "all"


def _simple_slug(text: str) -> str:
    """Грубая транслитерация кириллицы для имени файла"""
    mapping = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh",
        "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
        "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
        "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ы": "y",
        "э": "e", "ю": "yu", "я": "ya", "ь": "", "ъ": "", "ё": "e",
    }
    out = []
    for ch in text.lower():
        out.append(mapping.get(ch, ch if ch.isalnum() else ""))
    return "".join(out) or "x"


def _log_summary(df: pd.DataFrame) -> None:
    logger.info("Итоговый датасет %d строк", len(df))
    if not df.empty:
        logger.info(
            "Зарплата медиана=%.0f ₽, средняя=%.0f ₽, min=%.0f ₽, max=%.0f ₽",
            df["salary_rub"].median(),
            df["salary_rub"].mean(),
            df["salary_rub"].min(),
            df["salary_rub"].max(),
        )