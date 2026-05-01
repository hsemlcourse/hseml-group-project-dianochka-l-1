"""
Парсер вакансий с hh.ru через официальный API
Документация API: https://api.hh.ru/openapi/redoc
Справочник ролей: https://api.hh.ru/professional_roles
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)

HH_API_BASE = "https://api.hh.ru"
HH_VACANCIES_ENDPOINT = f"{HH_API_BASE}/vacancies"
HH_ROLES_ENDPOINT = f"{HH_API_BASE}/professional_roles"

# Лимиты API
MAX_PER_PAGE = 100
MAX_PAGES = 20
MAX_RESULTS_PER_QUERY = MAX_PER_PAGE * MAX_PAGES  # 2000

REQUEST_TIMEOUT = 15
REQUEST_DELAY = 0.8  # пауза между запросами

TARGET_CATEGORIES = [
    "Информационные технологии",
    "Маркетинг, реклама, PR",
    "Продажи, обслуживание клиентов",
    "Финансы, бухгалтерия",
    "Управление персоналом, тренинги",
    "Высший и средний менеджмент",
    "Юристы",
    "Медицина, фармацевтика",
    "Наука, образование",
    "Искусство, развлечения, массмедиа",
    "Строительство, недвижимость",
    "Производство, сервисное обслуживание",
    "Транспорт, логистика, перевозки",
    "Закупки",
    "Туризм, гостиницы, рестораны",
    "Розничная торговля",
    "Рабочий персонал",
    "Автомобильный бизнес",
    "Сельское хозяйство",
    "Безопасность",
    "Спортивные клубы, фитнес, салоны красоты",
    "Домашний, обслуживающий персонал",
]


@dataclass
class ParserConfig:
    """Конфигурация парсера

    Attributes:
        user_agent: обязательный заголовок с контактом разработчика
        area: ID региона hh.ru (113 = Россия, 1 = Москва, 2 = СПб)
        per_category_limit: сколько вакансий собираем из каждой категории
        only_with_salary: брать только вакансии с указанной зарплатой
            True оставляет больше полезных строк для задачи регрессии
        raw_dir: куда сохраняем сырые JSON-ответы
    """

    user_agent: str
    area: int = 113
    per_category_limit: int = 500
    only_with_salary: bool = True
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))


def _build_session(user_agent: str, access_token: str | None = None) -> requests.Session:
    session = requests.Session()
    headers = {
        "User-Agent": user_agent,
        "HH-User-Agent": user_agent,
        "Accept": "application/json",
    }
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    session.headers.update(headers)

    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _fetch_page(
    session: requests.Session,
    params: dict[str, Any],
) -> dict[str, Any]:
    response = session.get(
        HH_VACANCIES_ENDPOINT,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == 403:
        logger.error(
            "403 Forbidden от hh.ru... %s",
            response.text[:500],
        )
    response.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return response.json()


def fetch_professional_roles(session: requests.Session) -> dict[str, list[str]]:
    """Загружает справочник профессиональных ролей hh.ru
    Returns:
        Словарь {название_категории: [role_id, role_id, ...]}
        Категория = крупная отрасль,
        внутри неё — конкретные роли
    """
    response = session.get(HH_ROLES_ENDPOINT, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    categories: dict[str, list[str]] = {}
    for category in data.get("categories", []):
        name = category.get("name", "")
        role_ids = [role["id"] for role in category.get("roles", []) if "id" in role]
        if name and role_ids:
            categories[name] = role_ids
    return categories


def _iter_query(
    session: requests.Session,
    base_params: dict[str, Any],
    limit: int,
) -> Iterator[dict[str, Any]]:
    collected = 0
    first_page = _fetch_page(
        session, {**base_params, "page": 0, "per_page": MAX_PER_PAGE}
    )
    total_found = first_page.get("found", 0)
    total_pages = min(first_page.get("pages", 0), MAX_PAGES)

    logger.info(
        "Запрос %s найдено %d вакансий, страниц для обхода %d (лимит=%d)",
        base_params, total_found, total_pages, limit,
    )

    for item in first_page.get("items", []):
        if collected >= limit:
            return
        yield item
        collected += 1

    for page in range(1, total_pages):
        if collected >= limit:
            return
        data = _fetch_page(
            session, {**base_params, "page": page, "per_page": MAX_PER_PAGE}
        )
        for item in data.get("items", []):
            if collected >= limit:
                return
            yield item
            collected += 1


def fetch_balanced_vacancy_ids(config: ParserConfig) -> list[str]:
    session = _build_session(config.user_agent, config.access_token)
    all_categories = fetch_professional_roles(session)

    ids: list[str] = []
    seen: set[str] = set()
    consecutive_failures = 0

    for category_name in TARGET_CATEGORIES:
        role_ids = all_categories.get(category_name)
        if not role_ids:
            logger.warning("Категория %r не найдена в справочнике", category_name)
            continue

        base_params: dict[str, Any] = {
            "area": config.area,
            "professional_role": role_ids,
            "only_with_salary": config.only_with_salary,
        }

        category_ids: list[str] = []
        try:
            for item in _iter_query(session, base_params, config.per_category_limit):
                vacancy_id = item.get("id")
                if vacancy_id and vacancy_id not in seen:
                    seen.add(vacancy_id)
                    category_ids.append(vacancy_id)
            consecutive_failures = 0
        except requests.exceptions.RequestException as exc:
            logger.error("Категория %r упала %s", category_name, exc)
            consecutive_failures += 1
            if consecutive_failures >= 3:
                logger.error(
                    "Остановка парсинга..."
                )
                break
            continue

        ids.extend(category_ids)
        logger.info(
            "Категория %r собрано %d ID (всего уникальных %d)",
            category_name, len(category_ids), len(ids),
        )

    return ids


def fetch_vacancy_details(
    vacancy_ids: list[str],
    config: ParserConfig,
) -> list[dict[str, Any]]:
    session = _build_session(config.user_agent, config.access_token)
    config.raw_dir.mkdir(parents=True, exist_ok=True)

    vacancies: list[dict[str, Any]] = []
    batch: list[dict[str, Any]] = []
    batch_size = 500
    batch_idx = 0

    for i, vacancy_id in enumerate(vacancy_ids, start=1):
        try:
            response = session.get(
                f"{HH_VACANCIES_ENDPOINT}/{vacancy_id}",
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code == 404:
                logger.debug("Вакансия %s не найдена", vacancy_id)
                continue
            response.raise_for_status()
            data = response.json()
            vacancies.append(data)
            batch.append(data)
        except requests.HTTPError as exc:
            logger.warning("Не удалось скачать вакансию %s %s", vacancy_id, exc)
            continue
        finally:
            time.sleep(REQUEST_DELAY)

        if len(batch) >= batch_size:
            _save_batch(batch, config.raw_dir, batch_idx)
            batch = []
            batch_idx += 1

        if i % 200 == 0:
            logger.info("Скачано %d / %d вакансий", i, len(vacancy_ids))

    if batch:
        _save_batch(batch, config.raw_dir, batch_idx)

    return vacancies


def _save_batch(batch: list[dict[str, Any]], raw_dir: Path, idx: int) -> None:
    path = raw_dir / f"vacancies_batch_{idx:04d}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)
    logger.info("Сохранён батч %d %d вакансий %s", idx, len(batch), path)


def run_parser(config: ParserConfig) -> Path:
    logger.info(
        "Регион=%s, лимит на категорию=%d",
        config.area, config.per_category_limit,
    )
    ids = fetch_balanced_vacancy_ids(config)
    logger.info("Всего уникальных ID %d", len(ids))

    vacancies = fetch_vacancy_details(ids, config)
    logger.info("Скачано полных карточек %d", len(vacancies))

    output_path = config.raw_dir / "vacancies_all.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(vacancies, f, ensure_ascii=False, indent=2)
    logger.info("Итог сохранён в %s", output_path)

    return output_path