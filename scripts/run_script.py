"""
Единая точка входа для пайплайна.

Команды:
    parse    — собрать вакансии с hh.ru API в data/raw/
    process  — обработать сырые JSON из data/raw/ в data/processed/vacancies.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Добавляем корень проекта в sys.path, чтобы импорты src.* работали
# даже если запускаем без `pip install -e .`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Подгрузим .env, если он есть. python-dotenv опционален —
# если пакета нет, просто читаем os.environ как обычно.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def setup_logging() -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Пайплайн данных для salary-predictor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- parse: сбор вакансий через API ---
    p_parse = subparsers.add_parser("parse", help="Парсить вакансии через hh.ru API")
    p_parse.add_argument(
        "--user-agent",
        required=True,
        help='User-Agent для hh.ru API, например: "MyProject/1.0 (email@example.com)"',
    )
    p_parse.add_argument("--area", type=int, default=113, help="ID региона hh.ru (113 = Россия)")
    p_parse.add_argument(
        "--per-category", type=int, default=500, help="Лимит вакансий на категорию"
    )
    p_parse.add_argument(
        "--all-salaries",
        action="store_true",
        help="Собирать вакансии и без указанной зарплаты",
    )
    p_parse.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Куда сохранять сырые JSON",
    )

    # --- process: обработка собранных JSON ---
    p_proc = subparsers.add_parser(
        "process", help="Обработать собранные JSON в parquet"
    )
    p_proc.add_argument(
        "--raw-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "vacancies_all.json",
        help="Путь к итоговому JSON, который вернул парсер",
    )
    p_proc.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Куда сохранить parquet",
    )

    return parser.parse_args()


def run_parse(args: argparse.Namespace) -> int:
    from src.parser import ParserConfig, fetch_access_token, run_parser

    # Получаем токен. Приоритет:
    # 1) готовый HH_ACCESS_TOKEN, если уже выдан
    # 2) HH_CLIENT_ID + HH_CLIENT_SECRET через client_credentials
    # 3) без токена (анонимные запросы, лимиты жёстче)
    access_token: str | None = os.environ.get("HH_ACCESS_TOKEN")
    if access_token:
        logging.info("Используется готовый HH_ACCESS_TOKEN из окружения")
    else:
        client_id = os.environ.get("HH_CLIENT_ID")
        client_secret = os.environ.get("HH_CLIENT_SECRET")
        if client_id and client_secret:
            logging.info("Получаю access_token по client_credentials")
            try:
                access_token = fetch_access_token(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=args.user_agent,
                )
            except Exception as exc:  # noqa: BLE001
                logging.error("Не удалось получить access_token: %s", exc)
                return 1
        else:
            logging.warning(
                "Нет HH_ACCESS_TOKEN и нет пары HH_CLIENT_ID/HH_CLIENT_SECRET. "
                "Парсер пойдёт анонимно — лимиты будут жёстче."
            )

    config = ParserConfig(
        user_agent=args.user_agent,
        access_token=access_token,
        area=args.area,
        per_category_limit=args.per_category,
        only_with_salary=not args.all_salaries,
        raw_dir=args.raw_dir,
    )

    try:
        raw_file = run_parser(config)
    except Exception as exc:  # noqa: BLE001
        logging.error("Парсер упал: %s", exc)
        return 1

    if not raw_file.exists() or raw_file.stat().st_size < 100:
        logging.error("Парсер не собрал данные (файл пустой или отсутствует)")
        return 1

    logging.info("Парсинг завершён. Сырой JSON: %s", raw_file)
    return 0


def run_process(args: argparse.Namespace) -> int:
    from src.preprocessing import run_preprocessing

    if not args.raw_file.exists():
        logging.error("Не найден файл с сырыми данными: %s", args.raw_file)
        return 1

    try:
        output_path = run_preprocessing(
            raw_path=args.raw_file,
            processed_dir=args.processed_dir,
        )
    except ValueError as exc:
        logging.error("Обработка упала: %s", exc)
        return 2

    logging.info("Обработка завершена. Parquet: %s", output_path)
    return 0


def main() -> int:
    setup_logging()
    args = parse_args()

    if args.command == "parse":
        return run_parse(args)
    if args.command == "process":
        return run_process(args)

    logging.error("Неизвестная команда: %s", args.command)
    return 2


if __name__ == "__main__":
    sys.exit(main())