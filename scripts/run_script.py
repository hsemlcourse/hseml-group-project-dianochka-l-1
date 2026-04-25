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

# Добавляем src в путь, чтобы запускать скрипт без pip install -e
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import ParserConfig, run_parser
from src.preprocessing import run_preprocessing


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("parser.log", encoding="utf-8"),
        ],
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Пайплайн данных для salary-predictor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # обработка готового CSV
    p_m = subparsers.add_parser(
        "m",
        help="Обработать готовый датасет",
    )
    p_m.add_argument(
        "--raw-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "vacancies.csv",
        help="Путь к исходному CSV",
    )
    p_m.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Куда сохранить обработанный parquet",
    )
    p_m.add_argument(
        "--sources",
        nargs="*",
        default=["hh"],
        help="Какие data_source оставить",
    )
    p_m.add_argument(
        "--countries",
        nargs="*",
        default=["Россия"],
        help="Какие country_name оставить"
             "Передать --countries '' чтобы выключить фильтр",
    )
    p_m.add_argument(
        "--no-filtered-csv",
        action="store_true",
        help="Не сохранять промежуточный CSV после фильтрации",
    )

    # Режим parse (парсер hh.ru API)
    p_parse = subparsers.add_parser(
        "parse",
        help="Парсить вакансии через hh.ru API (требует работающий доступ)",
    )
    p_parse.add_argument("--user-agent", required=True, help="User-Agent для hh.ru API")
    p_parse.add_argument("--area", type=int, default=113, help="ID региона hh.ru")
    p_parse.add_argument("--per-category", type=int, default=500, help="Вакансий на категорию")
    p_parse.add_argument("--all-salaries", action="store_true", help="Не только с зарплатой")
    p_parse.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Куда класть сырые JSON",
    )

    return parser.parse_args()


def run_m(args: argparse.Namespace) -> int:
    from src.preprocessing_data import run_preprocessing

    if not args.raw_file.exists():
        logging.error(
            "Не найден CSV %s\n",
            args.raw_file, args.raw_file,
        )
        return 1

    sources = [s for s in args.sources if s] or None
    countries = [c for c in args.countries if c] or None

    try:
        run_preprocessing(
            raw_path=args.raw_file,
            processed_dir=args.processed_dir,
            sources=sources,
            countries=countries,
            save_filtered_csv=not args.no_filtered_csv,
        )
    except ValueError as exc:
        logging.error("Предобработка упала %s", exc)
        return 2

    logging.info("Done")
    return 0

def run_parse(args: argparse.Namespace) -> int:
    from src.parser import ParserConfig, run_parser

    access_token = os.environ.get("HH_ACCESS_TOKEN")
    if access_token:
        logging.info("Используется OAuth-токен из HH_ACCESS_TOKEN")

    config = ParserConfig(
        user_agent=args.user_agent,
        access_token=access_token,
        area=args.area,
        per_category_limit=args.per_category,
        only_with_salary=not args.all_salaries,
        raw_dir=args.raw_dir,
    )
    raw_file = run_parser(config)
    if not raw_file.exists() or raw_file.stat().st_size < 100:
        logging.error(
            "Парсер не собрал данные"
        )
        return 1

    logging.info(
        "Парсинг завершён... Сырой JSON лежит в %s ",
        raw_file,
    )
    return 0

def main() -> int:
    setup_logging()
    args = parse_args()

    if args.command == "m":
        return run_m(args)
    if args.command == "parse":
        return run_parse(args)

if __name__ == "__main__":
    sys.exit(main())