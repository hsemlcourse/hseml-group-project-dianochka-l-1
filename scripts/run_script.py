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
    p = argparse.ArgumentParser(description="Парсер вакансий с hh.ru")
    p.add_argument(
        "--user-agent",
        required=True,
        help='Заголовок User-Agent, например: "SalaryPredictor/1.0 (email@.ru)"',
    )
    p.add_argument(
        "--area",
        type=int,
        default=113,
        help="ID региона",
    )
    p.add_argument(
        "--per-category",
        type=int,
        default=500,
        help="Сколько вакансий брать из каждой профессиональной категории "
             "(по умолчанию 500",
    )
    p.add_argument(
        "--all-salaries",
        action="store_true",
        help="Собирать и вакансии без указанной зарплаты...по умолчанию — только с зарплатой",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Куда складывать сырые JSON",
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Куда складывать обработанные parquet",
    )
    p.add_argument(
        "--skip-parse",
        action="store_true",
        help="Пропустить парсинг и только обработать уже скачанные данные",
    )
    return p.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    raw_file = args.raw_dir / "vacancies_all.json"

    if not args.skip_parse:
        access_token = os.environ.get("HH_ACCESS_TOKEN")
        if access_token:
            logging.info("Используется OAuth-токен из HH_ACCESS_TOKEN")
        else:
            logging.info("HH_ACCESS_TOKEN не задан... работа без авторизации")

        config = ParserConfig(
            user_agent=args.user_agent,
            access_token=access_token,
            area=args.area,
            per_category_limit=args.per_category,
            only_with_salary=not args.all_salaries,
            raw_dir=args.raw_dir,
        )
        raw_file = run_parser(config)

    if not raw_file.exists():
        logging.error("Нет сырых данных по пути %s... Запустите без --skip-parse", raw_file)
        return 1

    try:
        run_preprocessing(raw_file, args.processed_dir)
    except ValueError as exc:
        logging.error("Предобработка не запущена %s", exc)
        return 2

    logging.info("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())