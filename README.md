[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание заработной платы любой вакансии

**Студент:** Левитская Диана Юрьевна

**Группа:** БИВ 238

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи
Проект посвящен задаче регрессии... прогнозу уровня зарплаты по данным вакансий.
В репозитории реализованы:
- парсер вакансий через API hh.ru (`src/parser.py`);
- два пайплайна предобработки:
  - для JSON-данных из API (`src/preprocessing.py`);
  - для готового CSV-датасета (`src/preprocessing_data.py`);
- ноутбуки с EDA и baseline-моделью (`notebooks/01_eda.ipynb`, `notebooks/02_baseline.ipynb`);
- базовые тесты на этапы предобработки (`tests/test.py`).

**Задача:** Регрессия

**Датасет:** Парсинг HH.ru

**Целевая метрика:** MAE
Выбраны и вспомогательные метрики для лучшего понимания ситуации: RMSE и R2
**MAE** ... главное число для бизнеса и для финального вывода о качестве
**RMSE** ... диагностика поведения на высоких зарплатах + стандарт
**R2** ... удобный безразмерный показатель для сравнения моделей

## Структура репозитория
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных для скачанного датасета
│   └── parser.py               # Готовый парсер с API hh.ru 
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone <https://github.com/hsemlcourse/hseml-group-project-dianochka-l-1>
cd hseml-group-project-dianochka-l-1

# 2. Создать и активировать окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```

## Запуск пайплайна

### 1) Предобработка готового CSV-датасета

По умолчанию скрипт ожидает файл `data/raw/vacancies.csv`, фильтрует данные и сохраняет parquet в `data/processed/vacancies.parquet`.

```bash
python scripts/run_script.py m
```

Полезные аргументы:
- `--raw-file` - путь к исходному CSV;
- `--processed-dir` - директория для результата;
- `--sources` - фильтр по `data_source` (по умолчанию `hh`);
- `--countries` - фильтр по `country_name` (по умолчанию `Россия`);
- `--no-filtered-csv` - не сохранять промежуточный отфильтрованный CSV.

Пример:

```bash
python scripts/run_script.py m --raw-file data/raw/vacancies.csv --sources hh --countries Россия
```

### 2) Парсинг вакансий с hh.ru API

Перед парсингом необходимо создать файл .env и закинуть в него HH_CLIENT_ID и HH_CLIENT_SECRET. Которые можно получить после регистрации и одобрения заявки на приложение со стороны API HH.ru (https://dev.hh.ru/admin)

```bash
python scripts/run_script.py parse --user-agent "YourProjectName/1.0 (email@example.com)"
```

Дополнительно можно задать:
- `--area` - регион (`113` = Россия)
- `--per-category` - лимит вакансий на категорию
- `--all-salaries` - собирать не только вакансии с указанной зарплатой
- `--raw-dir` - куда сохранять сырые JSON

## Запуск через Docker

```bash
docker compose build
docker compose run --rm ml pytest          # тесты
docker compose up jupyter                  # Jupyter Lab на :8888
```

## Тесты и проверка кода

```bash
pytest
ruff check src/ --line-length 120
```

## Данные

- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные
- Крупные data-файлы исключены из git через `.gitignore`

## Результаты

### Baseline (на test)
| Модель                  | MAE, руб | RMSE, руб | R2    |
|-------------------------|----------|-----------|-------|
| DummyRegressor (median) | 43035.17 | 85793.23  | -0.05 |
| Ridge (α=1.0)           | 26765.24 | 66303.87  | 0.375 |

> Ridge даёт ошибку ≈27 тыс. ₽, что в ~1.6 раза лучше тривиального бейзлайна
> Подробнее в `notebooks/02_baseline.ipynb` и `notebooks/03_experiments.ipynb`

## Отчёт
Финальный отчёт: [`report/report.md`](report/report.md)
