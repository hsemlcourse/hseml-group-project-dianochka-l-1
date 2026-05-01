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

Для этапа CP1 стоит посмотреть ноутбуки 01_eda и 02_baseline

**Задача:** Регрессия

**Датасет:** Парсинг HH.ru
Но заявку на создание проекта и получение лично токена HH.ru не успел рассмотреть к дедлайну этапа CP1, поэтому в качестве датасета пока использовались даннные с диска: https://drive.google.com/file/d/1DSojrM7FSwJPQWjMa93KFg2Necy89-CK/view?usp=sharing
(К следующему этапу датасет будет дополнен фрагментом, собранным мной самостоятельно)

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
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных для датасета с API hh.ru
|   ├── preprocessing_data.py   # Предобработка данных для скачанного датасета
│   └── modeling.py             # Обучение и оценка моделей
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

```bash
python scripts/run_script.py parse --user-agent "YourName/1.0 (email@example.com)"
```

Дополнительно можно задать:
- `--area` - регион (`113` = Россия);
- `--per-category` - лимит вакансий на категорию;
- `--all-salaries` - собирать не только вакансии с указанной зарплатой;
- `--raw-dir` - куда сохранять сырые JSON.

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
Здесь коротко выпишите результаты.
| Модель | [Метрика 1] | [Метрика 2] | Примечание |
|--------|-------------|-------------|------------|
| Baseline | — | — | |
| Лучшая модель | — | — | |


Текущие результаты и эксперименты отражены в ноутбуках:
- `notebooks/01_eda.ipynb`
- `notebooks/02_baseline.ipynb`

## Отчёт
Финальный отчёт: [`report/report.md`](report/report.md)
