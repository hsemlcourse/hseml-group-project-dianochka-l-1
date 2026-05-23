[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание заработной платы любой вакансии

**Студент:** Левитская Диана Юрьевна

**Группа:** БИВ 238

## Оглавление
1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Быстрый старт](#быстрый-старт)
4. [Запуск пайплайна](#запуск-пайплайна)
5. [Запуск через Docker](#запуск-через-docker)
6. [Тесты и линтер](#тесты-и-линтер)
7. [Данные](#данные)
8. [Результаты](#результаты)
9. [Отчёт](#отчёт)


## Описание задачи

Проект посвящён задаче регрессии — предсказание уровня зарплаты по описанию вакансии.

**Задача:** регрессия
**Датасет:** парсинг hh.ru (API) или готовый CSV-датасет
**Таргет:** `salary_mid_rub` (средняя зарплата в рублях/мес, net)

**Целевая метрика:** MAE
Выбраны и вспомогательные метрики для лучшего понимания ситуации: RMSE и R2
**MAE** ... главное число для бизнеса и для финального вывода о качестве
**RMSE** ... диагностика поведения на высоких зарплатах + стандарт
**R2** ... удобный безразмерный показатель для сравнения моделей

В репозитории реализованы:
- парсер вакансий через API hh.ru (`src/parser.py`);
- предобработка JSON-данных из API (`src/preprocessing.py`);
- ноутбуки: EDA (`notebooks/01_eda.ipynb`), baseline (`notebooks/02_baseline.ipynb`),
  эксперименты (`notebooks/03_experiments.ipynb`);
- тесты пайплайна предобработки (`tests/test.py`);
- единая точка входа `scripts/run_script.py`;
- Docker-сборка (`Dockerfile` + `docker-compose.yml`).

## Структура репозитория

```
.
├── data
│   ├── processed/              # vacancies.parquet, vacancies_clean.parquet
│   └── raw/                    # сырой JSON / CSV
├── models/                     # сохранённые модели и метрики (.joblib, .json)
├── notebooks
│   ├── 01_eda.ipynb            # EDA, очистка выбросов, train/val/test split
│   ├── 02_baseline.ipynb       # Baseline (Dummy + LinearRegression)
│   └── 03_experiments.ipynb    # Эксперименты, финальная модель, выводы
├── report
│   ├── images/                 # графики для отчёта (создаются ноутбуками)
│   └── experiments.csv         # таблица экспериментов
├── src
│   ├── parser.py               # парсер hh.ru API
│   └── preprocessing.py        # JSON - DataFrame - parquet
├── tests
│   └── test.py                 # pytest-тесты предобработки
├── scripts
│   └── run_script.py           # CLI parse | process
├── Dockerfile
├── docker-compose.yml
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

### Вариант A. Использовать готовый CSV-датасет
Скачать готовый `vacancies.parquet` по ссылке и положить в `data/processed/`:
[Google Drive](https://drive.google.com/file/d/11NMJcb2o3Og7kanMeWPD7k1rPlPmsMp2/view?usp=sharing)

Далее открыть `notebooks/01_eda.ipynb` — он построит `vacancies_clean.parquet`,
от которого зависят `02_baseline.ipynb` и `03_experiments.ipynb`.

### Вариант B. Парсинг с hh.ru

1. Зарегистрировать приложение на https://dev.hh.ru/admin и получить
   `HH_CLIENT_ID` / `HH_CLIENT_SECRET`, положить их в `.env`:
   ```
   HH_CLIENT_ID=...
   HH_CLIENT_SECRET=...
   ```

2. Запустить парсер:
   ```bash
   python scripts/run_script.py parse \
       --user-agent "YourProjectName/1.0 (email@example.com)"
   ```
   Дополнительные параметры:
   - `--area` — регион (по умолчанию `113` = Россия)
   - `--per-category` — лимит вакансий на категорию (default 500)
   - `--all-salaries` — собирать вакансии и без указанной зарплаты
   - `--raw-dir` — куда сохранять сырой JSON

3. Запустить обработку JSON → parquet:
   ```bash
   python scripts/run_script.py process
   ```
   Параметры:
   - `--raw-file` — путь к JSON парсера (default `data/raw/vacancies_all.json`)
   - `--processed-dir` — куда положить parquet (default `data/processed/`)

После шагов A или B пройти ноутбуки в порядке `01 - 02 - 03`.

## Запуск через Docker

```bash
docker compose build
docker compose run --rm ml          # тесты
docker compose up jupyter           # Jupyter Lab на http://localhost:8888
```
Данные и модели подключены через volumes, поэтому изменения в `./data` и
`./models` хоста сразу видны в контейнере...

## Тесты и проверка кода

```bash
pytest
flake8 src/ --max-line-length 120
```

## Данные

- `data/raw/` — сырые JSON от парсера (батчи `vacancies_batch_*.json` +
  склейка `vacancies_all.json`)
- `data/processed/vacancies.parquet` — результат `preprocessing.py`
  (плоский DataFrame, отсечены явные выбросы по зарплате)
- `data/processed/vacancies_clean.parquet` — результат `01_eda.ipynb`
  (доп.очистка: drop leakage-колонок и константных полей, фильтр по 3σ
  на таргет)
- Крупные data-файлы исключены из git через `.gitignore`

## Результаты

### Baseline на test (`notebooks/02_baseline.ipynb`)

| Модель                  | MAE, ₽   | RMSE, ₽   | R²    |
|-------------------------|----------|-----------|-------|
| DummyRegressor (median) | 43 035   | 85 793    | −0.05 |
| LinearRegression        | 125 740  | 315 852   | −13.18 |

> LinearRegression «из коробки» на примерно 22k разреженных tf-idf фичах при 10k train
> предсказуемо переобучается — это и есть наш «честный» baseline.
> Регуляризация (Ridge) рассматривается уже в экспериментах.

### Финальная модель на test (`notebooks/03_experiments.ipynb`)

| Модель                       | MAE, ₽  | RMSE, ₽ | R²    |
|------------------------------|---------|---------|-------|
| **LightGBM (Optuna, 20 tr.)**| **21 785** | **32 497** | **0.582** |

Финальная модель в примерно 2 раза точнее тривиального бейзлайна.
Полное сравнение моделей, обсуждение TruncatedSVD, SpectralClustering и
обоснование выбора — в разделе «14. Выводы» в `03_experiments.ipynb`

## Запуск интерфейса (деплой модели)

Для работы интерфейса необходим файл `models/final_model.joblib`
(создаётся после прохождения `notebooks/03_experiments.ipynb`).

Добавьте в `.env` переменные для hh.ru API:
```
HH_CLIENT_ID=...
HH_CLIENT_SECRET=...
HH_USER_AGENT=YourProjectName/1.0 (email@example.com)
```

Запуск:
```bash
docker compose up api streamlit
```

- **Streamlit UI** → http://localhost:8501
- **FastAPI (Swagger)** → http://localhost:8080/docs

Интерфейс поддерживает два режима:
- **Ручной ввод** — заполните параметры вакансии в форме и получите предсказание зарплаты
- **Поиск на hh.ru** — вставьте ссылку на вакансию (`https://hh.ru/vacancy/...`) или введите поисковый запрос, выберите вакансию из списка и запустите предсказание

Ссылка на демонстрацию деплоя: https://disk.yandex.ru/d/sN59VO-5z5OMAw

## Отчёт
Полноценный отчёт в формате `report/report.md` будет добавлен на этапе CP3.
Промежуточные артефакты экспериментов уже доступны в `report/experiments.csv`
и `report/images/`.
Финальный отчёт: [`report/report.md`](report/report.md)