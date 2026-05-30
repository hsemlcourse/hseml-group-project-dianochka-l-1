"""
Streamlit-интерфейс для предсказания зарплаты вакансий hh.ru.
"""
from __future__ import annotations

import html as _html
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Добавляем корень проекта в sys.path, чтобы импортировать src.*
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import _extract_row  # noqa: E402

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://api:8080")
HH_API_BASE   = "https://api.hh.ru"

HH_CLIENT_ID     = os.getenv("HH_CLIENT_ID", "")
HH_CLIENT_SECRET = os.getenv("HH_CLIENT_SECRET", "")
HH_USER_AGENT    = os.getenv("HH_USER_AGENT", "SalaryPredictor/1.0 (support@example.com)")
HH_ACCESS_TOKEN  = os.getenv("HH_ACCESS_TOKEN", "")

EXPERIENCE_MAP: dict[str, dict] = {
    "Нет опыта":      {"id": "noExperience",  "name": "Нет опыта"},
    "От 1 до 3 лет":  {"id": "between1And3",  "name": "От 1 года до 3 лет"},
    "От 3 до 6 лет":  {"id": "between3And6",  "name": "От 3 до 6 лет"},
    "Более 6 лет":    {"id": "moreThan6",      "name": "Более 6 лет"},
}

CITIES = [
    "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург",
    "Нижний Новгород", "Казань", "Челябинск", "Самара", "Уфа",
    "Ростов-на-Дону", "Краснодар", "Пермь", "Воронеж", "Волгоград",
    "Красноярск", "Саратов", "Тюмень", "Омск", "Тольятти", "Ижевск",
]

AREAS = {113: "🇷🇺 Вся Россия", 1: "🏙 Москва", 2: "🌊 Санкт-Петербург"}

SALARY_LEVELS = [
    (40_000,        "🟡 Начальный"),
    (90_000,        "🟠 Junior+"),
    (160_000,       "🔵 Middle"),
    (300_000,       "🟣 Senior"),
    (float("inf"),  "⭐ Lead / Staff"),
]

st.set_page_config(
    page_title="Salary Predictor · hh.ru",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Общие ── */
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }

/* ── Карточка результата ── */
.salary-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 36px 40px;
    text-align: center;
    color: #fff;
    margin: 20px 0 12px;
    position: relative;
    overflow: hidden;
}
.salary-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 60% 40%, rgba(99,102,241,0.18) 0%, transparent 60%);
    pointer-events: none;
}
.salary-card .sc-label   { font-size: 13px; letter-spacing: 2px; text-transform: uppercase; opacity: 0.55; margin-bottom: 8px; }
.salary-card .sc-amount  { font-size: 56px; font-weight: 800; letter-spacing: -2px; line-height: 1; color: #e0e7ff; }
.salary-card .sc-sub     { font-size: 14px; opacity: 0.5; margin-top: 6px; }
.salary-card .sc-level   { display: inline-block; margin-top: 16px; padding: 4px 16px;
                            background: rgba(255,255,255,0.08); border-radius: 100px;
                            font-size: 14px; font-weight: 600; letter-spacing: 0.5px; }

/* ── Карточка вакансии из hh ── */
.vacancy-card {
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 12px 0;
    background: #fafafa;
}
.vacancy-card h4 { margin: 0 0 8px; font-size: 18px; font-weight: 700; color: #111; }
.vacancy-card .vc-meta { font-size: 13px; color: #6b7280; display: flex; gap: 16px; flex-wrap: wrap; }
.vacancy-card .vc-badge {
    background: #eef2ff; color: #4f46e5; font-size: 12px; font-weight: 600;
    padding: 2px 10px; border-radius: 100px; display: inline-block; margin-right: 4px;
}

/* ── Sidebar метки статуса ── */
.status-ok  { color: #16a34a; font-weight: 600; }
.status-err { color: #dc2626; font-weight: 600; }
.status-warn { color: #d97706; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_hh_session() -> tuple[requests.Session, bool]:
    """Создаёт requests.Session с авторизацией hh.ru."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": HH_USER_AGENT,
        "HH-User-Agent": HH_USER_AGENT,
        "Accept": "application/json",
    })

    token = HH_ACCESS_TOKEN
    if not token and HH_CLIENT_ID and HH_CLIENT_SECRET:
        try:
            resp = session.post(
                f"{HH_API_BASE}/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": HH_CLIENT_ID,
                    "client_secret": HH_CLIENT_SECRET,
                },
                timeout=10,
            )
            if resp.ok:
                token = resp.json().get("access_token", "")
        except Exception:
            pass

    if token:
        session.headers["Authorization"] = f"Bearer {token}"

    return session, bool(token)


@st.cache_resource(show_spinner=False)
def _check_api() -> tuple[bool, str]:
    try:
        r = requests.get(f"{INFERENCE_URL}/health", timeout=5)
        return r.ok, INFERENCE_URL
    except Exception:
        return False, INFERENCE_URL


def api_status() -> tuple[bool, str]:
    return _check_api()


def predict_salary(features: dict) -> float:
    """Отправляет dict признаков в FastAPI → возвращает зарплату в рублях."""
    api_ok, url = api_status()
    if not api_ok:
        raise RuntimeError(f"Inference API недоступен по адресу {url}")

    resp = requests.post(
        f"{INFERENCE_URL}/predict",
        json={"vacancies": [features]},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Ошибка API {resp.status_code}: {resp.text[:300]}")
    return float(resp.json()["predictions"][0])


def show_salary(salary: float, title: str = "") -> None:
    """Рендерит карточку с предсказанной зарплатой."""
    fmt   = f"{salary:,.0f}".replace(",", "\u00a0")  # неразрывный пробел
    level = SALARY_LEVELS[-1][1]
    for threshold, label in SALARY_LEVELS:
        if salary <= threshold:
            level = label
            break

    st.markdown(f"""
    <div class="salary-card">
        <div class="sc-label">{"📋 " + title if title else "Предсказание модели"}</div>
        <div class="sc-amount">₽ {fmt}</div>
        <div class="sc-sub">в месяц · на руки · медиана</div>
        <div class="sc-level">{level}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Годовой доход", f"₽ {salary * 12 / 1_000_000:.1f}М")
    c2.metric("В рабочий день", f"₽ {salary / 22:,.0f}")
    c3.metric("Грейд (оценка)", level.split(" ", 1)[1])


def _strip_html(text: str) -> str:
    text = _html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _salary_str(sal: dict | None) -> str:
    if not sal:
        return "не указана"
    parts = []
    if sal.get("from"):
        parts.append(f"от {sal['from']:,}")
    if sal.get("to"):
        parts.append(f"до {sal['to']:,}")
    cur = sal.get("currency", "")
    return (" ".join(parts) + f" {cur}").strip() if parts else "не указана"

def extract_id(url_or_id: str) -> str | None:
    s = url_or_id.strip()
    if s.isdigit():
        return s
    m = re.search(r"/vacancy/(\d+)", s)
    return m.group(1) if m else None


def hh_get_vacancy(vacancy_id: str) -> dict | None:
    session, _ = get_hh_session()
    try:
        r = session.get(f"{HH_API_BASE}/vacancies/{vacancy_id}", timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Ошибка API hh.ru: {e}")
        return None


def hh_search(query: str, area_id: int, n: int = 15) -> list[dict]:
    session, _ = get_hh_session()
    try:
        r = session.get(
            f"{HH_API_BASE}/vacancies",
            params={"text": query, "area": area_id, "per_page": n,
                    "only_with_salary": False, "order_by": "relevance"},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("items", [])
    except Exception as e:
        st.error(f"Ошибка поиска: {e}")
        return []


def vacancy_to_features(raw: dict) -> dict:
    """Конвертирует сырой JSON hh.ru → dict для пайплайна (через preprocessing.py)."""
    return _extract_row(raw)


def manual_to_features(
    name: str, employer: str, area: str, exp_key: str,
    description: str, skills: list[str], role: str,
    has_test: bool, letter: bool,
) -> dict:
    """Собирает dict признаков из ручной формы."""
    exp = EXPERIENCE_MAP[exp_key]
    return {
        "id": None,
        "name": name,
        "area_id": None,
        "area_name": area,
        "employer_name": employer,
        "employer_id": None,
        "employer_trusted": True,
        "experience": exp["id"],
        "experience_name": exp["name"],
        "employment": None,
        "schedule": None,
        "professional_role_ids": [],
        "professional_role_names": [role] if role else [],
        "has_test": has_test,
        "response_letter_required": letter,
        "description": description,
        "key_skills": skills,
        "key_skills_count": len(skills),
        "published_at": None,
        "salary_from_rub": None,
        "salary_to_rub": None,
        "salary_mid_rub": None,
    }


def show_vacancy_card(raw: dict) -> None:
    """Рендерит HTML-карточку загруженной вакансии."""
    name    = raw.get("name", "Вакансия")
    emp     = (raw.get("employer") or {}).get("name", "—")
    area    = (raw.get("area") or {}).get("name", "—")
    exp     = (raw.get("experience") or {}).get("name", "—")
    sal_str = _salary_str(raw.get("salary"))
    skills  = [s.get("name", "") for s in (raw.get("key_skills") or [])]
    badges  = "".join(f'<span class="vc-badge">{s}</span>' for s in skills[:12])
    url     = raw.get("alternate_url", f"https://hh.ru/vacancy/{raw.get('id','')}")

    st.markdown(f"""
    <div class="vacancy-card">
        <h4>{name}</h4>
        <div class="vc-meta">
            <span>🏢 {emp}</span>
            <span>📍 {area}</span>
            <span>⏱ {exp}</span>
            <span>💰 {sal_str}</span>
            <span><a href="{url}" target="_blank" style="color:#4f46e5">Открыть на hh.ru →</a></span>
        </div>
        {"<div style='margin-top:12px;'>" + badges + "</div>" if badges else ""}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Описание вакансии"):
        desc = _strip_html(raw.get("description") or "")
        st.write(desc[:3000] + ("…" if len(desc) > 3000 else ""))


with st.sidebar:
    st.markdown("## 💼 Salary Predictor")
    st.caption("Предсказание зарплат на основе данных hh.ru")
    st.divider()

    st.markdown("#### Статус сервисов")
    api_ok, api_url = api_status()
    _, hh_auth = get_hh_session()

    if api_ok:
        st.markdown(f'<p class="status-ok">✅ Inference API · {api_url}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-err">❌ Inference API недоступен</p>', unsafe_allow_html=True)
        st.caption(f"Ожидаемый адрес: {api_url}")

    if hh_auth:
        st.markdown('<p class="status-ok">✅ hh.ru API · авторизован</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-warn">⚠️ hh.ru API · анонимный режим</p>', unsafe_allow_html=True)
        if not HH_CLIENT_ID:
            st.caption("Добавьте HH_CLIENT_ID и HH_CLIENT_SECRET в .env")

    st.divider()
    st.markdown("#### О модели")
    st.markdown("""
| Параметр | Значение |
|---|---|
| Алгоритм | LightGBM (Optuna) |
| MAE | 21 785 ₽ |
| RMSE | 32 497 ₽ |
| R² | 0.582 |
| Таргет | salary_mid_rub |
""")

    with st.expander("Параметры подключения"):
        st.code(f"INFERENCE_URL={INFERENCE_URL}", language="bash")

    if not api_ok:
        st.divider()
        st.warning("Inference API не отвечает. Убедитесь, что контейнер api запущен.")
        if st.button("🔄 Повторить проверку"):
            st.cache_resource.clear()
            st.rerun()


st.markdown("# Предсказание зарплаты вакансий")
st.caption(
    "Введите параметры вакансии вручную или найдите её на hh.ru — "
    "LightGBM-модель предскажет медианную зарплату на руки в рублях."
)
st.divider()

tab_manual, tab_hh = st.tabs(["📝 Ручной ввод", "🔍 Поиск на hh.ru"])


with tab_manual:
    with st.form("manual_form", border=True):
        col_l, col_r = st.columns([3, 2])

        with col_l:
            f_name = st.text_input(
                "Название вакансии *",
                placeholder="Senior Data Scientist",
            )
            f_description = st.text_area(
                "Описание вакансии",
                placeholder=(
                    "Мы ищем опытного Data Scientist для разработки моделей "
                    "рекомендательных систем. Требования: знание Python, "
                    "опыт с LightGBM/CatBoost, понимание статистики..."
                ),
                height=160,
            )
            f_skills_raw = st.text_area(
                "Ключевые навыки (через запятую)",
                placeholder="Python, SQL, LightGBM, Docker, Spark",
                height=80,
            )

        with col_r:
            f_employer = st.text_input("Работодатель", placeholder="Яндекс")
            f_area = st.selectbox("Город", options=CITIES, index=0)
            f_experience = st.selectbox(
                "Опыт работы *",
                options=list(EXPERIENCE_MAP.keys()),
                index=1,
            )
            f_role = st.text_input(
                "Профессиональная роль",
                placeholder="Data Scientist",
            )
            f_has_test = st.checkbox("Тестовое задание")
            f_letter   = st.checkbox("Нужно сопроводительное письмо")

        submitted = st.form_submit_button(
            "🔮 Предсказать зарплату",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not f_name.strip():
            st.warning("Укажите название вакансии — это ключевой признак для модели.")
        elif not api_ok:
            st.error("API сервер недоступен. Проверьте, что контейнер поднят.")
        else:
            skills = [s.strip() for s in f_skills_raw.split(",") if s.strip()]
            features = manual_to_features(
                name=f_name, employer=f_employer, area=f_area,
                exp_key=f_experience, description=f_description,
                skills=skills, role=f_role,
                has_test=f_has_test, letter=f_letter,
            )
            with st.spinner("Запускаю модель…"):
                try:
                    salary = predict_salary(features)
                    show_salary(salary, f_name)
                except Exception as e:
                    st.error(f"Ошибка предсказания: {e}")


with tab_hh:
    hh_mode = st.radio(
        "Способ выбора вакансии",
        ["🔗 По ссылке или ID", "🔍 Поиск по запросу"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    if hh_mode == "🔗 По ссылке или ID":
        url_col, btn_col = st.columns([5, 1])
        with url_col:
            url_input = st.text_input(
                "Ссылка на вакансию или числовой ID",
                placeholder="https://hh.ru/vacancy/123456789  или  123456789",
                label_visibility="collapsed",
            )
        with btn_col:
            fetch_btn = st.button("Загрузить", type="primary", use_container_width=True)

        if fetch_btn:
            if not url_input.strip():
                st.warning("Вставьте ссылку или ID вакансии.")
            else:
                vid = extract_id(url_input)
                if not vid:
                    st.error(
                        "Не удалось распознать ID. "
                        "Вставьте ссылку вида `https://hh.ru/vacancy/XXXXXXX` или числовой ID."
                    )
                else:
                    with st.spinner(f"Загружаю вакансию #{vid}…"):
                        raw = hh_get_vacancy(vid)
                    if raw is None:
                        st.error("Вакансия не найдена или удалена.")
                    else:
                        st.session_state["hh_vacancy"] = raw

    else:
        q_col, area_col, btn_col = st.columns([4, 2, 1])
        with q_col:
            search_q = st.text_input(
                "Запрос", placeholder="Senior Python Developer",
                label_visibility="collapsed",
            )
        with area_col:
            area_sel = st.selectbox(
                "Регион", options=list(AREAS.keys()),
                format_func=lambda x: AREAS[x],
                label_visibility="collapsed",
            )
        with btn_col:
            search_btn = st.button("Найти", type="primary", use_container_width=True)

        if search_btn:
            if not search_q.strip():
                st.warning("Введите поисковый запрос.")
            else:
                with st.spinner("Ищу вакансии на hh.ru…"):
                    items = hh_search(search_q, area_sel, n=15)
                if not items:
                    st.warning("Ничего не найдено. Попробуйте другой запрос или регион.")
                else:
                    st.session_state["search_items"] = items
                    st.session_state.pop("hh_vacancy", None)  # сбрасываем старую

        if "search_items" in st.session_state:
            items = st.session_state["search_items"]
            rows = [
                {
                    "ID":       v.get("id", ""),
                    "Название": v.get("name", ""),
                    "Компания": (v.get("employer") or {}).get("name", ""),
                    "Опыт":     (v.get("experience") or {}).get("name", ""),
                    "Зарплата": _salary_str(v.get("salary")),
                    "Город":    (v.get("area") or {}).get("name", ""),
                }
                for v in items
            ]
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            sel_id = st.selectbox(
                "Выберите вакансию",
                options=[r["ID"] for r in rows],
                format_func=lambda x: next(
                    (f"{r['Название']} · {r['Компания']}" for r in rows if r["ID"] == x),
                    str(x),
                ),
            )
            if st.button("Загрузить карточку выбранной вакансии", type="secondary"):
                with st.spinner("Загружаю полную карточку…"):
                    raw = hh_get_vacancy(str(sel_id))
                if raw:
                    st.session_state["hh_vacancy"] = raw
                else:
                    st.error("Не удалось загрузить карточку вакансии.")

    if "hh_vacancy" in st.session_state:
        raw = st.session_state["hh_vacancy"]
        st.divider()
        show_vacancy_card(raw)

        col_predict, col_clear = st.columns([4, 1])
        with col_predict:
            predict_btn = st.button(
                "🔮 Предсказать зарплату",
                type="primary",
                use_container_width=True,
                key="hh_predict_btn",
            )
        with col_clear:
            if st.button("✕ Сбросить", use_container_width=True, key="clear_btn"):
                del st.session_state["hh_vacancy"]
                st.rerun()

        if predict_btn:
            if not api_ok:
                st.error("API сервер недоступен.")
            else:
                features = vacancy_to_features(raw)
                with st.spinner("Запускаю модель…"):
                    try:
                        salary = predict_salary(features)
                        show_salary(salary, raw.get("name", ""))
                    except Exception as e:
                        st.error(f"Ошибка предсказания: {e}")