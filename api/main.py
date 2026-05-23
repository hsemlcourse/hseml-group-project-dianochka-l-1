from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class ListToStringTransformer(BaseEstimator, TransformerMixin):
    """Превращает колонку со списками в строку через пробел для TfidfVectorizer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(X.ravel() if hasattr(X, "ravel") else X)
        return s.apply(self._join).values

    @staticmethod
    def _join(x) -> str:
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
        try:
            return " ".join(str(i) for i in x)
        except TypeError:
            return str(x)


class TextConcatTransformer(BaseEstimator, TransformerMixin):
    """Склеивает несколько текстовых колонок в одну строку."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (
            pd.DataFrame(X)
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .values
        )


class BoolToFloatTransformer(BaseEstimator, TransformerMixin):
    """Приводит булевы колонки к float для SimpleImputer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).astype(float).values


class SpectralClusterFeature(BaseEstimator, TransformerMixin):
    """
    TF-IDF → TruncatedSVD → SpectralClustering на train.
    На transform: ближайший центроид в SVD-пространстве → метка кластера.
    """

    def __init__(
        self,
        text_col: str = "name",
        n_clusters: int = 20,
        svd_components: int = 50,
        knn_neighbors: int = 15,
        random_state: int = 42,
    ):
        self.text_col = text_col
        self.n_clusters = n_clusters
        self.svd_components = svd_components
        self.knn_neighbors = knn_neighbors
        self.random_state = random_state

    def _extract_text(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            s = X[self.text_col]
        else:
            s = pd.Series(np.asarray(X).ravel())
        return s.fillna("").astype(str).values

    def fit(self, X, y=None):
        texts = self._extract_text(X)
        self.tfidf_ = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), min_df=2
        )
        X_tfidf = self.tfidf_.fit_transform(texts)
        n_comp = min(self.svd_components, X_tfidf.shape[1] - 1)
        self.svd_ = TruncatedSVD(
            n_components=n_comp, random_state=self.random_state
        )
        X_svd = self.svd_.fit_transform(X_tfidf)
        n_nb = min(self.knn_neighbors, max(2, X_svd.shape[0] - 1))
        self.spectral_ = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=n_nb,
            assign_labels="kmeans",
            random_state=self.random_state,
            n_jobs=-1,
        )
        labels = self.spectral_.fit_predict(X_svd)
        centroids = np.zeros(
            (self.n_clusters, X_svd.shape[1]), dtype=np.float32
        )
        for k in range(self.n_clusters):
            mask = labels == k
            centroids[k] = (
                X_svd[mask].mean(axis=0)
                if mask.any()
                else np.full(X_svd.shape[1], np.inf, dtype=np.float32)
            )
        self.centroids_ = centroids
        self.train_labels_ = labels
        return self

    def transform(self, X) -> pd.DataFrame:
        X_svd = self.svd_.transform(
            self.tfidf_.transform(self._extract_text(X))
        )
        labels = pairwise_distances_argmin(X_svd, self.centroids_)
        return pd.DataFrame(
            {"name_cluster_id": [f"c{lbl}" for lbl in labels]}
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(["name_cluster_id"])


_CUSTOM_CLASSES = [
    BoolToFloatTransformer,
    ListToStringTransformer,
    TextConcatTransformer,
    SpectralClusterFeature,
]

for _mod_name in ("__main__", "__mp_main__"):
    _mod = sys.modules.get(_mod_name)
    if _mod is not None:
        for _cls in _CUSTOM_CLASSES:
            setattr(_mod, _cls.__name__, _cls)
        logger.debug("Классы зарегистрированы в модуле %s", _mod_name)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/final_model.joblib"))

app = FastAPI(
    title="Salary Predictor API",
    description="Предсказание зарплаты вакансий hh.ru на основе LightGBM-пайплайна",
    version="1.1.0",
)

# Глобальный пайплайн — загружается один раз при старте сервера
_pipeline = None


@app.on_event("startup")
def load_model() -> None:
    global _pipeline

    for _mod_name in ("__main__", "__mp_main__"):
        _mod = sys.modules.get(_mod_name)
        if _mod is not None:
            for _cls in _CUSTOM_CLASSES:
                setattr(_mod, _cls.__name__, _cls)

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Файл модели не найден: {MODEL_PATH}")

    logger.info("Загружаю модель из %s …", MODEL_PATH)
    _pipeline = joblib.load(MODEL_PATH)
    logger.info(
        "Модель загружена: %s", type(_pipeline).__name__
    )

class VacancyFeatures(BaseModel):
    """Признаки одной вакансии. Лишние поля игнорируются (extra='allow')."""

    # Основные признаки модели
    name: str = ""
    area_name: str | None = None
    employer_name: str | None = None
    experience: str | None = None
    experience_name: str | None = None
    employment: str | None = None
    schedule: str | None = None
    description: str = ""
    key_skills: list[str] = []
    key_skills_count: int = 0
    professional_role_names: list[str] = []
    professional_role_ids: list[Any] = []
    has_test: bool = False
    response_letter_required: bool = False

    # Служебные поля из hh API — пайплайн их не использует (remainder="drop")
    id: Any = None
    area_id: Any = None
    employer_id: Any = None
    employer_trusted: bool | None = None
    published_at: Any = None
    salary_from_rub: float | None = None
    salary_to_rub: float | None = None
    salary_mid_rub: float | None = None

    model_config = {"extra": "allow"}


class PredictRequest(BaseModel):
    vacancies: list[VacancyFeatures]


class PredictResponse(BaseModel):
    predictions: list[float]

@app.get("/health", tags=["infra"])
def health():
    """Healthcheck для docker-compose и Streamlit-сайдбара."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest):
    """Предсказывает зарплату для одной или нескольких вакансий."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    if not request.vacancies:
        raise HTTPException(status_code=422, detail="Список вакансий пустой")

    try:
        rows = [v.model_dump() for v in request.vacancies]
        df = pd.DataFrame(rows)
        preds = _pipeline.predict(df)
        preds = np.clip(preds, 10_000, None).tolist()
    except Exception:
        logger.exception("Ошибка инференса")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка модели")

    return PredictResponse(predictions=preds)