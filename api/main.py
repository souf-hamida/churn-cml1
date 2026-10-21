"""API d'inférence FastAPI (TP3).

    uvicorn api.main:app --host 0.0.0.0 --port 8000

- GET  /health  : 200 si le modèle est chargé, 503 sinon (utilisé par le HEALTHCHECK Docker) ;
- POST /predict : un client -> {"exited": 0|1, "probability": float, "model_version": str}.

Le modèle (Pipeline complet : prétraitement + modèle) est lu dans MODEL_PATH
(défaut : artifacts/model.joblib) ; MODEL_VERSION l'identifie dans les réponses.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="Churn — API de prédiction", version="0.1.0")
_cache: dict[str, object] = {}

# Domaine d'entraînement : le nettoyage écarte les clients de plus de data.max_age ans
# (config/train.yaml, 12 clients sur 10 000). Le modèle n'a jamais vu ces âges : l'API les
# refuse (HTTP 422) plutôt que d'extrapoler. Cohérence vérifiée par tests/test_api.py.
MIN_AGE, MAX_AGE = 18, 80


class Client(BaseModel):
    """Variables d'entrée du modèle (les identifiants Surname, CustomerId… en sont exclus)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "CreditScore": 600,
                "Geography": "Germany",
                "Gender": "Female",
                "Age": 52,
                "Tenure": 3,
                "Balance": 120000.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 0,
                "EstimatedSalary": 90000.0,
            }
        },
    )

    CreditScore: int = Field(ge=0, le=1000)
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Female", "Male"]
    Age: int = Field(ge=MIN_AGE, le=MAX_AGE)
    Tenure: int = Field(ge=0, le=100)
    Balance: float = Field(ge=0)
    NumOfProducts: int = Field(ge=1, le=10)
    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1]
    EstimatedSalary: float = Field(ge=0)


class Prediction(BaseModel):
    exited: int
    probability: float
    model_version: str


def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", "artifacts/model.joblib"))


def model_version() -> str:
    return os.environ.get("MODEL_VERSION", "dev")


def get_model():
    """Chargement paresseux et mis en cache ; None si le fichier n'existe pas (encore)."""
    path = model_path()
    if _cache.get("path") != str(path):
        _cache.clear()
    if "model" not in _cache:
        if not path.is_file():
            return None
        _cache["model"] = joblib.load(path)
        _cache["path"] = str(path)
    return _cache["model"]


@app.get("/health")
def health():
    loaded = get_model() is not None
    body = {
        "status": "ok" if loaded else "no-model",
        "model_loaded": loaded,
        "model_version": model_version(),
    }
    return JSONResponse(body, status_code=200 if loaded else 503)


@app.post("/predict", response_model=Prediction)
def predict(client: Client) -> Prediction:
    model = get_model()
    if model is None:  # erreur côté serveur (503), pas côté client (400)
        raise HTTPException(status_code=503, detail=f"Modèle introuvable : {model_path()}")
    X = pd.DataFrame([client.model_dump()])
    # Conversion explicite numpy -> types Python : sinon la réponse JSON échoue (numpy.int64)
    exited = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0, 1])
    return Prediction(
        exited=exited, probability=round(probability, 4), model_version=model_version()
    )
