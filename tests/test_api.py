"""API d'inférence : /health, /predict, validation des entrées, modèle absent."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import main as api

CLIENT = {
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


@pytest.fixture
def client(trained, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(trained["dir"] / "model.joblib"))
    monkeypatch.setenv("MODEL_VERSION", "test")
    api._cache.clear()
    return TestClient(api.app)


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": "test"}


def test_predict_types_json(client):
    r = client.post("/predict", json=CLIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["exited"] in (0, 1)
    assert isinstance(body["probability"], float) and 0 <= body["probability"] <= 1
    assert body["model_version"] == "test"


def test_membre_actif_moins_risque(client):
    """IsActiveMember est bien une variable du modèle : elle change la probabilité."""
    inactif = client.post("/predict", json=CLIENT).json()["probability"]
    actif = client.post("/predict", json={**CLIENT, "IsActiveMember": 1}).json()["probability"]
    assert actif < inactif


@pytest.mark.parametrize(
    "change",
    [{"Geography": "Italy"}, {"Age": -5}, {"Age": 81}, {"HasCrCard": 2}, {"Surname": "Doe"}],
)
def test_entree_invalide_422(client, change):
    assert client.post("/predict", json={**CLIENT, **change}).status_code == 422


def test_modele_absent_503(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "absent.joblib"))
    api._cache.clear()
    c = TestClient(api.app)
    assert c.get("/health").status_code == 503
    assert c.post("/predict", json=CLIENT).status_code == 503


def test_age_maximal_coherent_avec_le_nettoyage():
    """L'API refuse exactement les âges écartés à l'entraînement (data.max_age)."""
    from pathlib import Path

    from churn.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "train.yaml")
    assert api.MAX_AGE == cfg["data"]["max_age"]
