"""Fixtures communes : configuration du projet et modèle entraîné une seule fois."""

from __future__ import annotations

from pathlib import Path

import pytest

from churn.config import load_config
from churn.train import run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path):
    """Configuration du dépôt, avec des chemins absolus et des artefacts dans tmp_path."""
    c = load_config(ROOT / "config" / "train.yaml")
    c["data"]["csv_path"] = str(ROOT / c["data"]["csv_path"])
    c["artifacts_dir"] = str(tmp_path / "artifacts")
    return c


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Entraîne le modèle de référence une fois pour toute la session de tests."""
    c = load_config(ROOT / "config" / "train.yaml")
    c["data"]["csv_path"] = str(ROOT / c["data"]["csv_path"])
    out = tmp_path_factory.mktemp("run")
    c["artifacts_dir"] = str(out)
    metrics = run(c)
    return {"cfg": c, "dir": out, "metrics": metrics}
