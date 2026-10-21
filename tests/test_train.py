"""Entraînement : artefacts, reproductibilité, seuil de qualité, variantes."""

from __future__ import annotations

import json

import pytest

from churn.config import apply_overrides
from churn.data import save_prepared
from churn.evaluate import check_gate
from churn.train import run


def test_artefacts_ecrits(trained):
    d = trained["dir"]
    for name in ("model.joblib", "metrics.json", "confusion_matrix.png", "report.json"):
        assert (d / name).is_file(), name
    report = json.loads((d / "report.json").read_text(encoding="utf-8"))
    assert report["seed"] == 42
    assert len(report["data"]["sha256"]) == 64
    assert report["data"]["n_test"] == 1998
    assert sum(report["confusion_matrix"].values()) == 1998


def test_metriques_calculees_sur_le_test(trained):
    metrics = json.loads((trained["dir"] / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert metrics == trained["metrics"]


def test_seuil_minimal_de_qualite(trained):
    assert check_gate(trained["metrics"], trained["cfg"]) == []
    assert trained["metrics"]["f1"] >= 0.45
    assert trained["metrics"]["roc_auc"] >= 0.75


def test_reproductibilite_deux_entrainements(trained, cfg):
    """Même code + mêmes données + même configuration = mêmes métriques."""
    assert run(cfg) == trained["metrics"]


def test_porte_de_qualite_echoue_si_seuil_trop_haut(trained):
    cfg = apply_overrides(trained["cfg"], ["evaluation.min_f1=0.99"])
    assert check_gate(trained["metrics"], cfg) != []


@pytest.mark.parametrize("method", ["none", "smote"])
def test_variantes_desequilibre(cfg, method):
    metrics = run(apply_overrides(cfg, [f"imbalance.method={method}"]))
    assert 0 < metrics["f1"] < 1


def test_donnees_preparees_memes_metriques(trained, cfg, tmp_path):
    """Le chemin « prepare -> train » (pipeline DVC, Partie 6) donne les mêmes métriques."""
    save_prepared(cfg, tmp_path / "prepared")
    assert run(cfg, prepared_dir=tmp_path / "prepared") == trained["metrics"]


def test_surcharge_invalide_refusee(cfg):
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["model.name=svm"])
