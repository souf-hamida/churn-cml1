"""Prétraitement : ordre déterministe des colonnes et absence de fuite."""

from __future__ import annotations

import joblib
import pytest

from churn.features import build_preprocessor, feature_columns


def test_ordre_des_colonnes_deterministe(cfg):
    assert feature_columns(cfg) == [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
        "HasCrCard",
        "IsActiveMember",
        "Geography",
        "Gender",
    ]


def test_identifiant_refuse_comme_variable(cfg):
    cfg["features"]["categorical"].append("Surname")
    with pytest.raises(ValueError, match="Fuite"):
        build_preprocessor(cfg)


def test_cible_refusee_comme_variable(cfg):
    cfg["features"]["numeric"].append("Exited")
    with pytest.raises(ValueError, match="Fuite"):
        feature_columns(cfg)


def test_variables_en_sortie_sans_identifiant(trained):
    model = joblib.load(trained["dir"] / "model.joblib")
    out = list(model[0].get_feature_names_out())
    assert "HasCrCard" in out and "IsActiveMember" in out
    assert "Geography_Germany" in out and "Gender_Male" in out
    assert not any(c.startswith(("Surname", "CustomerId", "RowNumber")) for c in out)
    assert len(out) == 13
