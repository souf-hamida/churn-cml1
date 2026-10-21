"""Contrat des données : schéma, qualité, nettoyage, découpage."""

from __future__ import annotations

import pandas as pd
import pytest

from churn.data import EXPECTED_SCHEMA, DataValidationError, clean, load_raw, split, validate


@pytest.fixture
def raw(cfg):
    return load_raw(cfg["data"]["csv_path"])


def test_schema_et_dimensions(raw):
    assert raw.shape == (10_000, 14)
    assert list(raw.columns) == list(EXPECTED_SCHEMA)


def test_controles_qualite_ok(raw):
    report = validate(raw)
    assert report["n_rows"] == 10_000
    assert report["target_rate"] == pytest.approx(0.2037)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Age", -5, "Age hors"),
        ("Geography", "Italy", "inconnue"),
        ("Exited", 2, "non binaire"),
        ("Balance", -1.0, "Balance"),
    ],
)
def test_controles_qualite_detectent_les_erreurs(raw, column, value, message):
    bad = raw.copy()
    bad.loc[0, column] = value
    with pytest.raises(DataValidationError, match=message):
        validate(bad)


def test_valeur_manquante_detectee(raw):
    bad = raw.copy()
    bad["CreditScore"] = bad["CreditScore"].astype("float64")
    bad.loc[3, "CreditScore"] = float("nan")
    with pytest.raises(DataValidationError):
        validate(bad)


def test_colonne_manquante_detectee(raw):
    with pytest.raises(DataValidationError, match="colonnes manquantes"):
        validate(raw.drop(columns=["Age"]))


def test_nettoyage_retire_identifiants_et_ages_extremes(raw, cfg):
    cleaned = clean(raw, cfg)
    for col in cfg["data"]["drop_columns"]:
        assert col not in cleaned.columns
    assert len(cleaned) == 10_000 - 12
    assert cleaned["Age"].max() <= 80


def test_split_stratifie_et_reproductible(raw, cfg):
    cleaned = clean(raw, cfg)
    X_tr, X_te, y_tr, y_te = split(cleaned, cfg)
    _, X_te2, _, _ = split(cleaned, cfg)
    assert len(X_te) == 1998
    assert abs(y_tr.mean() - y_te.mean()) < 0.002
    pd.testing.assert_frame_equal(X_te, X_te2)
    assert set(X_tr.index).isdisjoint(X_te.index)
