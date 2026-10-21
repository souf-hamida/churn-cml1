"""Rapport Markdown de la CI (TP4)."""

from __future__ import annotations

import json
import shutil

from churn.report import build_report


def test_rapport_sans_baseline(trained):
    text = build_report(trained["dir"])
    assert "| f1 |" in text
    assert "Baseline indisponible" in text


def test_rapport_avec_baseline_et_matrice(trained, tmp_path):
    base = tmp_path / "baseline"
    shutil.copytree(trained["dir"], base)
    m = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
    m["f1"] = round(m["f1"] - 0.01, 4)
    (base / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    matrix = tmp_path / "matrix"
    shutil.copytree(trained["dir"], matrix / "matrix-C-1.0")
    text = build_report(trained["dir"], base, matrix, image="./artifacts/confusion_matrix.png")
    assert "main (baseline)" in text and "+0.0100" in text and "amélioration" in text
    assert "### Matrice d'hyperparamètres" in text and "matrix-C-1.0" in text
    assert "![Matrice de confusion](./artifacts/confusion_matrix.png)" in text
