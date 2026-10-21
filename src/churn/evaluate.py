"""Évaluation sur le jeu de TEST uniquement et écriture des artefacts.

    python -m churn.evaluate --config config/train.yaml          # ré-évalue model.joblib
    python -m churn.evaluate --config config/train.yaml --check-gate   # porte de qualité (CI)

Artefacts : metrics.json (5 métriques), confusion_matrix.png (fond clair, effectifs entiers),
report.json (configuration, graine, versions, empreinte sha256 des données, commit Git).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # pas d'écran : terminal, Docker, CI
import imblearn  # noqa: E402
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import churn  # noqa: E402
from churn.config import apply_overrides, load_config, seed_of  # noqa: E402
from churn.data import load_and_split, load_prepared  # noqa: E402
from churn.features import feature_columns  # noqa: E402

LABELS = ["Reste (0)", "Part (1)"]


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Classe positive = 1 (le client part). ROC AUC calculée sur les probabilités."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def save_confusion_matrix(y_true, y_pred, path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150, facecolor="white")
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=LABELS,
        cmap="Blues",
        colorbar=False,
        values_format="d",
        ax=ax,
    )
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def git_commit() -> str | None:
    """Commit Git courant (None hors dépôt ou si Git est absent, par ex. dans l'image)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "churn": churn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "imbalanced-learn": imblearn.__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
    }


def evaluate_and_save(
    model,
    X_test,
    y_test,
    cfg: dict[str, Any],
    out_dir: str | Path,
    data_info: dict[str, Any] | None = None,
    train_seconds: float | None = None,
) -> dict[str, float]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X = X_test[feature_columns(cfg)]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    # newline="\n" : fichier identique (même sha256) sous Windows, Linux et dans Docker
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    m = cfg["model"]
    title = f"{m['name']} · {cfg['imbalance']['method']} · test (n={len(y_test)})"
    save_confusion_matrix(y_test, y_pred, out / "confusion_matrix.png", title)

    cm = confusion_matrix(y_test, y_pred)
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed_of(cfg),
        "config": cfg,
        "data": data_info,
        "git_commit": git_commit(),
        "versions": versions(),
        "platform": f"{platform.system()} {platform.machine()}",
        "train_seconds": train_seconds,
        "features_in": feature_columns(cfg),
        "features_out": [str(c) for c in model[0].get_feature_names_out()],
        "metrics": metrics,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return metrics


def check_gate(metrics: dict[str, float], cfg: dict[str, Any]) -> list[str]:
    """Renvoie la liste des seuils non atteints (liste vide = porte franchie)."""
    ev = cfg.get("evaluation", {})
    failures = []
    for key, name in (("min_f1", "f1"), ("min_roc_auc", "roc_auc")):
        if key in ev and metrics[name] < float(ev[key]):
            failures.append(f"{name} = {metrics[name]} < {ev[key]}")
    return failures


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Évalue model.joblib sur le jeu de test.")
    p.add_argument("--config", default="config/train.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--artifacts-dir", default=None)
    p.add_argument("--prepared-dir", default=None)
    p.add_argument(
        "--check-gate",
        action="store_true",
        help="lit metrics.json et échoue (code 1) sous les seuils evaluation.*",
    )
    args = p.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.overrides)
    out = Path(args.artifacts_dir or cfg.get("artifacts_dir", "artifacts"))

    if args.check_gate:
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        failures = check_gate(metrics, cfg)
        if failures:
            print("Porte de qualité NON franchie :", "; ".join(failures))
            sys.exit(1)
        print("Porte de qualité franchie :", metrics)
        return

    if args.prepared_dir:
        _, X_test, _, y_test, info = load_prepared(args.prepared_dir, cfg)
    else:
        _, X_test, _, y_test, info = load_and_split(cfg)
    model = joblib.load(out / "model.joblib")
    metrics = evaluate_and_save(model, X_test, y_test, cfg, out, data_info=info)
    print("OK:", metrics)


if __name__ == "__main__":
    main()
