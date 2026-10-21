"""Entraînement : Pipeline (prétraitement + modèle), graine unique, option déséquilibre.

    python -m churn.train --config config/train.yaml
    python -m churn.train --config config/train.yaml --set model.C=0.1 --artifacts-dir out/C-0.1
    python -m churn.train --prepared-dir data/prepared --skip-eval   # étape DVC « train »

Artefacts écrits dans artifacts_dir : model.joblib, metrics.json, confusion_matrix.png,
report.json (voir churn.evaluate).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.config import apply_overrides, load_config, seed_of
from churn.data import load_and_split, load_prepared
from churn.evaluate import evaluate_and_save
from churn.features import build_preprocessor, feature_columns


def build_model(cfg: dict[str, Any]):
    m = cfg["model"]
    class_weight = "balanced" if cfg["imbalance"]["method"] == "class_weight" else None
    if m["name"] == "logistic_regression":
        return LogisticRegression(
            C=float(m.get("C", 1.0)),
            max_iter=int(m.get("max_iter", 1000)),
            class_weight=class_weight,
            random_state=seed_of(cfg),
        )
    return RandomForestClassifier(
        n_estimators=int(m.get("n_estimators", 300)),
        max_depth=None if m.get("max_depth") is None else int(m["max_depth"]),
        class_weight=class_weight,
        random_state=seed_of(cfg),
        n_jobs=-1,
    )


def build_pipeline(cfg: dict[str, Any]):
    """Un seul objet à sauvegarder : en production, pipe.predict(clients) suffit."""
    steps = [("preprocess", build_preprocessor(cfg))]
    if cfg["imbalance"]["method"] == "smote":
        # imblearn applique SMOTE pendant fit() seulement : le test n'est jamais rééchantillonné
        steps.append(
            (
                "smote",
                SMOTE(
                    sampling_strategy=cfg["imbalance"].get("smote_sampling_strategy", "auto"),
                    random_state=seed_of(cfg),
                ),
            )
        )
        steps.append(("model", build_model(cfg)))
        return ImbPipeline(steps)
    steps.append(("model", build_model(cfg)))
    return Pipeline(steps)


def fit(cfg: dict[str, Any], X_train, y_train):
    pipe = build_pipeline(cfg)
    pipe.fit(X_train[feature_columns(cfg)], y_train)
    return pipe


def run(
    cfg: dict[str, Any],
    prepared_dir: str | Path | None = None,
    skip_eval: bool = False,
) -> dict[str, Any] | None:
    """Entraîne, sauvegarde model.joblib puis (sauf skip_eval) évalue sur le TEST."""
    out = Path(cfg.get("artifacts_dir", "artifacts"))
    out.mkdir(parents=True, exist_ok=True)
    if prepared_dir is None:
        X_train, X_test, y_train, y_test, data_info = load_and_split(cfg)
    else:
        X_train, X_test, y_train, y_test, data_info = load_prepared(prepared_dir, cfg)

    start = time.perf_counter()
    pipe = fit(cfg, X_train, y_train)
    train_seconds = round(time.perf_counter() - start, 3)
    joblib.dump(pipe, out / "model.joblib")
    if skip_eval:
        return None
    return evaluate_and_save(
        pipe, X_test, y_test, cfg, out, data_info=data_info, train_seconds=train_seconds
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Entraîne le modèle churn et écrit les artefacts.")
    p.add_argument("--config", default="config/train.yaml")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="surcharge cle.sous_cle=valeur (répétable)",
    )
    p.add_argument("--artifacts-dir", default=None, help="remplace artifacts_dir")
    p.add_argument("--prepared-dir", default=None, help="lit train.csv/test.csv préparés")
    p.add_argument("--skip-eval", action="store_true", help="n'écrit que model.joblib")
    args = p.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.overrides)
    if args.artifacts_dir:
        cfg["artifacts_dir"] = args.artifacts_dir
    metrics = run(cfg, prepared_dir=args.prepared_dir, skip_eval=args.skip_eval)
    out = cfg.get("artifacts_dir", "artifacts")
    if metrics is None:
        print(f"OK modèle -> {out}/model.joblib")
    else:
        print("OK:", metrics)
        print(f"Artefacts -> {out}/")


if __name__ == "__main__":
    main()
