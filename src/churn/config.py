"""Lecture de la configuration YAML (config/train.yaml ou params.yaml).

Le schéma est unique pour tout le module ; ``--set cle.sous_cle=valeur`` permet de
surcharger une valeur sans modifier le fichier (comme ``dvc exp run -S`` en Partie 6).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = ("data", "split", "features", "model", "imbalance")
MODELS = ("logistic_regression", "random_forest")
IMBALANCE_METHODS = ("none", "class_weight", "smote")


def load_config(path: str | Path = "config/train.yaml") -> dict[str, Any]:
    """Lit le fichier YAML et vérifie les sections obligatoires."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    check_config(cfg)
    return cfg


def check_config(cfg: dict[str, Any]) -> None:
    """Contrôles simples : on préfère une erreur claire à un entraînement faux."""
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"Sections manquantes dans la configuration : {missing}")
    if cfg["data"].get("source", "csv") != "csv":
        raise ValueError("data.source : seule la valeur 'csv' est prise en charge")
    if cfg["model"]["name"] not in MODELS:
        raise ValueError(f"model.name doit valoir {MODELS}, reçu {cfg['model']['name']!r}")
    if cfg["imbalance"]["method"] not in IMBALANCE_METHODS:
        raise ValueError(
            f"imbalance.method doit valoir {IMBALANCE_METHODS}, reçu {cfg['imbalance']['method']!r}"
        )
    if not 0 < float(cfg["split"]["test_size"]) < 1:
        raise ValueError("split.test_size doit être strictement entre 0 et 1")


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Applique des surcharges ``a.b=valeur`` (valeur lue comme du YAML : 0.1, true, [..])."""
    new = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Surcharge invalide {item!r} : forme attendue cle.sous_cle=valeur")
        key, raw = item.split("=", 1)
        node = new
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    check_config(new)
    return new


def seed_of(cfg: dict[str, Any]) -> int:
    """Graine unique du projet (split, modèle, SMOTE)."""
    return int(cfg["split"]["random_state"])
