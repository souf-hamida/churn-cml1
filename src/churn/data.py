"""Données : chargement, contrôles qualité (fail-fast), nettoyage et découpage train/test.

Usage en ligne de commande (étape « prepare », réutilisée par DVC en Partie 6) :

    python -m churn.data --config config/train.yaml --out-dir data/prepared
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import apply_overrides, load_config, seed_of

# Schéma attendu du fichier brut : colonne -> nature ("int", "float", "str")
EXPECTED_SCHEMA: dict[str, str] = {
    "RowNumber": "int",
    "CustomerId": "int",
    "Surname": "str",
    "CreditScore": "int",
    "Geography": "str",
    "Gender": "str",
    "Age": "int",
    "Tenure": "int",
    "Balance": "float",
    "NumOfProducts": "int",
    "HasCrCard": "int",
    "IsActiveMember": "int",
    "EstimatedSalary": "float",
    "Exited": "int",
}
GEOGRAPHIES = ("France", "Germany", "Spain")
GENDERS = ("Female", "Male")


class DataValidationError(ValueError):
    """Les données ne respectent pas le contrat : on arrête tout (fail-fast)."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Données invalides :\n- " + "\n- ".join(problems))


def file_sha256(path: str | Path) -> str:
    """Empreinte du fichier de données : prouve « quelles données » ont produit le modèle."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_raw(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _kind_ok(series: pd.Series, kind: str) -> bool:
    if kind == "str":
        return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
    if kind == "int":
        return pd.api.types.is_integer_dtype(series)
    return pd.api.types.is_numeric_dtype(series)


def validate(df: pd.DataFrame, target: str = "Exited") -> dict[str, Any]:
    """Contrôles qualité : schéma, valeurs manquantes, domaines, plages, doublons.

    Lève DataValidationError avec la liste de TOUS les problèmes trouvés ;
    renvoie sinon un petit rapport de qualité (inclus dans report.json).
    """
    problems: list[str] = []
    missing = [c for c in EXPECTED_SCHEMA if c not in df.columns]
    if missing:
        raise DataValidationError([f"colonnes manquantes : {missing}"])

    for col, kind in EXPECTED_SCHEMA.items():
        if not _kind_ok(df[col], kind):
            problems.append(f"type inattendu pour {col} : {df[col].dtype} (attendu {kind})")
    n_missing = df[list(EXPECTED_SCHEMA)].isna().sum()
    for col, n in n_missing[n_missing > 0].items():
        problems.append(f"{n} valeur(s) manquante(s) dans {col}")
    if problems:  # inutile d'aller plus loin si les types sont faux
        raise DataValidationError(problems)

    if not set(df[target].unique()) <= {0, 1}:
        problems.append(f"cible {target} non binaire : {sorted(df[target].unique())}")
    for col in ("HasCrCard", "IsActiveMember"):
        if not set(df[col].unique()) <= {0, 1}:
            problems.append(f"{col} doit valoir 0 ou 1")
    rules = {
        "Age": (0, 120),
        "CreditScore": (0, 1000),
        "Tenure": (0, 100),
        "NumOfProducts": (1, 10),
    }
    for col, (low, high) in rules.items():
        bad = int(((df[col] < low) | (df[col] > high)).sum())
        if bad:
            problems.append(f"{bad} valeur(s) de {col} hors de [{low}, {high}]")
    for col in ("Balance", "EstimatedSalary"):
        bad = int((df[col] < 0).sum())
        if bad:
            problems.append(f"{bad} valeur(s) négative(s) dans {col}")
    for col, allowed in (("Geography", GEOGRAPHIES), ("Gender", GENDERS)):
        unknown = sorted(set(df[col].unique()) - set(allowed))
        if unknown:
            problems.append(f"modalité(s) inconnue(s) dans {col} : {unknown}")
    n_dup = int(df["CustomerId"].duplicated().sum())
    if n_dup:
        problems.append(f"{n_dup} client(s) en double (CustomerId)")
    if problems:
        raise DataValidationError(problems)

    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target_rate": round(float(df[target].mean()), 4),
        "n_missing": 0,
        "n_duplicates": 0,
        "age_max": int(df["Age"].max()),
    }


def clean(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Retire les identifiants (anti-fuite) et applique le nettoyage simple de la config."""
    d = cfg["data"]
    out = df.drop(columns=[c for c in d.get("drop_columns", []) if c in df.columns])
    if d.get("max_age") is not None:
        out = out[out["Age"] <= int(d["max_age"])]
    return out.reset_index(drop=True)


def split(df: pd.DataFrame, cfg: dict[str, Any]):
    """Découpage reproductible (graine) et stratifié (même taux de départs des deux côtés)."""
    target = cfg["data"]["target"]
    y = df[target]
    X = df.drop(columns=[target])
    s = cfg["split"]
    return train_test_split(
        X,
        y,
        test_size=float(s["test_size"]),
        random_state=seed_of(cfg),
        stratify=y if s.get("stratify", True) else None,
    )


def load_and_split(cfg: dict[str, Any]):
    """Chaîne complète : lecture -> contrôles -> nettoyage -> split.

    Renvoie (X_train, X_test, y_train, y_test, data_info).
    """
    path = Path(cfg["data"]["csv_path"])
    raw = load_raw(path)
    quality = validate(raw, cfg["data"]["target"])
    cleaned = clean(raw, cfg)
    X_train, X_test, y_train, y_test = split(cleaned, cfg)
    info = {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "quality": quality,
        "n_rows_clean": int(len(cleaned)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return X_train, X_test, y_train, y_test, info


def save_prepared(cfg: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Écrit train.csv et test.csv (cible incluse) et data_report.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, info = load_and_split(cfg)
    X_train.assign(**{y_train.name: y_train}).to_csv(
        out / "train.csv", index=False, lineterminator="\n"
    )
    X_test.assign(**{y_test.name: y_test}).to_csv(
        out / "test.csv", index=False, lineterminator="\n"
    )
    (out / "data_report.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8", newline="\n"
    )
    return info


def load_prepared(prepared_dir: str | Path, cfg: dict[str, Any]):
    """Relit train.csv / test.csv produits par save_prepared."""
    target = cfg["data"]["target"]
    d = Path(prepared_dir)
    train = pd.read_csv(d / "train.csv")
    test = pd.read_csv(d / "test.csv")
    info = json.loads((d / "data_report.json").read_text(encoding="utf-8"))
    return (
        train.drop(columns=[target]),
        test.drop(columns=[target]),
        train[target],
        test[target],
        info,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Contrôle, nettoie et découpe les données churn.")
    p.add_argument("--config", default="config/train.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--out-dir", default="data/prepared")
    args = p.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.overrides)
    info = save_prepared(cfg, args.out_dir)
    print(
        f"OK données : {info['quality']['n_rows']} lignes brutes, "
        f"{info['n_train']} train / {info['n_test']} test -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
