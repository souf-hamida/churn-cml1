"""Prétraitement : un ColumnTransformer placé DANS le Pipeline (ajusté sur le train seulement).

- numériques : imputation médiane + standardisation ;
- binaires (HasCrCard, IsActiveMember) : conservées telles quelles (0/1) ;
- catégorielles (Geography, Gender) : one-hot ; une modalité inconnue donne des zéros
  (handle_unknown="ignore", sans drop : aucune ambiguïté avec une modalité supprimée).

L'ordre des colonnes est celui de la configuration : il est donc déterministe.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def feature_columns(cfg: dict[str, Any]) -> list[str]:
    """Colonnes d'entrée du modèle, dans un ordre fixe ; refuse toute colonne identifiant."""
    f = cfg["features"]
    cols = list(f["numeric"]) + list(f.get("binary", [])) + list(f["categorical"])
    forbidden = set(cfg["data"].get("drop_columns", [])) | {cfg["data"]["target"]}
    leak = sorted(set(cols) & forbidden)
    if leak:
        raise ValueError(f"Fuite de données : {leak} ne doivent pas être des variables")
    if len(cols) != len(set(cols)):
        raise ValueError("Une colonne apparaît dans deux familles de variables")
    return cols


def build_preprocessor(cfg: dict[str, Any]) -> ColumnTransformer:
    feature_columns(cfg)  # contrôle anti-fuite
    f = cfg["features"]
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = [("num", numeric, list(f["numeric"]))]
    if f.get("binary"):
        transformers.append(("bin", "passthrough", list(f["binary"])))
    transformers.append(("cat", categorical, list(f["categorical"])))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
