# churn — projet ML reproductible (fil rouge MLOps)

Prédire si un client quitte la banque (`Exited` = 1) à partir de 10 variables
(10 000 clients, 20,4 % de départs).

## Structure

```
config/train.yaml        configuration unique (= schéma de params.yaml en Partie 6)
data/churn.csv           données (ignorées par Git sauf au TP4 ; DVC à partir du TP5)
src/churn/               paquet installable
  config.py              lecture YAML + surcharges --set cle=valeur
  data.py                chargement, contrôles qualité (fail-fast), nettoyage, split
  features.py            ColumnTransformer (numériques, binaires, one-hot)
  train.py               Pipeline prétraitement + modèle, CLI d'entraînement
  evaluate.py            métriques sur le TEST, matrice de confusion, report.json
  report.py              rapport Markdown pour la CI (TP4)
api/main.py              API FastAPI d'inférence (TP3)
Dockerfile, Dockerfile.api, compose.yaml, .dockerignore     TP3
tests/                   pytest : données, fuite, reproductibilité, seuil de qualité
.github/workflows/ci.yml tests, entraînement, rapport CML (TP4)
```

## Installation (Python 3.11 ou 3.12)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
pytest -q
```

## Commandes (à lancer depuis la racine du dépôt)

| Commande | Effet |
|---|---|
| `python -m churn.train --config config/train.yaml` | entraîne et écrit `artifacts/` : `model.joblib`, `metrics.json`, `confusion_matrix.png`, `report.json` |
| `python -m churn.train --set model.C=0.1 --artifacts-dir runs/C-0.1` | même chose avec une surcharge de configuration |
| `python -m churn.train --set imbalance.method=smote` | gestion du déséquilibre par SMOTE (au lieu de `class_weight`) |
| `python -m churn.data --out-dir data/prepared` | contrôles + nettoyage + split → `train.csv`, `test.csv`, `data_report.json` |
| `python -m churn.train --prepared-dir data/prepared --skip-eval` | entraîne à partir des données préparées (étape DVC « train ») |
| `python -m churn.evaluate --prepared-dir data/prepared` | évalue `model.joblib` sur le test (étape DVC « evaluate ») |
| `python -m churn.evaluate --check-gate` | porte de qualité : code 1 si `f1 < evaluation.min_f1` ou `roc_auc < evaluation.min_roc_auc` |
| `python -m churn.report --current artifacts --out report.md` | rapport Markdown (CI) |
| `uvicorn api.main:app --port 8000` | API : `GET /health`, `POST /predict` |

## Résultats de référence

Baseline : régression logistique, `class_weight="balanced"`, graine 42, test de 1 998 clients :
accuracy 0,7052 · précision 0,3753 · rappel 0,6732 · F1 0,4820 · ROC AUC 0,7641.
