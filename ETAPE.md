# Étape TP4 — Partie 4 « CI/CD pour le ML : GitHub Actions + CML » (version simplifiée du 2026-10-21)

État du dépôt à la **fin du TP4** : le projet du TP3 + un workflow `.github/workflows/ci.yml`
et le générateur de rapport `src/churn/report.py`. **Point de départ du TP5 (DVC).**

## Contenu ajouté par rapport au TP3

| Fichier | Rôle |
|---|---|
| `.github/workflows/ci.yml` | 2 jobs : `test` (pytest) → `train` (entraînement, `report.md`, commentaire CML dans la PR) |
| `src/churn/report.py`, `tests/test_report.py` | rapport Markdown : métriques, matrice de confusion (fournis aux étudiants) |
| `.gitignore` | **provisoire** : `!/data/churn.csv` (données versionnées pour que la CI entraîne ; retirées au TP5) |

## Le workflow en bref

- Déclencheurs : toute `pull_request`, et `push` sur `main` (après une fusion).
- Permissions : `contents: read` et `pull-requests: write` (CML écrit un commentaire dans la PR).
- Actions : `actions/checkout@v7`, `actions/setup-python@v7`, `iterative/setup-cml@v2`
  (versions de `plan/decisions.md`, à revérifier avant la séance).
- Publication : `cml comment create report.md`, seulement pour une pull request.

Les éléments avancés de l'ancienne version (baseline `main`, porte de qualité, matrice,
artefacts entre jobs, commentaire unique mis à jour, concurrency, cache pip, actionlint) sont
traités en « Pour aller plus loin » : solutions dans `../aller_plus_loin/`.
