"""Rapport Markdown pour la CI (TP4) : métriques de la PR, comparaison avec la baseline
(main ré-entraînée dans la CI), matrice de confusion et matrice d'hyperparamètres.

    python -m churn.report --current artifacts --baseline baseline/artifacts \
        --matrix-dir matrix --out report.md

Chaque dossier de run contient metrics.json et report.json (écrits par churn.train).
Le rapport est ensuite publié par CML (``cml comment update report.md``) ou ajouté au
résumé du job (``$GITHUB_STEP_SUMMARY``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def read_run(run_dir: str | Path) -> dict[str, Any] | None:
    """Lit metrics.json (+ report.json s'il existe) ; None si le run est absent."""
    d = Path(run_dir)
    metrics_path = d / "metrics.json"
    if not metrics_path.is_file():
        return None
    run: dict[str, Any] = {"metrics": json.loads(metrics_path.read_text(encoding="utf-8"))}
    report_path = d / "report.json"
    if report_path.is_file():
        run["report"] = json.loads(report_path.read_text(encoding="utf-8"))
    return run


def _fmt_delta(delta: float) -> str:
    if abs(delta) < 5e-5:
        return "="
    return f"{delta:+.4f}"


def metrics_table(current: dict[str, float], baseline: dict[str, float] | None) -> list[str]:
    if baseline is None:
        lines = ["| Métrique | PR |", "|---|---:|"]
        lines += [f"| {k} | {current[k]:.4f} |" for k in METRICS]
        return lines
    lines = ["| Métrique | main (baseline) | PR | Écart |", "|---|---:|---:|---:|"]
    for k in METRICS:
        delta = _fmt_delta(current[k] - baseline[k])
        lines.append(f"| {k} | {baseline[k]:.4f} | {current[k]:.4f} | {delta} |")
    return lines


def _describe(report: dict[str, Any] | None) -> str:
    if not report:
        return "configuration inconnue"
    cfg = report["config"]
    m = cfg["model"]
    hp = (
        f"C={m.get('C')}"
        if m["name"] == "logistic_regression"
        else (f"n_estimators={m.get('n_estimators')}, max_depth={m.get('max_depth')}")
    )
    return f"`{m['name']}` ({hp}), déséquilibre `{cfg['imbalance']['method']}`"


def matrix_table(matrix_dir: str | Path) -> list[str]:
    runs = []
    for d in sorted(Path(matrix_dir).iterdir()):
        run = read_run(d) if d.is_dir() else None
        if run:
            runs.append((d.name, run))
    if not runs:
        return []
    lines = ["| Expérience | Configuration | f1 | recall | roc_auc |", "|---|---|---:|---:|---:|"]
    best = max(runs, key=lambda r: r[1]["metrics"]["f1"])[0]
    for name, run in runs:
        m = run["metrics"]
        label = f"**{name}**" if name == best else name
        lines.append(
            f"| {label} | {_describe(run.get('report'))} | {m['f1']:.4f} | "
            f"{m['recall']:.4f} | {m['roc_auc']:.4f} |"
        )
    lines.append("")
    lines.append(f"Meilleur F1 : **{best}** (en gras).")
    return lines


def build_report(
    current_dir: str | Path,
    baseline_dir: str | Path | None = None,
    matrix_dir: str | Path | None = None,
    image: str | None = None,
) -> str:
    current = read_run(current_dir)
    if current is None:
        raise FileNotFoundError(f"metrics.json introuvable dans {current_dir}")
    baseline = read_run(baseline_dir) if baseline_dir else None
    rep = current.get("report", {})
    commit = (rep.get("git_commit") or "inconnu")[:7]

    lines = ["## Rapport d'entraînement — churn", ""]
    lines.append(
        f"Commit `{commit}` · modèle {_describe(rep)} · jeu de test : "
        f"{rep.get('data', {}).get('n_test', '?')} clients"
    )
    lines.append("")
    lines.append("### Métriques sur le jeu de test (classe positive : le client part)")
    lines.append("")
    lines += metrics_table(current["metrics"], baseline["metrics"] if baseline else None)
    lines.append("")
    if baseline is None:
        lines.append("_Baseline indisponible (premier run, ou `main` non entraînable)._")
    else:
        delta = current["metrics"]["f1"] - baseline["metrics"]["f1"]
        verdict = "amélioration" if delta > 0 else ("dégradation" if delta < 0 else "aucun écart")
        lines.append(f"Écart de F1 par rapport à `main` : **{_fmt_delta(delta)}** ({verdict}).")
    cm = rep.get("confusion_matrix")
    if cm:
        counts = f"TN={cm['tn']} · FP={cm['fp']} · FN={cm['fn']} · TP={cm['tp']}"
        lines += ["", f"Matrice de confusion : {counts}"]
    if image:
        lines += ["", f"![Matrice de confusion]({image})"]
    if matrix_dir and Path(matrix_dir).is_dir():
        table = matrix_table(matrix_dir)
        if table:
            lines += ["", "### Matrice d'hyperparamètres", ""] + table
    if rep.get("data", {}).get("sha256"):
        lines += [
            "",
            f"<sub>Données : sha256 `{rep['data']['sha256'][:12]}…` · "
            f"Python {rep['versions']['python']} · "
            f"scikit-learn {rep['versions']['scikit-learn']}</sub>",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Génère report.md pour la CI.")
    p.add_argument("--current", default="artifacts", help="dossier du run de la PR")
    p.add_argument("--baseline", default=None, help="dossier du run de main")
    p.add_argument("--matrix-dir", default=None, help="un sous-dossier par expérience")
    p.add_argument("--image", default=None, help="chemin de l'image à insérer")
    p.add_argument("--out", default="report.md")
    args = p.parse_args(argv)
    text = build_report(args.current, args.baseline, args.matrix_dir, args.image)
    Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    print(f"OK rapport -> {args.out}")


if __name__ == "__main__":
    main()
