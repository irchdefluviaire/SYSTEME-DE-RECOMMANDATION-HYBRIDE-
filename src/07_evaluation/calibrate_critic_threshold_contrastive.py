"""
calibrate_critic_threshold_contrastive.py
===========================================================================
Calibration contrastive non supervisee du seuil du critic GraphRAG.

Le script ne demande aucune annotation humaine. Il construit des cas positifs
a partir des sorties fonctionnelles Neo4j deja produites, puis fabrique des
cas contrastifs en associant chaque reponse a un contexte d'une autre requete.

La logique est volontairement prudente : une reponse ne doit pas etre acceptee
quand elle est comparee a un contexte melange. Le seuil recommande minimise
d'abord les fausses acceptations sur ces contextes contrastifs, puis maximise
le F1 pseudo-supervise.

Sorties :
  - critic_contrastive_cases.csv
  - critic_contrastive_threshold_grid.csv
  - critic_contrastive_summary.json
  - critic_contrastive_distributions.png
  - critic_contrastive_threshold_grid.png
===========================================================================
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "05_graphrag"))

from answer_critic import critique_answer  # noqa: E402


DEFAULT_INPUT_DIR = ROOT / "outputs" / "memoire_stats" / "implementation_deep_current"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "evaluation" / "critic_contrastive_calibration"
DEFAULT_FIGURES_DIR = ROOT / "rapport" / "figures" / "generated" / "evaluation"


@dataclass
class EvidenceCase:
    query_id: str
    family: str
    question: str
    answer: str
    context: str


def _parse_list(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [part.strip() for part in text.split(";") if part.strip()]


def _short_items(items: list[str], n: int = 5) -> str:
    return ", ".join(items[:n])


def _build_metier_cases(input_dir: Path, limit: int) -> list[EvidenceCase]:
    path = input_dir / "19_eval_graph_metier_competences.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path).head(limit)
    cases: list[EvidenceCase] = []
    for i, row in df.iterrows():
        metier = str(row["metier"])
        examples = _parse_list(row.get("exemples"))
        n_comp = int(row["n_competences"])
        question = f"Quelles competences sont associees au metier {metier} ?"
        answer = (
            f"Pour le metier {metier}, Neo4j relie {n_comp} competences. "
            f"Les preuves recuperees citent notamment : {_short_items(examples)}."
        )
        context = (
            f"Type: metier_competences\nMetier: {metier}\n"
            f"Nombre de competences: {n_comp}\nCompetences exemples: {_short_items(examples, 8)}"
        )
        cases.append(EvidenceCase(f"metier_{i:03d}", "metier_competences", question, answer, context))
    return cases


def _build_competence_cases(input_dir: Path, limit: int) -> list[EvidenceCase]:
    path = input_dir / "19_eval_graph_competence_offres.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path).head(limit)
    cases: list[EvidenceCase] = []
    for i, row in df.iterrows():
        competence = str(row["competence"])
        examples = _parse_list(row.get("exemples_offres"))
        n_offres = int(row["n_offres"])
        question = f"Quelles offres demandent la competence {competence} ?"
        answer = (
            f"La competence {competence} est reliee a {n_offres} offres dans Neo4j. "
            f"Les offres recuperees incluent : {_short_items(examples)}."
        )
        context = (
            f"Type: competence_offres\nCompetence: {competence}\n"
            f"Nombre d'offres: {n_offres}\nOffres exemples: {_short_items(examples, 8)}"
        )
        cases.append(EvidenceCase(f"competence_{i:03d}", "competence_offres", question, answer, context))
    return cases


def _build_summary_cases(input_dir: Path) -> list[EvidenceCase]:
    path = input_dir / "19_eval_graph_functional_summary.json"
    if not path.exists():
        return []
    summary = json.loads(path.read_text(encoding="utf-8"))
    coverage = summary.get("coverage", {})
    answer = (
        "Le graphe couvre 13 957 offres et 41 298 candidats. "
        f"{coverage.get('n_offres_avec_comp')} offres ont au moins une competence, "
        f"soit {coverage.get('taux_offres_avec_comp', 0):.2%}. "
        f"{coverage.get('n_candidats_avec_comp')} candidats ont au moins une competence, "
        f"soit {coverage.get('taux_candidats_avec_comp', 0):.2%}."
    )
    context = json.dumps(summary, ensure_ascii=False)
    return [
        EvidenceCase(
            "summary_coverage",
            "coverage",
            "Quelle est la couverture du graphe pour les offres et candidats ?",
            answer,
            context,
        )
    ]


def build_evidence_cases(input_dir: Path, per_family: int) -> list[EvidenceCase]:
    cases = []
    cases.extend(_build_metier_cases(input_dir, per_family))
    cases.extend(_build_competence_cases(input_dir, per_family))
    cases.extend(_build_summary_cases(input_dir))
    if len(cases) < 4:
        raise RuntimeError("Pas assez de cas disponibles pour une calibration contrastive.")
    return cases


def score_cases(cases: list[EvidenceCase], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled_idx = np.arange(len(cases))
    while True:
        rng.shuffle(shuffled_idx)
        if np.all(shuffled_idx != np.arange(len(cases))):
            break

    rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        true_critic = critique_answer(case.answer, case.context, min_faithfulness=0.0)
        wrong_context = cases[int(shuffled_idx[i])].context
        wrong_critic = critique_answer(case.answer, wrong_context, min_faithfulness=0.0)

        rows.append(
            {
                "query_id": case.query_id,
                "family": case.family,
                "contrast_type": "true_context",
                "pseudo_label_accept": 1,
                "question": case.question,
                "answer": case.answer,
                "faithfulness": float(true_critic["faithfulness"]),
                "evidence_coverage": float(true_critic["evidence_coverage"]),
                "unsupported_terms": json.dumps(true_critic["unsupported_terms"], ensure_ascii=False),
            }
        )
        rows.append(
            {
                "query_id": case.query_id,
                "family": case.family,
                "contrast_type": "shuffled_context",
                "pseudo_label_accept": 0,
                "question": case.question,
                "answer": case.answer,
                "faithfulness": float(wrong_critic["faithfulness"]),
                "evidence_coverage": float(wrong_critic["evidence_coverage"]),
                "unsupported_terms": json.dumps(wrong_critic["unsupported_terms"], ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def build_threshold_grid(scores: pd.DataFrame) -> pd.DataFrame:
    y_true = scores["pseudo_label_accept"].to_numpy(dtype=int)
    values = scores["faithfulness"].to_numpy(dtype=float)
    thresholds = np.round(np.arange(0.0, max(0.8, float(values.max()) + 0.03), 0.01), 2)
    rows = []
    for threshold in thresholds:
        y_pred = (values >= threshold).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        false_accept_rate = fp / max(int((y_true == 0).sum()), 1)
        false_revise_rate = fn / max(int((y_true == 1).sum()), 1)
        rows.append(
            {
                "threshold": threshold,
                "precision_accept": round(precision, 4),
                "recall_accept": round(recall, 4),
                "f1_accept": round(f1, 4),
                "false_accept": fp,
                "false_revise": fn,
                "false_accept_rate": round(false_accept_rate, 4),
                "false_revise_rate": round(false_revise_rate, 4),
                "true_accept": tp,
                "true_revise": tn,
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(grid: pd.DataFrame, max_false_accept_rate: float) -> dict[str, Any]:
    eligible = grid[grid["false_accept_rate"] <= max_false_accept_rate].copy()
    if eligible.empty:
        eligible = grid.copy()
    eligible = eligible.sort_values(
        ["false_accept_rate", "f1_accept", "recall_accept"],
        ascending=[True, False, False],
    )
    return eligible.iloc[0].to_dict()


def write_figures(scores: pd.DataFrame, grid: pd.DataFrame, output_dir: Path, figures_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dist_path = output_dir / "critic_contrastive_distributions.png"
    grid_path = output_dir / "critic_contrastive_threshold_grid.png"
    report_dist_path = figures_dir / dist_path.name
    report_grid_path = figures_dir / grid_path.name

    true_scores = scores.loc[scores["contrast_type"] == "true_context", "faithfulness"]
    shuffled_scores = scores.loc[scores["contrast_type"] == "shuffled_context", "faithfulness"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bins = np.linspace(0, max(0.8, float(scores["faithfulness"].max()) + 0.05), 22)
    ax.hist(true_scores, bins=bins, alpha=0.72, label="contexte reel", color="#2878b5")
    ax.hist(shuffled_scores, bins=bins, alpha=0.72, label="contexte melange", color="#c44e52")
    ax.set_xlabel("Score faithfulness du critic lexical")
    ax.set_ylabel("Nombre de cas")
    ax.set_title("Calibration contrastive du critic sans annotation")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(dist_path, dpi=180)
    fig.savefig(report_dist_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(grid["threshold"], grid["precision_accept"], label="precision accept", color="#2878b5")
    ax.plot(grid["threshold"], grid["recall_accept"], label="rappel accept", color="#5f9e6e")
    ax.plot(grid["threshold"], grid["f1_accept"], label="F1 accept", color="#8064a2")
    ax.plot(grid["threshold"], grid["false_accept_rate"], label="taux fausse acceptation", color="#c44e52")
    ax.set_xlabel("Seuil applique au score faithfulness")
    ax.set_ylabel("Valeur")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Effet du seuil sur accept/revise")
    ax.legend(ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(grid_path, dpi=180)
    fig.savefig(report_grid_path, dpi=180)
    plt.close(fig)

    return {
        "distribution": str(report_dist_path.relative_to(ROOT / "rapport")),
        "threshold_grid": str(report_grid_path.relative_to(ROOT / "rapport")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--per-family", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-false-accept-rate", type=float, default=0.10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_evidence_cases(args.input_dir, args.per_family)
    scores = score_cases(cases, args.seed)
    grid = build_threshold_grid(scores)
    selected = choose_threshold(grid, args.max_false_accept_rate)
    figures = write_figures(scores, grid, args.output_dir, args.figures_dir)

    true_scores = scores[scores["contrast_type"] == "true_context"]["faithfulness"]
    shuffled_scores = scores[scores["contrast_type"] == "shuffled_context"]["faithfulness"]
    current = grid.loc[np.isclose(grid["threshold"], 0.12)]
    current_row = current.iloc[0].to_dict() if not current.empty else {}

    summary = {
        "method": "contrastive_unsupervised",
        "n_base_cases": len(cases),
        "n_scored_cases": len(scores),
        "families": scores["family"].value_counts().to_dict(),
        "true_context_mean": round(float(true_scores.mean()), 4),
        "true_context_median": round(float(true_scores.median()), 4),
        "shuffled_context_mean": round(float(shuffled_scores.mean()), 4),
        "shuffled_context_median": round(float(shuffled_scores.median()), 4),
        "recommended_threshold": round(float(selected["threshold"]), 4),
        "selection_rule": (
            "minimise false_accept_rate under the configured cap, then maximise F1_accept"
        ),
        "max_false_accept_rate": args.max_false_accept_rate,
        "selected_metrics": selected,
        "current_threshold_0_12": current_row,
        "figures": figures,
        "limits": [
            "pseudo-labels generated by context shuffling, not human labels",
            "answers are deterministic evidence summaries built from graph outputs",
            "the threshold controls lexical grounding, not business correctness",
        ],
    }

    scores.to_csv(args.output_dir / "critic_contrastive_cases.csv", index=False)
    grid.to_csv(args.output_dir / "critic_contrastive_threshold_grid.csv", index=False)
    (args.output_dir / "critic_contrastive_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
