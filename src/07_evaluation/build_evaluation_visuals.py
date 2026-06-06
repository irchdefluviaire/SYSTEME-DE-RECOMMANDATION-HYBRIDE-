"""
Figures complementaires pour le chapitre d'evaluation.

Ce script ne recalcule pas les metriques. Il transforme les sorties deja
produites par l'ablation pgvector vs graphe en figures directement exploitables
dans le memoire.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "outputs" / "evaluation" / "pgvector_vs_graph" / "ablation_summary.json"
FIGURES_DIR = ROOT / "rapport" / "figures" / "generated" / "evaluation"


def build_operational_summary() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    requested = int(summary["n_queries_requested"])
    evaluated = int(summary["n_queries_evaluated"])
    errors = int(summary["n_errors"])
    elapsed = float(summary["elapsed_s"])

    labels = ["Requetes demandees", "Requetes evaluees", "Erreurs Neo4j"]
    values = [requested, evaluated, errors]
    colors = ["#1D4ED8", "#15803D", "#B91C1C"]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.58)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.025,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    error_rate = errors / requested if requested else 0.0
    ax.set_title("Stabilite operationnelle de l'evaluation pgvector + graphe", fontsize=13)
    ax.set_ylabel("Nombre de requetes")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.5,
        -0.20,
        f"Temps d'execution : {elapsed:.1f} s | Taux d'erreur : {error_rate:.1%}",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=10,
    )
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / "evaluation_operational_summary.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Sortie d'ablation introuvable : {SUMMARY_PATH}")
    output = build_operational_summary()
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
