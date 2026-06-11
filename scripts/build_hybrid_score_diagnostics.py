from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DETAILS = ROOT / "outputs" / "evaluation" / "hybrid_weight_optimization" / "optimized_ranking_details_ndcg.csv"
WEIGHTS = ROOT / "outputs" / "evaluation" / "hybrid_weight_optimization" / "multi_objective_weight_comparison.csv"
BEST = ROOT / "outputs" / "evaluation" / "hybrid_weight_optimization" / "best_weights_ndcg.json"
ABLATION = ROOT / "outputs" / "evaluation" / "pgvector_vs_graph" / "ablation_metrics.csv"
OUT_DIR = ROOT / "outputs" / "memoire_stats" / "implementation_deep_current"
FIG_DIR = ROOT / "rapport" / "figures" / "generated" / "implementation_deep_current"


def describe_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in [
        ("score_sem", "Score pgvector"),
        ("taux_match", "Score Neo4j"),
        ("score_hybride_optuna", "Score hybride Optuna"),
    ]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append(
            {
                "score": label,
                "n": int(series.size),
                "mean": round(float(series.mean()), 6),
                "std": round(float(series.std()), 6),
                "min": round(float(series.min()), 6),
                "p25": round(float(series.quantile(0.25)), 6),
                "median": round(float(series.median()), 6),
                "p75": round(float(series.quantile(0.75)), 6),
                "max": round(float(series.max()), 6),
                "zero_pct": round(float((series == 0).mean() * 100), 2),
            }
        )
    return pd.DataFrame(rows)


def build_distribution_figure(df: pd.DataFrame, summary: pd.DataFrame) -> Path:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    bins = [i / 20 for i in range(21)]
    ax.hist(
        df["score_sem"],
        bins=bins,
        density=True,
        alpha=0.55,
        label="Score pgvector (similarite semantique)",
        color="#2563eb",
        edgecolor="white",
    )
    ax.hist(
        df["taux_match"],
        bins=bins,
        density=True,
        alpha=0.55,
        label="Score Neo4j (couverture des competences)",
        color="#16a34a",
        edgecolor="white",
    )

    med_pg = float(summary.loc[summary["score"] == "Score pgvector", "median"].iloc[0])
    med_neo = float(summary.loc[summary["score"] == "Score Neo4j", "median"].iloc[0])
    ax.axvline(med_pg, color="#1d4ed8", linestyle="--", linewidth=1.4)
    ax.axvline(med_neo, color="#15803d", linestyle="--", linewidth=1.4)
    ax.text(med_pg + 0.015, ax.get_ylim()[1] * 0.88, f"mediane pgvector={med_pg:.2f}", color="#1d4ed8")
    ax.text(med_neo + 0.015, ax.get_ylim()[1] * 0.78, f"mediane Neo4j={med_neo:.2f}", color="#15803d")

    ax.set_title("Distribution comparee des deux scores utilises par le moteur hybride")
    ax.set_xlabel("Score normalise entre 0 et 1")
    ax.set_ylabel("Densite")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", ncol=1, frameon=True)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "13_distribution_scores_pgvector_neo4j.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def build_reranking_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rank_shift_optuna"] = out["rank_pgvector"] - out["rank_optuna"]
    shift = out["rank_shift_optuna"]
    return pd.DataFrame(
        [
            {
                "n_pairs": int(len(out)),
                "n_candidates": int(out["candidat_id"].nunique()),
                "rank_pgvector_mean": round(float(out["rank_pgvector"].mean()), 6),
                "rank_optuna_mean": round(float(out["rank_optuna"].mean()), 6),
                "rank_shift_mean": round(float(shift.mean()), 6),
                "rank_shift_median": round(float(shift.median()), 6),
                "rank_shift_p75": round(float(shift.quantile(0.75)), 6),
                "rank_shift_min": round(float(shift.min()), 6),
                "rank_shift_max": round(float(shift.max()), 6),
                "gain_pct": round(float((shift > 0).mean() * 100), 2),
                "loss_pct": round(float((shift < 0).mean() * 100), 2),
                "same_pct": round(float((shift == 0).mean() * 100), 2),
            }
        ]
    )


def build_reranking_figure(df: pd.DataFrame, ablation: pd.DataFrame) -> Path:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})

    metrics = ["precision@1", "ndcg@5", "ndcg@10", "recall@10"]
    labels = ["P@1", "NDCG@5", "NDCG@10", "Recall@10"]
    pg = ablation.loc[ablation["system"] == "pgvector_only", metrics].iloc[0].astype(float).to_numpy()
    graph = ablation.loc[ablation["system"] == "pgvector_plus_graph", metrics].iloc[0].astype(float).to_numpy()
    x = range(len(metrics))
    width = 0.36
    axes[0].bar([i - width / 2 for i in x], pg, width, label="pgvector seul", color="#2563eb")
    axes[0].bar([i + width / 2 for i in x], graph, width, label="pgvector + Neo4j", color="#16a34a")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score moyen")
    axes[0].set_title("Gain de ranking apres enrichissement graphe")
    axes[0].legend(frameon=True)
    for i, (base, new) in enumerate(zip(pg, graph)):
        axes[0].text(i, min(new + 0.035, 0.97), f"+{new - base:.3f}", ha="center", fontsize=9, color="#14532d")

    shift = df["rank_pgvector"] - df["rank_optuna"]
    bins = list(range(int(shift.min()) - 1, int(shift.max()) + 2))
    axes[1].hist(shift, bins=bins, color="#0f766e", alpha=0.82, edgecolor="white")
    axes[1].axvline(0, color="#991b1b", linestyle="--", linewidth=1.4)
    axes[1].set_title("Deplacement de rang apres reranking Optuna")
    axes[1].set_xlabel("Rang gagne (+) ou perdu (-)")
    axes[1].set_ylabel("Nombre de couples")
    axes[1].text(0.02, 0.95, "0 = rang inchange", transform=axes[1].transAxes, va="top", color="#991b1b")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "13_effet_reranking_pgvector_neo4j.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DETAILS)
    for col in ("score_sem", "taux_match", "score_hybride_optuna"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = describe_scores(df)
    summary.to_csv(OUT_DIR / "13_distribution_scores_pgvector_neo4j.csv", index=False)

    weights = pd.read_csv(WEIGHTS)
    weights.to_csv(OUT_DIR / "13_optuna_poids_score_hybride.csv", index=False)
    ablation = pd.read_csv(ABLATION)
    ablation.to_csv(OUT_DIR / "13_ablation_pgvector_neo4j_reranking.csv", index=False)

    best = json.loads(BEST.read_text(encoding="utf-8"))
    (OUT_DIR / "13_score_hybride_optuna_summary.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fig = build_distribution_figure(df, summary)
    rerank_summary = build_reranking_summary(df)
    rerank_summary.to_csv(OUT_DIR / "13_effet_reranking_pgvector_neo4j.csv", index=False)
    rerank_fig = build_reranking_figure(df, ablation)
    print(
        {
            "n_pairs": int(len(df)),
            "summary_csv": str(OUT_DIR / "13_distribution_scores_pgvector_neo4j.csv"),
            "weights_csv": str(OUT_DIR / "13_optuna_poids_score_hybride.csv"),
            "figure": str(fig),
            "reranking_csv": str(OUT_DIR / "13_effet_reranking_pgvector_neo4j.csv"),
            "reranking_figure": str(rerank_fig),
        }
    )


if __name__ == "__main__":
    main()
