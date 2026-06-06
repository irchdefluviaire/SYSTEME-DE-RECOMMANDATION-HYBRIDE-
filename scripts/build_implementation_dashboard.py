from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "memoire_stats" / "implementation_results"
FIG = ROOT / "rapport" / "figures" / "generated" / "implementation_results"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    FIG.mkdir(parents=True, exist_ok=True)
    summary = json.loads((OUT / "00_resume_implementation.json").read_text(encoding="utf-8"))
    pg = pd.read_csv(OUT / "11_pgvector_embeddings.csv")
    bench = pd.read_csv(OUT / "13_embedding_benchmark.csv")
    ablation = pd.read_csv(OUT / "14_ablation_pgvector_graph.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 140,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )
    colors = {
        "blue": "#2457A6",
        "teal": "#008B8B",
        "orange": "#E68619",
        "green": "#3A7D44",
        "red": "#B13E3E",
        "gray": "#5F6B7A",
        "light": "#EEF3F8",
    }
    fmt = FuncFormatter(lambda x, _pos=None: f"{int(x):,}".replace(",", " "))

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.25, 1.15], height_ratios=[1, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    volumes = pd.DataFrame(
        {
            "bloc": ["Offres", "Candidats", "Embeddings", "Noeuds Neo4j"],
            "n": [
                summary["n_offres"],
                summary["n_candidats"],
                summary["n_pgvector_embeddings"],
                summary["n_neo4j_nodes"],
            ],
        }
    ).sort_values("n")
    ax0.barh(volumes["bloc"], volumes["n"], color=[colors["orange"], colors["teal"], colors["blue"], colors["green"]])
    ax0.xaxis.set_major_formatter(fmt)
    ax0.set_title("Volume alimente par brique")
    ax0.set_xlabel("Nombre d'entites")

    ax1 = fig.add_subplot(gs[0, 1])
    pg_plot = pg.sort_values("n", ascending=False).head(6)
    ax1.bar(pg_plot["entity_kind"], pg_plot["n"], color=colors["blue"])
    ax1.plot(pg_plot["entity_kind"], pg_plot["n_with_neo4j_id"], color=colors["red"], marker="o", linewidth=2.4, label="Relie Neo4j")
    ax1.yaxis.set_major_formatter(fmt)
    ax1.set_title("pgvector et synchronisation graphe")
    ax1.set_ylabel("Embeddings")
    ax1.tick_params(axis="x", rotation=25)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 2])
    top_bench = bench.sort_values("ndcg@10", ascending=False).head(5).sort_values("ndcg@10")
    ax2.barh(top_bench["label"], top_bench["ndcg@10"], color=colors["teal"])
    ax2.set_xlim(0, max(0.7, float(top_bench["ndcg@10"].max()) + 0.05))
    ax2.set_title("Qualite du modele d'embedding")
    ax2.set_xlabel("NDCG@10")
    for y, val in enumerate(top_bench["ndcg@10"]):
        ax2.text(val + 0.01, y, f"{val:.3f}", va="center", fontsize=9)

    ax3 = fig.add_subplot(gs[1, 0])
    chunks = summary["n_doc_chunks"]
    synced = summary["n_doc_chunks_synced_neo4j"]
    ax3.pie(
        [synced, max(chunks - synced, 0)],
        labels=["Synchronises", "Non synchronises"],
        autopct="%1.0f%%",
        colors=[colors["green"], colors["light"]],
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax3.set_title(f"Fragments documentaires: {chunks}")

    ax4 = fig.add_subplot(gs[1, 1])
    metrics = ["precision@1", "recall@10", "ndcg@10", "mrr@10"]
    ab = ablation.set_index("system")[metrics].T
    ab.plot(kind="bar", ax=ax4, color=[colors["orange"], colors["blue"]])
    ax4.set_ylim(0, max(0.75, float(ab.max().max()) + 0.08))
    ax4.set_title("Effet observe du reranking graphe")
    ax4.set_ylabel("Score")
    ax4.tick_params(axis="x", rotation=0)
    ax4.legend(["pgvector seul", "pgvector + graphe"], loc="upper left")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    lines = [
        ("Offres normalisees", summary["n_offres"]),
        ("Candidats normalises", summary["n_candidats"]),
        ("Offres eligibles FT", summary["n_ft_eligible_offres"]),
        ("Relations Neo4j", summary["n_neo4j_relationships"]),
        ("Meilleur NDCG@10", summary["best_embedding_model"]["ndcg@10"]),
        ("Latence embedding", f"{summary['best_embedding_model']['latency_ms_per_sentence']:.1f} ms"),
    ]
    y = 0.95
    ax5.text(0, y, "Indicateurs retenus", fontsize=14, weight="bold", color=colors["blue"], transform=ax5.transAxes)
    y -= 0.13
    for label, value in lines:
        if isinstance(value, int):
            value = f"{value:,}".replace(",", " ")
        elif isinstance(value, float):
            value = f"{value:.4f}"
        ax5.text(0, y, label, fontsize=10.5, color=colors["gray"], transform=ax5.transAxes)
        ax5.text(0.95, y, str(value), fontsize=12, weight="bold", ha="right", color=colors["blue"], transform=ax5.transAxes)
        y -= 0.12

    fig.suptitle("Resultats d'implementation par brique du systeme", fontsize=17, weight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "00_dashboard_briques.png", dpi=240, bbox_inches="tight")


if __name__ == "__main__":
    main()
