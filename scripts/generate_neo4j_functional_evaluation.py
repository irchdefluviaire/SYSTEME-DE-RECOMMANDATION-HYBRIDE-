from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "memoire_stats" / "implementation_deep_current"
FIG = ROOT / "rapport" / "figures" / "generated" / "evaluation"

COLORS = {
    "blue": "#2457A6",
    "teal": "#008B8B",
    "green": "#3A7D44",
    "orange": "#E68619",
    "red": "#B13E3E",
    "purple": "#6F4DA8",
    "gray": "#5F6B7A",
}


def setup_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )

    def fmt_int(x, _pos=None):
        if pd.isna(x):
            return ""
        return f"{int(x):,}".replace(",", " ")

    return plt, FuncFormatter(fmt_int)


def save_table(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_json(obj: dict[str, Any], name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_fig(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return path


def run_query(session, cypher: str, **params: Any) -> pd.DataFrame:
    return pd.DataFrame([dict(r) for r in session.run(cypher, **params)])


def main() -> None:
    plt, fmt_int = setup_plot()
    sys.path.insert(0, str(ROOT / "src" / "03_knowledge_graph"))
    from neo4j import GraphDatabase
    from config_neo4j import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(database=NEO4J_DATABASE) as session:
        metier_comp = run_query(
            session,
            """
            MATCH (m:Métier)-[:NECESSITE]->(c:Compétence)
            WITH m, count(DISTINCT c) AS n_competences,
                 collect(DISTINCT coalesce(c.preferredLabel, c.label, c.nom, c.id))[0..8] AS exemples
            RETURN coalesce(m.preferredLabel, m.label, m.nom, m.id) AS metier,
                   n_competences,
                   exemples
            ORDER BY n_competences DESC
            LIMIT 25
            """,
        )

        comp_offres = run_query(
            session,
            """
            MATCH (o:OffreEmploi)-[:REQUIERT]->(c:Compétence)
            WITH c, count(DISTINCT o) AS n_offres,
                 collect(DISTINCT coalesce(o.titre_poste, o.label, o.nom, o.id))[0..5] AS exemples_offres
            RETURN coalesce(c.preferredLabel, c.label, c.nom, c.id) AS competence,
                   n_offres,
                   exemples_offres
            ORDER BY n_offres DESC
            LIMIT 25
            """,
        )

        skill_gap = run_query(
            session,
            """
            MATCH (cand:Candidat)
            WITH cand
            ORDER BY rand()
            LIMIT 350
            MATCH (offre:OffreEmploi)
            WITH cand, offre
            ORDER BY rand()
            LIMIT 7000
            MATCH (offre)-[:REQUIERT]->(req:Compétence)
            WITH cand, offre, collect(DISTINCT req) AS required
            OPTIONAL MATCH (cand)-[:POSSEDE]->(got:Compétence)
            WITH cand, offre, required, collect(DISTINCT got) AS possessed
            WITH cand, offre,
                 size(required) AS n_required,
                 size([s IN required WHERE s IN possessed]) AS n_acquired
            WHERE n_required > 0
            RETURN n_required,
                   n_acquired,
                   n_required - n_acquired AS skill_gap,
                   toFloat(n_acquired) / n_required AS taux_match
            """,
        )

        ncf_pairs = run_query(
            session,
            """
            CALL {
              MATCH (cand:Candidat)-[:A_NIVEAU]->(cn:NiveauFormationNCF)
              RETURN cand, cn
              ORDER BY rand()
              LIMIT 120
            }
            CALL {
              MATCH (offre:OffreEmploi)-[:REQUIERT_NIVEAU|REQUIERT_NIVEAU_NCF]->(on:NiveauFormationNCF)
              RETURN offre, on
              ORDER BY rand()
              LIMIT 120
            }
            WITH cand, cn, offre, on
            LIMIT 7000
            RETURN coalesce(cn.code, cn.label, cn.nom, cn.id) AS niveau_candidat,
                   coalesce(on.code, on.label, on.nom, on.id) AS niveau_offre,
                   CASE WHEN elementId(cn) = elementId(on) THEN 1 ELSE 0 END AS meme_niveau
            """,
        )

        coverage = run_query(
            session,
            """
            MATCH (o:OffreEmploi)
            WITH count(o) AS n_offres
            MATCH (o2:OffreEmploi)-[:REQUIERT]->(:Compétence)
            WITH n_offres, count(DISTINCT o2) AS n_offres_avec_comp
            MATCH (cand:Candidat)
            WITH n_offres, n_offres_avec_comp, count(cand) AS n_candidats
            MATCH (cand2:Candidat)-[:POSSEDE]->(:Compétence)
            WITH n_offres, n_offres_avec_comp, n_candidats, count(DISTINCT cand2) AS n_candidats_avec_comp
            RETURN n_offres, n_offres_avec_comp,
                   toFloat(n_offres_avec_comp) / n_offres AS taux_offres_avec_comp,
                   n_candidats, n_candidats_avec_comp,
                   toFloat(n_candidats_avec_comp) / n_candidats AS taux_candidats_avec_comp
            """,
        )
    driver.close()

    save_table(metier_comp, "19_eval_graph_metier_competences")
    save_table(comp_offres, "19_eval_graph_competence_offres")
    save_table(skill_gap, "19_eval_graph_skill_gap_sample")
    save_table(ncf_pairs, "19_eval_graph_ncf_pairs_sample")
    save_table(coverage, "19_eval_graph_coverage")

    sg_summary = skill_gap.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).reset_index()
    ncf_summary = pd.DataFrame(
        [
            {
                "n_pairs": int(len(ncf_pairs)),
                "taux_meme_niveau": float(ncf_pairs["meme_niveau"].mean()) if len(ncf_pairs) else None,
            }
        ]
    )
    save_table(sg_summary, "19_eval_graph_skill_gap_summary")
    save_table(ncf_summary, "19_eval_graph_ncf_summary")

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    plot = metier_comp.head(12).sort_values("n_competences")
    axes[0, 0].barh(plot["metier"].astype(str).str.slice(0, 34), plot["n_competences"], color=COLORS["purple"])
    axes[0, 0].set_title("Recherche métier vers compétences")
    axes[0, 0].set_xlabel("Compétences NECESSITE")

    plot = comp_offres.head(12).sort_values("n_offres")
    axes[0, 1].barh(plot["competence"].astype(str).str.slice(0, 34), plot["n_offres"], color=COLORS["green"])
    axes[0, 1].xaxis.set_major_formatter(fmt_int)
    axes[0, 1].set_title("Recherche compétence vers offres")
    axes[0, 1].set_xlabel("Offres REQUIERT")

    axes[1, 0].hist(skill_gap["skill_gap"], bins=range(0, int(skill_gap["skill_gap"].max()) + 2), color=COLORS["red"], alpha=0.86)
    axes[1, 0].set_title("Skill gap échantillonné")
    axes[1, 0].set_xlabel("Compétences manquantes")

    ncf_counts = ncf_pairs["meme_niveau"].map({0: "Niveau différent", 1: "Même niveau"}).value_counts()
    axes[1, 1].bar(ncf_counts.index, ncf_counts.values, color=[COLORS["orange"], COLORS["blue"]])
    axes[1, 1].yaxis.set_major_formatter(fmt_int)
    axes[1, 1].set_title("Compatibilité NCF exacte")
    axes[1, 1].set_ylabel("Couples échantillonnés")

    save_fig(fig, "functional_graph_evaluation")
    plt.close(fig)

    summary = {
        "metier_competences_top_mean": float(metier_comp["n_competences"].mean()) if len(metier_comp) else None,
        "competence_offres_top_mean": float(comp_offres["n_offres"].mean()) if len(comp_offres) else None,
        "skill_gap_mean": float(skill_gap["skill_gap"].mean()) if len(skill_gap) else None,
        "skill_gap_median": float(skill_gap["skill_gap"].median()) if len(skill_gap) else None,
        "ncf_same_level_rate": float(ncf_pairs["meme_niveau"].mean()) if len(ncf_pairs) else None,
        "coverage": coverage.to_dict("records")[0] if len(coverage) else {},
        "figure": "rapport/figures/generated/evaluation/functional_graph_evaluation.png",
    }
    save_json(summary, "19_eval_graph_functional_summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
