from __future__ import annotations

import json
import math
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "memoire_stats" / "implementation_results"
FIG = ROOT / "rapport" / "figures" / "generated" / "implementation_results"

COLORS = {
    "blue": "#2457A6",
    "teal": "#008B8B",
    "green": "#3A7D44",
    "orange": "#E68619",
    "red": "#B13E3E",
    "gray": "#5F6B7A",
    "light": "#EEF3F8",
    "dark": "#1F2A44",
}


def _setup_plot():
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


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")


def save_json(obj: dict[str, Any], name: str) -> None:
    (OUT / f"{name}.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_fig(fig, name: str) -> str:
    path = FIG / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return str(path.relative_to(ROOT))


def pct(part: float) -> float:
    return round(float(part) * 100, 2)


def count_non_empty(s: pd.Series) -> int:
    if s.dtype == "object":
        return int(s.fillna("").astype(str).str.strip().ne("").sum())
    return int(s.notna().sum())


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((name for name in names if name in df.columns), None)


def explode_list_column(df: pd.DataFrame, col: str) -> list[str]:
    values: list[str] = []
    if col not in df.columns:
        return values
    for raw in df[col].dropna():
        if isinstance(raw, (list, tuple, set)):
            items = raw
        else:
            text = str(raw).strip()
            if not text or text.lower() in {"nan", "none", "[]", "{}", "()"}:
                continue
            quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
            if quoted:
                items = [a or b for a, b in quoted]
            else:
                for sep in ["|", ";", ","]:
                    if sep in text:
                        items = text.split(sep)
                        break
                else:
                    items = [text]
        for item in items:
            cleaned = str(item).strip()
            if cleaned and cleaned.lower() not in {"nan", "none", "[]", "{}", "()"}:
                values.append(cleaned)
    return values


def top_counts(df: pd.DataFrame, col: str, name: str, top_n: int = 12) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[name, "n", "part"])
    values = df[col].astype("object").where(df[col].notna(), "Non renseigné")
    out = (
        values
        .astype(str)
        .str.strip()
        .replace({"": "Non renseigné", "nan": "Non renseigné", "None": "Non renseigné"})
        .value_counts()
        .head(top_n)
        .rename_axis(name)
        .reset_index(name="n")
    )
    out["part"] = out["n"] / len(df) if len(df) else 0
    out["part_pct"] = out["part"].map(pct)
    return out


def query_pgvector() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sys.path.insert(0, str(ROOT / "src" / "04_pgvector"))
    try:
        import psycopg
        from config_pgvector import PG_CONN

        with psycopg.connect(**PG_CONN) as conn:
            emb = pd.read_sql(
                """
                SELECT entity_kind, COUNT(*)::int AS n,
                       COUNT(neo4j_node_id)::int AS n_with_neo4j_id,
                       COUNT(DISTINCT model_id)::int AS n_models
                FROM embeddings
                GROUP BY entity_kind
                ORDER BY n DESC
                """,
                conn,
            )
            docs = pd.read_sql(
                """
                SELECT source, COUNT(*)::int AS n,
                       COUNT(embedding)::int AS n_with_embedding,
                       COUNT(neo4j_node_id)::int AS n_with_neo4j_id
                FROM doc_chunks
                GROUP BY source
                ORDER BY source
                """,
                conn,
            )
        return emb, docs, {"available": True, "error": None}
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), {"available": False, "error": str(exc)}


def query_neo4j() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sys.path.insert(0, str(ROOT / "src" / "03_knowledge_graph"))
    try:
        from neo4j import GraphDatabase
        from config_neo4j import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session(database=NEO4J_DATABASE) as session:
            labels = session.run(
                """
                CALL db.labels() YIELD label
                CALL {
                  WITH label
                  MATCH (n)
                  WHERE label IN labels(n)
                  RETURN count(n) AS n
                }
                RETURN label, n
                ORDER BY n DESC
                """
            )
            nodes = pd.DataFrame([dict(r) for r in labels])
            totals_row = session.run(
                """
                MATCH (n)
                WITH count(n) AS nodes
                MATCH ()-[r]->()
                RETURN nodes, count(r) AS relationships
                """
            ).single()
        driver.close()
        totals = pd.DataFrame([dict(totals_row)]) if totals_row else pd.DataFrame()
        return nodes, totals, {"available": True, "error": None}
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), {"available": False, "error": str(exc)}


def copy_existing_evaluation_figures() -> dict[str, str]:
    mapping = {
        "embedding_benchmark_plot": ROOT / "outputs" / "evaluation" / "embedding_benchmark_plot.png",
        "ablation_metrics_comparison": ROOT
        / "outputs"
        / "evaluation"
        / "pgvector_vs_graph"
        / "ablation_metrics_comparison.png",
        "graph_rerank_rank_shift": ROOT
        / "outputs"
        / "evaluation"
        / "pgvector_vs_graph"
        / "graph_rerank_rank_shift.png",
    }
    copied: dict[str, str] = {}
    for name, src in mapping.items():
        if src.exists():
            dst = FIG / f"{name}.png"
            shutil.copy2(src, dst)
            copied[name] = str(dst.relative_to(ROOT))
    return copied


def main() -> None:
    ensure_dirs()
    plt, fmt_int = _setup_plot()

    offres = pd.read_parquet(ROOT / "data" / "processed" / "offres_normalized.parquet")
    candidats = pd.read_parquet(ROOT / "data" / "processed" / "candidats_normalized.parquet")
    mepc = pd.read_parquet(ROOT / "data" / "processed" / "mepc_groupes_base.parquet")
    ncf = pd.read_parquet(ROOT / "data" / "processed" / "ncf_dom_detailles.parquet")

    overview = pd.DataFrame(
        [
            {"bloc": "Offres d'emploi normalisées", "lignes": len(offres), "colonnes": offres.shape[1]},
            {"bloc": "Profils candidats normalisés", "lignes": len(candidats), "colonnes": candidats.shape[1]},
            {"bloc": "Groupes de base MEPC", "lignes": len(mepc), "colonnes": mepc.shape[1]},
            {"bloc": "Domaines détaillés NCF", "lignes": len(ncf), "colonnes": ncf.shape[1]},
        ]
    )
    save_table(overview, "01_vue_ensemble")

    quality_specs = [
        ("Offres", offres, "titre_poste", "Intitulé"),
        ("Offres", offres, "secteur_principal", "Secteur"),
        ("Offres", offres, "ville_principale", "Ville"),
        ("Offres", offres, "ncf_niveau_code", "Niveau NCF"),
        ("Offres", offres, "experience_min_ans", "Expérience min."),
        ("Offres", offres, "skills_list", "Compétences"),
        ("Offres", offres, "details_clean", "Détails"),
        ("Offres", offres, "text_to_embed", "Texte embedding"),
        ("Candidats", candidats, "ncf_niveau_final", "Niveau NCF"),
        ("Candidats", candidats, "secteur_metier", "Secteur visé"),
        ("Candidats", candidats, "metier_vise", "Métier visé"),
        ("Candidats", candidats, "objectif", "Objectif"),
        ("Candidats", candidats, "text_to_embed", "Texte embedding"),
    ]
    quality_rows = []
    for source, df, col, label in quality_specs:
        if col in df.columns:
            n = count_non_empty(df[col])
            quality_rows.append(
                {
                    "source": source,
                    "champ": label,
                    "n_renseignes": n,
                    "total": len(df),
                    "taux_pct": round(n / len(df) * 100, 2) if len(df) else 0,
                }
            )
    quality = pd.DataFrame(quality_rows)
    save_table(quality, "02_qualite_champs")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    plot = quality.sort_values("taux_pct")
    colors = plot["source"].map({"Offres": COLORS["blue"], "Candidats": COLORS["orange"]})
    labels = plot["source"] + " - " + plot["champ"]
    ax.barh(labels, plot["taux_pct"], color=colors)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Taux de renseignement (%)")
    ax.set_title("Disponibilité des champs utilisés par les briques du système")
    for y, value in enumerate(plot["taux_pct"]):
        ax.text(value + 1, y, f"{value:.1f}%", va="center", fontsize=8)
    save_fig(fig, "02_qualite_champs")
    plt.close(fig)

    sector = top_counts(offres, "secteur_principal", "secteur", 15)
    city = top_counts(offres, "ville_principale", "ville", 15)
    save_table(sector, "03_offres_par_secteur")
    save_table(city, "04_offres_par_ville")

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    plot = sector.sort_values("n")
    ax.barh(plot["secteur"], plot["n"], color=COLORS["teal"])
    ax.xaxis.set_major_formatter(fmt_int)
    ax.set_xlabel("Nombre d'offres")
    ax.set_title("Secteurs les plus représentés dans les offres")
    for y, row in enumerate(plot.itertuples()):
        ax.text(row.n, y, f" {row.part_pct:.1f}%", va="center", fontsize=8)
    save_fig(fig, "03_offres_par_secteur")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    plot = city.sort_values("n")
    ax.barh(plot["ville"], plot["n"], color=COLORS["orange"])
    ax.xaxis.set_major_formatter(fmt_int)
    ax.set_xlabel("Nombre d'offres")
    ax.set_title("Concentration géographique des offres")
    for y, row in enumerate(plot.itertuples()):
        ax.text(row.n, y, f" {row.part_pct:.1f}%", va="center", fontsize=8)
    save_fig(fig, "04_offres_par_ville")
    plt.close(fig)

    ncf_col = first_existing(offres, ["ncf_niveau_code", "niveau_etudes_raw"])
    ncf_counts = top_counts(offres, ncf_col, "niveau", 12) if ncf_col else pd.DataFrame()
    exp = pd.to_numeric(offres.get("experience_min_ans"), errors="coerce")
    exp_summary = exp.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).rename("valeur").reset_index()
    save_table(ncf_counts, "05_offres_par_niveau_ncf")
    save_table(exp_summary, "05_experience_minimale_resume")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    if not ncf_counts.empty:
        plot = ncf_counts.sort_values("n")
        axes[0].barh(plot["niveau"], plot["n"], color=COLORS["blue"])
        axes[0].xaxis.set_major_formatter(fmt_int)
    axes[0].set_title("Niveaux NCF demandés")
    axes[0].set_xlabel("Nombre d'offres")
    exp.dropna().clip(upper=20).plot(kind="hist", bins=20, ax=axes[1], color=COLORS["green"], edgecolor="white")
    axes[1].set_title("Expérience minimale demandée")
    axes[1].set_xlabel("Années")
    save_fig(fig, "05_ncf_et_experience_offres")
    plt.close(fig)

    cand_ncf_col = first_existing(candidats, ["ncf_niveau_final", "ncf_code_niveau_etude"])
    cand_sector_col = first_existing(candidats, ["secteur_metier", "secteur_demande", "secteur_activite_cand"])
    cand_ncf = top_counts(candidats, cand_ncf_col, "niveau", 12) if cand_ncf_col else pd.DataFrame()
    cand_sector = top_counts(candidats, cand_sector_col, "secteur", 12) if cand_sector_col else pd.DataFrame()
    age = pd.to_numeric(candidats.get("age"), errors="coerce")
    save_table(cand_ncf, "06_candidats_par_niveau_ncf")
    save_table(cand_sector, "06_candidats_par_secteur")
    save_table(age.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).rename("valeur").reset_index(), "06_age_candidats_resume")

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
    if not cand_ncf.empty:
        plot = cand_ncf.sort_values("n")
        axes[0].barh(plot["niveau"], plot["n"], color=COLORS["blue"])
        axes[0].xaxis.set_major_formatter(fmt_int)
    axes[0].set_title("Niveaux candidats")
    if not cand_sector.empty:
        plot = cand_sector.sort_values("n")
        axes[1].barh(plot["secteur"], plot["n"], color=COLORS["teal"])
        axes[1].xaxis.set_major_formatter(fmt_int)
    axes[1].set_title("Secteurs visés")
    age.dropna().clip(lower=15, upper=70).plot(kind="hist", bins=20, ax=axes[2], color=COLORS["orange"], edgecolor="white")
    axes[2].set_title("Âge des candidats")
    axes[2].set_xlabel("Âge")
    save_fig(fig, "06_structure_candidats")
    plt.close(fig)

    skill_values = explode_list_column(offres, "skills_list")
    skills = pd.DataFrame(Counter(skill_values).most_common(20), columns=["competence", "n"])
    skills["part_offres_pct"] = skills["n"].map(lambda n: round(n / len(offres) * 100, 2))
    save_table(skills, "07_top_competences_offres")
    if not skills.empty:
        fig, ax = plt.subplots(figsize=(10.5, 7.0))
        plot = skills.sort_values("n")
        ax.barh(plot["competence"], plot["n"], color=COLORS["orange"])
        ax.xaxis.set_major_formatter(fmt_int)
        ax.set_xlabel("Nombre d'offres")
        ax.set_title("Compétences et signaux métiers les plus fréquents")
        save_fig(fig, "07_top_competences_offres")
        plt.close(fig)

    text_rows = []
    for source, df in [("Offres", offres), ("Candidats", candidats)]:
        lengths = df["text_to_embed"].fillna("").astype(str).str.len()
        for stat, value in lengths.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).items():
            text_rows.append({"source": source, "stat": stat, "longueur": float(value)})
    text_stats = pd.DataFrame(text_rows)
    save_table(text_stats, "08_longueur_text_to_embed")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.boxplot(
        [
            offres["text_to_embed"].fillna("").astype(str).str.len(),
            candidats["text_to_embed"].fillna("").astype(str).str.len(),
        ],
        labels=["Offres", "Candidats"],
        patch_artist=True,
        boxprops={"facecolor": COLORS["light"], "color": COLORS["dark"]},
        medianprops={"color": COLORS["red"], "linewidth": 2},
    )
    ax.set_ylabel("Nombre de caractères")
    ax.set_title("Richesse textuelle transmise au modèle d'embeddings")
    ax.yaxis.set_major_formatter(fmt_int)
    save_fig(fig, "08_longueur_text_to_embed")
    plt.close(fig)

    if cand_sector_col:
        offer_sector_all = top_counts(offres, "secteur_principal", "secteur", 1000)[["secteur", "n"]].rename(columns={"n": "offres"})
        cand_sector_all = top_counts(candidats, cand_sector_col, "secteur", 1000)[["secteur", "n"]].rename(columns={"n": "candidats"})
        balance = offer_sector_all.merge(cand_sector_all, on="secteur", how="outer").fillna(0)
        balance["offres"] = balance["offres"].astype(int)
        balance["candidats"] = balance["candidats"].astype(int)
        balance["total"] = balance["offres"] + balance["candidats"]
        balance["ratio_offres_candidats"] = balance.apply(
            lambda r: round(r["offres"] / r["candidats"], 3) if r["candidats"] else math.inf,
            axis=1,
        )
        balance = balance.sort_values("total", ascending=False)
    else:
        balance = pd.DataFrame()
    save_table(balance, "09_equilibre_offres_candidats_secteurs")
    if not balance.empty:
        fig, ax = plt.subplots(figsize=(10.8, 6.2))
        plot = balance.head(14).sort_values("total")
        y = range(len(plot))
        ax.barh([i - 0.18 for i in y], plot["offres"], height=0.34, color=COLORS["blue"], label="Offres")
        ax.barh([i + 0.18 for i in y], plot["candidats"], height=0.34, color=COLORS["orange"], label="Candidats")
        ax.set_yticks(list(y))
        ax.set_yticklabels(plot["secteur"])
        ax.xaxis.set_major_formatter(fmt_int)
        ax.set_xlabel("Nombre")
        ax.set_title("Déséquilibres sectoriels entre opportunités et profils")
        ax.legend()
        save_fig(fig, "09_equilibre_offres_candidats_secteurs")
        plt.close(fig)

    mepc_col = first_existing(mepc, ["grand_groupe", "code_grand_groupe", "libelle_grand_groupe"])
    ncf_col_ref = first_existing(ncf, ["grand_domaine", "code_grand_domaine", "libelle_grand_domaine"])
    mepc_groups = top_counts(mepc, mepc_col, "groupe", 12) if mepc_col else pd.DataFrame()
    ncf_groups = top_counts(ncf, ncf_col_ref, "domaine", 12) if ncf_col_ref else pd.DataFrame()
    save_table(mepc_groups, "10_mepc_groupes_base")
    save_table(ncf_groups, "10_ncf_domaines_detailles")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    axes[0].bar(["MEPC\ngroupes de base", "NCF\ndomaines détaillés"], [len(mepc), len(ncf)], color=[COLORS["blue"], COLORS["green"]])
    axes[0].set_title("Couverture globale")
    axes[0].yaxis.set_major_formatter(fmt_int)
    if not mepc_groups.empty:
        axes[1].barh(mepc_groups.sort_values("n")["groupe"], mepc_groups.sort_values("n")["n"], color=COLORS["orange"])
        axes[1].xaxis.set_major_formatter(fmt_int)
    axes[1].set_title("MEPC par grands groupes")
    save_fig(fig, "10_referentiels_mepc_ncf")
    plt.close(fig)

    pg_counts, doc_chunks, pg_status = query_pgvector()
    neo_nodes, neo_totals, neo_status = query_neo4j()
    save_table(pg_counts, "11_pgvector_embeddings")
    save_table(doc_chunks, "11_pgvector_doc_chunks")
    save_table(neo_nodes, "12_neo4j_nodes")
    save_table(neo_totals, "12_neo4j_totals")
    if not pg_counts.empty:
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        plot = pg_counts.sort_values("n")
        ax.barh(plot["entity_kind"], plot["n"], color=COLORS["blue"])
        ax.xaxis.set_major_formatter(fmt_int)
        ax.set_xlabel("Nombre d'embeddings")
        ax.set_title("Entités indexées dans pgvector")
        save_fig(fig, "11_pgvector_embeddings")
        plt.close(fig)
    if not doc_chunks.empty:
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        x = range(len(doc_chunks))
        ax.bar(x, doc_chunks["n"], color=COLORS["green"], label="Chunks")
        ax.plot(x, doc_chunks["n_with_neo4j_id"], color=COLORS["red"], marker="o", linewidth=2.2, label="Synchronisés Neo4j")
        ax.set_xticks(list(x))
        ax.set_xticklabels(doc_chunks["source"], rotation=20, ha="right")
        ax.yaxis.set_major_formatter(fmt_int)
        ax.set_title("Fragments référentiels indexés et reliés au graphe")
        ax.legend()
        save_fig(fig, "11_pgvector_doc_chunks")
        plt.close(fig)
    if not neo_nodes.empty:
        fig, ax = plt.subplots(figsize=(9.8, 5.8))
        plot = neo_nodes.head(15).sort_values("n")
        ax.barh(plot["label"], plot["n"], color=COLORS["orange"])
        ax.xaxis.set_major_formatter(fmt_int)
        ax.set_xlabel("Nombre de nœuds")
        ax.set_title("Composition du graphe Neo4j")
        save_fig(fig, "12_neo4j_nodes")
        plt.close(fig)

    benchmark_path = ROOT / "outputs" / "evaluation" / "embedding_benchmark.csv"
    benchmark = pd.read_csv(benchmark_path) if benchmark_path.exists() else pd.DataFrame()
    save_table(benchmark, "13_embedding_benchmark")
    copied_eval = copy_existing_evaluation_figures()

    ablation_path = ROOT / "outputs" / "evaluation" / "pgvector_vs_graph" / "ablation_metrics.csv"
    ablation = pd.read_csv(ablation_path) if ablation_path.exists() else pd.DataFrame()
    save_table(ablation, "14_ablation_pgvector_graph")

    top_sector = sector.iloc[0].to_dict() if not sector.empty else {}
    top_city = city.iloc[0].to_dict() if not city.empty else {}
    top_skill = skills.iloc[0].to_dict() if not skills.empty else {}
    best_model = benchmark.sort_values("ndcg@10", ascending=False).iloc[0].to_dict() if "ndcg@10" in benchmark.columns and not benchmark.empty else {}
    summary = {
        "n_offres": int(len(offres)),
        "n_candidats": int(len(candidats)),
        "n_groupes_base_mepc": int(len(mepc)),
        "n_domaines_detailles_ncf": int(len(ncf)),
        "top_secteur_offres": top_sector,
        "top_ville_offres": top_city,
        "top_competence_offres": top_skill,
        "n_ft_eligible_offres": int(offres["ft_eligible"].fillna(False).astype(bool).sum()) if "ft_eligible" in offres else None,
        "n_pgvector_embeddings": int(pg_counts["n"].sum()) if not pg_counts.empty else None,
        "n_doc_chunks": int(doc_chunks["n"].sum()) if not doc_chunks.empty else None,
        "n_doc_chunks_synced_neo4j": int(doc_chunks["n_with_neo4j_id"].sum()) if not doc_chunks.empty else None,
        "n_neo4j_nodes": int(neo_totals["nodes"].iloc[0]) if not neo_totals.empty and "nodes" in neo_totals else None,
        "n_neo4j_relationships": int(neo_totals["relationships"].iloc[0]) if not neo_totals.empty and "relationships" in neo_totals else None,
        "pgvector_status": pg_status,
        "neo4j_status": neo_status,
        "best_embedding_model": best_model,
        "evaluation_figures": copied_eval,
    }
    save_json(summary, "00_resume_implementation")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
