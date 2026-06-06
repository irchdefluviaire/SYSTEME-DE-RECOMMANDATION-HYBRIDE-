from __future__ import annotations

import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "memoire_stats" / "implementation_deep_current"
FIG = ROOT / "rapport" / "figures" / "generated" / "implementation_deep_current"
NOTEBOOK = ROOT / "notebooks" / "12_analyse_approfondie_graph_pgvector.ipynb"

COLORS = {
    "blue": "#2457A6",
    "teal": "#008B8B",
    "green": "#3A7D44",
    "orange": "#E68619",
    "red": "#B13E3E",
    "purple": "#6F4DA8",
    "gray": "#5F6B7A",
    "light": "#EEF3F8",
    "dark": "#1F2A44",
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


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    for stale in ["06_pgvector_error.json", "08_neo4j_error.json"]:
        path = OUT / stale
        if path.exists():
            path.unlink()


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_json(obj: dict[str, Any], name: str) -> Path:
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_fig(fig, name: str) -> Path:
    path = FIG / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return path


def pct(x: float) -> float:
    return round(float(x) * 100, 2)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def vector_from_text(text: str) -> np.ndarray:
    return np.fromstring(str(text).strip().strip("[]"), sep=",", dtype=np.float32)


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def explode_values(series: pd.Series) -> list[str]:
    values: list[str] = []
    for raw in series.dropna():
        if isinstance(raw, (list, tuple, set, np.ndarray)):
            items = raw
        else:
            text = str(raw).strip()
            if not text or text.lower() in {"nan", "none", "[]"}:
                continue
            items = [part.strip(" '\"") for part in text.replace(";", ",").replace("|", ",").split(",")]
        for item in items:
            cleaned = str(item).strip()
            if cleaned and cleaned.lower() not in {"nan", "none", "[]"}:
                values.append(cleaned)
    return values


def load_current_data() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for key, path in {
        "offres": ROOT / "data" / "processed" / "offres_normalized.parquet",
        "candidats": ROOT / "data" / "processed" / "candidats_normalized.parquet",
        "vrai_offres": ROOT / "data" / "processed" / "vrai_data_offres_cleaned.parquet",
    }.items():
        if path.exists():
            data[key] = pd.read_parquet(path)
    return data


def build_data_description(data: dict[str, pd.DataFrame], plt, fmt_int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    offres = data.get("offres", pd.DataFrame())
    candidats = data.get("candidats", pd.DataFrame())
    vrai = data.get("vrai_offres", pd.DataFrame())

    overview = pd.DataFrame(
        [
            {"bloc": "Offres normalisées", "lignes": len(offres), "colonnes": offres.shape[1]},
            {"bloc": "Candidats normalisés", "lignes": len(candidats), "colonnes": candidats.shape[1]},
            {"bloc": "Offres vrai_data nettoyées", "lignes": len(vrai), "colonnes": vrai.shape[1]},
        ]
    )
    save_table(overview, "01_description_vue_ensemble")
    out["overview"] = overview.to_dict("records")

    if not vrai.empty and "source_clean" in vrai.columns:
        src = (
            vrai["source_clean"].fillna("Non précisée").replace("", "Non précisée")
            .value_counts().rename_axis("source").reset_index(name="n")
        )
        src["part_pct"] = (src["n"] / len(vrai) * 100).round(2)
        save_table(src, "01_offres_par_source")
        fig, ax = plt.subplots(figsize=(9.4, 5.2))
        plot = src.sort_values("n")
        ax.barh(plot["source"], plot["n"], color=COLORS["blue"])
        ax.xaxis.set_major_formatter(fmt_int)
        ax.set_xlabel("Nombre d'offres")
        ax.set_title("Offres par source après nettoyage")
        for y, row in enumerate(plot.itertuples()):
            ax.text(row.n, y, f" {row.part_pct:.1f}%", va="center", fontsize=8)
        save_fig(fig, "01_offres_par_source")
        plt.close(fig)
        out["sources"] = src.to_dict("records")

    zone_col = next((c for c in ["zone_clean", "zone_localisation", "zone"] if c in vrai.columns), None)
    city_col = next((c for c in ["ville_clean", "ville_principale_clean", "ville_principale"] if c in vrai.columns), None)
    if not vrai.empty:
        zones = (
            vrai[zone_col].fillna("Non précisée").replace("", "Non précisée")
            .value_counts().rename_axis("zone").reset_index(name="n")
            if zone_col
            else pd.DataFrame()
        )
        if not zones.empty:
            zones["part_pct"] = (zones["n"] / len(vrai) * 100).round(2)
            save_table(zones, "02_zones_localisation")
        cities = (
            vrai[city_col].fillna("Non précisée").replace("", "Non précisée")
            .value_counts().head(15).rename_axis("ville").reset_index(name="n")
            if city_col
            else pd.DataFrame()
        )
        if not cities.empty:
            cities["part_pct"] = (cities["n"] / len(vrai) * 100).round(2)
            save_table(cities, "02_villes_localisation")
        if not zones.empty and not cities.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
            axes[0].pie(
                zones["n"],
                labels=zones["zone"],
                autopct="%1.1f%%",
                startangle=90,
                colors=[COLORS["green"], COLORS["orange"], COLORS["gray"], COLORS["blue"]],
            )
            axes[0].set_title("Zones de localisation")
            plot = cities.sort_values("n")
            axes[1].barh(plot["ville"], plot["n"], color=COLORS["orange"])
            axes[1].xaxis.set_major_formatter(fmt_int)
            axes[1].set_title("Villes les plus fréquentes")
            axes[1].set_xlabel("Nombre d'offres")
            save_fig(fig, "02_localisation_mixte")
            plt.close(fig)
            out["zones"] = zones.to_dict("records")
            out["cities"] = cities.to_dict("records")

    skill_col = next((c for c in ["competences_list_clean", "competence_clean", "competences_clean", "skills_clean"] if c in vrai.columns), None)
    desc_col = next((c for c in ["description_disponible", "has_description"] if c in vrai.columns), None)
    if not vrai.empty:
        if skill_col:
            skill_values = explode_values(vrai[skill_col])
            skills = pd.Series(skill_values).value_counts().head(15).rename_axis("competence").reset_index(name="n")
        else:
            skills = pd.DataFrame()
        if not skills.empty:
            skills["part_pct"] = (skills["n"] / len(vrai) * 100).round(2)
            save_table(skills, "03_competences_frequentes")
        if desc_col:
            desc = (
                vrai.groupby("source_clean", dropna=False)[desc_col].agg(["count", "sum"]).reset_index()
                if "source_clean" in vrai.columns
                else pd.DataFrame()
            )
            if not desc.empty:
                desc.columns = ["source", "n_offres", "n_descriptions"]
                desc["taux_description_pct"] = (desc["n_descriptions"] / desc["n_offres"] * 100).round(2)
                save_table(desc, "03_description_par_source")
        else:
            desc = pd.DataFrame()
        if not skills.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
            plot = skills.sort_values("n")
            axes[0].barh(plot["competence"], plot["n"], color=COLORS["teal"])
            axes[0].xaxis.set_major_formatter(fmt_int)
            axes[0].set_title("Compétences et signaux métiers")
            axes[0].set_xlabel("Nombre d'offres")
            if not desc.empty:
                dplot = desc.sort_values("taux_description_pct")
                axes[1].barh(dplot["source"], dplot["taux_description_pct"], color=COLORS["purple"])
                axes[1].set_xlim(0, 105)
                axes[1].set_title("Descriptions disponibles par source")
                axes[1].set_xlabel("Taux (%)")
            else:
                axes[1].axis("off")
            save_fig(fig, "03_competences_descriptions")
            plt.close(fig)
            out["skills"] = skills.to_dict("records")
            if not desc.empty:
                out["descriptions"] = desc.to_dict("records")

    return out


def build_training_analysis(plt) -> dict[str, Any]:
    metrics_path = ROOT / "models" / "st_finetuned" / "evaluation_metrics.json"
    config_path = ROOT / "src" / "02_finetune_st" / "config_st.json"
    benchmark_path = ROOT / "outputs" / "evaluation" / "embedding_benchmark.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    logs = pd.DataFrame(metrics.get("training_logs", []))
    if not logs.empty:
        save_table(logs, "04_finetuning_logs")
        fig, axes = plt.subplots(3, 1, figsize=(9.8, 8.2), sharex=True)
        train = logs[logs["loss"].notna()] if "loss" in logs else pd.DataFrame()
        ndcg_cols = [c for c in logs.columns if c.endswith("ndcg@10")]
        mrr_cols = [c for c in logs.columns if c.endswith("mrr@10")]
        ndcg_col = ndcg_cols[0] if ndcg_cols else None
        mrr_col = mrr_cols[0] if mrr_cols else None
        evals = logs[logs[ndcg_col].notna()] if ndcg_col else pd.DataFrame()
        if not train.empty:
            axes[0].plot(train["epoch"], train["loss"], color=COLORS["blue"], marker="o", linewidth=1.8)
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Courbes de fine-tuning")
        if not evals.empty and ndcg_col:
            axes[1].plot(
                evals["epoch"],
                evals[ndcg_col],
                color=COLORS["green"],
                marker="o",
                linewidth=2.0,
                label="NDCG@10 validation",
            )
            if mrr_col:
                axes[1].plot(
                    evals["epoch"],
                    evals[mrr_col],
                    color=COLORS["orange"],
                    marker="s",
                    linewidth=1.8,
                    label="MRR@10 validation",
                )
            axes[1].legend()
            axes[1].set_ylim(0.50, 0.68)
        axes[1].set_ylabel("Score")
        if "learning_rate" in logs:
            axes[2].plot(logs["epoch"], logs["learning_rate"], color=COLORS["red"], linewidth=1.8)
        axes[2].set_ylabel("Learning rate")
        axes[2].set_xlabel("Époque")
        save_fig(fig, "04_finetuning_courbes")
        plt.close(fig)

    if benchmark_path.exists():
        benchmark = pd.read_csv(benchmark_path)
        save_table(benchmark, "05_benchmark_modeles_embeddings")
        fig, ax1 = plt.subplots(figsize=(10.8, 5.8))
        plot = benchmark.sort_values("ndcg@10", ascending=True)
        ax1.barh(plot["label"], plot["ndcg@10"], color=COLORS["blue"], label="NDCG@10")
        ax1.set_xlabel("NDCG@10")
        ax1.set_xlim(0, max(0.7, plot["ndcg@10"].max() + 0.05))
        ax2 = ax1.twiny()
        ax2.plot(plot["latency_ms_per_sentence"], plot["label"], color=COLORS["red"], marker="o", linewidth=2, label="Latence")
        ax2.set_xlabel("Latence ms / phrase")
        ax1.set_title("Benchmark : qualité de classement et coût d'encodage")
        save_fig(fig, "05_benchmark_modeles_embeddings")
        plt.close(fig)
    else:
        benchmark = pd.DataFrame()

    ndcg_cols = [c for c in logs.columns if c.endswith("ndcg@10")] if not logs.empty else []
    mrr_cols = [c for c in logs.columns if c.endswith("mrr@10")] if not logs.empty else []
    return {
        "metrics": metrics,
        "config": config,
        "best_validation_ndcg10": float(logs[ndcg_cols[0]].dropna().max()) if ndcg_cols else None,
        "best_validation_mrr10": float(logs[mrr_cols[0]].dropna().max()) if mrr_cols else None,
        "benchmark": benchmark.to_dict("records") if not benchmark.empty else [],
    }


def query_pgvector(plt, fmt_int) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "src" / "04_pgvector"))
    result: dict[str, Any] = {"available": False}
    try:
        import psycopg
        from config_pgvector import PG_CONN

        with psycopg.connect(**PG_CONN) as conn:
            counts = pd.read_sql(
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
            indexes = pd.read_sql(
                """
                SELECT schemaname, tablename, indexname, indexdef
                FROM pg_indexes
                WHERE tablename IN ('embeddings', 'doc_chunks')
                ORDER BY tablename, indexname
                """,
                conn,
            )
            sizes = pd.read_sql(
                """
                SELECT relname AS objet,
                       pg_total_relation_size(oid)::bigint AS total_bytes,
                       pg_relation_size(oid)::bigint AS table_bytes
                FROM pg_class
                WHERE relname IN ('embeddings', 'doc_chunks', 'emb_hnsw', 'doc_chunks_hnsw')
                ORDER BY total_bytes DESC
                """,
                conn,
            )
            sample = pd.read_sql(
                """
                SELECT entity_kind, entity_id, embedding::text AS embedding_text
                FROM embeddings
                WHERE entity_kind IN ('CANDIDAT', 'OFFRE_EMPLOI', 'COMPETENCE', 'METIER')
                ORDER BY random()
                LIMIT 1800
                """,
                conn,
            )
            qsample = pd.read_sql(
                """
                SELECT embedding::text AS embedding_text
                FROM embeddings
                WHERE entity_kind = 'CANDIDAT'
                ORDER BY random()
                LIMIT 25
                """,
                conn,
            )
            latencies = []
            with conn.cursor() as cur:
                for row in qsample.itertuples():
                    t0 = time.perf_counter()
                    cur.execute(
                        """
                        SELECT entity_id
                        FROM embeddings
                        ORDER BY embedding <=> %s::vector
                        LIMIT 10
                        """,
                        (row.embedding_text,),
                    )
                    cur.fetchall()
                    latencies.append((time.perf_counter() - t0) * 1000)

        save_table(counts, "06_pgvector_counts")
        save_table(docs, "06_pgvector_doc_chunks")
        save_table(indexes, "06_pgvector_indexes")
        sizes["total_mb"] = (sizes["total_bytes"] / 1024 / 1024).round(2)
        sizes["table_mb"] = (sizes["table_bytes"] / 1024 / 1024).round(2)
        save_table(sizes, "06_pgvector_sizes")
        lat_df = pd.DataFrame({"latence_ms_top10": latencies})
        save_table(lat_df.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).reset_index(), "06_pgvector_latency_summary")

        vectors = []
        labels = []
        ids = []
        for row in sample.itertuples():
            vec = vector_from_text(row.embedding_text)
            if vec.size:
                vectors.append(vec)
                labels.append(row.entity_kind)
                ids.append(row.entity_id)
        if vectors:
            x = np.vstack(vectors)
            norms = np.linalg.norm(x, axis=1)
            coords = pca_2d(x)
            emb_stats = pd.DataFrame({"entity_kind": labels, "entity_id": ids, "norme": norms, "pc1": coords[:, 0], "pc2": coords[:, 1]})
            save_table(emb_stats, "07_espace_vectoriel_sample")

            rng = random.Random(42)
            cosine_rows = []
            by_label = {label: np.where(np.array(labels) == label)[0].tolist() for label in sorted(set(labels))}
            for label, idxs in by_label.items():
                for _ in range(min(250, len(idxs) * 2)):
                    if len(idxs) < 2:
                        continue
                    i, j = rng.sample(idxs, 2)
                    cosine_rows.append({"type": f"intra_{label}", "cosine": float(np.dot(x[i], x[j]) / (norms[i] * norms[j]))})
            all_idxs = list(range(len(labels)))
            for _ in range(700):
                i, j = rng.sample(all_idxs, 2)
                if labels[i] != labels[j]:
                    cosine_rows.append({"type": "inter_types", "cosine": float(np.dot(x[i], x[j]) / (norms[i] * norms[j]))})
            cosines = pd.DataFrame(cosine_rows)
            save_table(cosines, "07_cosines_sample")

            fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
            palette = {
                "CANDIDAT": COLORS["blue"],
                "OFFRE_EMPLOI": COLORS["orange"],
                "COMPETENCE": COLORS["green"],
                "METIER": COLORS["purple"],
            }
            for label in sorted(set(labels)):
                mask = emb_stats["entity_kind"] == label
                axes[0].scatter(
                    emb_stats.loc[mask, "pc1"],
                    emb_stats.loc[mask, "pc2"],
                    s=13,
                    alpha=0.62,
                    label=label,
                    color=palette.get(label, COLORS["gray"]),
                )
            axes[0].set_title("Projection PCA des embeddings")
            axes[0].set_xlabel("PC1")
            axes[0].set_ylabel("PC2")
            axes[0].legend(markerscale=1.5, fontsize=8)
            cosines.groupby("type")["cosine"].plot(kind="kde", ax=axes[1])
            axes[1].set_title("Similarités cosinus échantillonnées")
            axes[1].set_xlabel("Cosinus")
            axes[1].legend(fontsize=8)
            save_fig(fig, "07_espace_vectoriel_embeddings")
            plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
        plot = counts.sort_values("n")
        axes[0].barh(plot["entity_kind"], plot["n"], color=COLORS["blue"])
        axes[0].xaxis.set_major_formatter(fmt_int)
        axes[0].set_title("Embeddings par type")
        if not sizes.empty:
            axes[1].barh(sizes["objet"], sizes["total_mb"], color=COLORS["green"])
            axes[1].set_title("Taille des objets")
            axes[1].set_xlabel("Mo")
        if latencies:
            axes[2].boxplot(latencies, patch_artist=True, boxprops={"facecolor": COLORS["light"]}, medianprops={"color": COLORS["red"], "linewidth": 2})
            axes[2].set_title("Latence top-10 HNSW")
            axes[2].set_ylabel("ms")
            axes[2].set_xticklabels(["Requêtes"])
        save_fig(fig, "06_pgvector_indicateurs")
        plt.close(fig)

        result.update(
            {
                "available": True,
                "counts": counts.to_dict("records"),
                "doc_chunks": docs.to_dict("records"),
                "n_indexes": int(len(indexes)),
                "latency_ms_median": float(np.median(latencies)) if latencies else None,
                "latency_ms_p90": float(np.percentile(latencies, 90)) if latencies else None,
            }
        )
        return result
    except Exception as exc:
        result["error"] = str(exc)
        save_json(result, "06_pgvector_error")
        return result


def query_neo4j(plt, fmt_int) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "src" / "03_knowledge_graph"))
    result: dict[str, Any] = {"available": False}
    try:
        from neo4j import GraphDatabase
        from config_neo4j import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session(database=NEO4J_DATABASE) as session:
            labels = pd.DataFrame(
                [dict(r) for r in session.run(
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
                )]
            )
            rels = pd.DataFrame(
                [dict(r) for r in session.run(
                    """
                    MATCH ()-[r]->()
                    RETURN type(r) AS relation, count(r) AS n
                    ORDER BY n DESC
                    """
                )]
            )
            pattern = pd.DataFrame(
                [dict(r) for r in session.run(
                    """
                    MATCH (a)-[r]->(b)
                    RETURN labels(a)[0] AS source, type(r) AS relation, labels(b)[0] AS cible, count(*) AS n
                    ORDER BY n DESC
                    LIMIT 40
                    """
                )]
            )
            degree = pd.DataFrame(
                [dict(r) for r in session.run(
                    """
                    MATCH (n)
                    WITH n, labels(n)[0] AS label
                    OPTIONAL MATCH (n)--()
                    WITH label, n, count(*) AS degree
                    RETURN label,
                           count(*) AS n_nodes,
                           avg(degree) AS avg_degree,
                           percentileCont(degree, 0.5) AS p50_degree,
                           percentileCont(degree, 0.9) AS p90_degree,
                           max(degree) AS max_degree
                    ORDER BY n_nodes DESC
                    """
                )]
            )
            hubs = pd.DataFrame(
                [dict(r) for r in session.run(
                    """
                    MATCH (n)
                    WITH n, labels(n)[0] AS label
                    OPTIONAL MATCH (n)--()
                    WITH n, label, count(*) AS degree
                    RETURN label,
                           coalesce(n.preferredLabel, n.label, n.nom, n.titre_poste, n.id, n.code, n.ville, n.source) AS nom,
                           degree
                    ORDER BY degree DESC
                    LIMIT 30
                    """
                )]
            )
            totals = session.run(
                """
                MATCH (n)
                WITH count(n) AS nodes
                MATCH ()-[r]->()
                RETURN nodes, count(r) AS relationships
                """
            ).single()
            try:
                gds = session.run(
                    "CALL dbms.procedures() YIELD name "
                    "RETURN any(x IN collect(name) WHERE x STARTS WITH 'gds.') AS available"
                ).single()
                gds_available = bool(gds and gds.get("available"))
                gds_error = None
            except Exception as exc:
                gds_available = False
                gds_error = str(exc)
        driver.close()

        totals_dict = dict(totals) if totals else {}
        n_nodes = int(totals_dict.get("nodes", 0))
        n_rels = int(totals_dict.get("relationships", 0))
        density = n_rels / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else math.nan

        save_table(labels, "08_neo4j_node_counts")
        save_table(rels, "08_neo4j_relation_counts")
        save_table(pattern, "08_neo4j_patterns_top")
        save_table(degree, "09_neo4j_degree_stats")
        save_table(hubs, "09_neo4j_top_hubs")

        fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
        plot = labels.head(16).sort_values("n")
        axes[0].barh(plot["label"], plot["n"], color=COLORS["orange"])
        axes[0].xaxis.set_major_formatter(fmt_int)
        axes[0].set_title("Types de noeuds")
        rplot = rels.head(16).sort_values("n")
        axes[1].barh(rplot["relation"], rplot["n"], color=COLORS["teal"])
        axes[1].xaxis.set_major_formatter(fmt_int)
        axes[1].set_title("Types de relations")
        save_fig(fig, "08_neo4j_composition")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
        dplot = degree.sort_values("p90_degree", ascending=False).head(14).sort_values("p90_degree")
        axes[0].barh(dplot["label"], dplot["p90_degree"], color=COLORS["purple"], label="p90")
        axes[0].scatter(dplot["avg_degree"], dplot["label"], color=COLORS["red"], label="moyenne", zorder=3)
        axes[0].set_title("Degré par type de noeud")
        axes[0].set_xlabel("Nombre de relations")
        axes[0].legend()
        hplot = hubs.sort_values("degree").tail(12)
        axes[1].barh(hplot["nom"].astype(str).str[:34], hplot["degree"], color=COLORS["green"])
        axes[1].set_title("Noeuds les plus centraux")
        axes[1].set_xlabel("Degré")
        save_fig(fig, "09_neo4j_degres_hubs")
        plt.close(fig)

        if not pattern.empty:
            mat = pattern.pivot_table(index="source", columns="cible", values="n", aggfunc="sum", fill_value=0)
            fig, ax = plt.subplots(figsize=(9.8, 6.2))
            im = ax.imshow(np.log1p(mat.values), cmap="Blues")
            ax.set_xticks(range(len(mat.columns)))
            ax.set_xticklabels(mat.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(mat.index)))
            ax.set_yticklabels(mat.index)
            ax.set_title("Intensité des liaisons entre familles de noeuds")
            fig.colorbar(im, ax=ax, label="log(1+n)")
            save_fig(fig, "10_neo4j_matrice_liaisons")
            plt.close(fig)

        build_graph_schema_figure(plt)
        result.update(
            {
                "available": True,
                "n_nodes": n_nodes,
                "n_relationships": n_rels,
                "density_directed": density,
                "gds_available": gds_available,
                "gds_error": gds_error,
                "top_labels": labels.head(8).to_dict("records"),
                "top_relations": rels.head(8).to_dict("records"),
            }
        )
        return result
    except Exception as exc:
        result["error"] = str(exc)
        save_json(result, "08_neo4j_error")
        return result


def build_graph_schema_figure(plt) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 6.8))
    ax.axis("off")
    nodes = {
        "Candidat": (0.08, 0.62, COLORS["blue"]),
        "OffreEmploi": (0.42, 0.62, COLORS["orange"]),
        "Compétence": (0.25, 0.35, COLORS["green"]),
        "Métier / MEPC": (0.60, 0.35, COLORS["purple"]),
        "NCF formation": (0.78, 0.62, COLORS["teal"]),
        "Secteur / Employeur /\nLocalisation": (0.42, 0.86, COLORS["gray"]),
        "DocChunk": (0.78, 0.18, COLORS["red"]),
    }
    for label, (x, y, color) in nodes.items():
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=color, edgecolor="none"),
        )

    arrows = [
        ("Candidat", "Compétence", "POSSEDE"),
        ("OffreEmploi", "Compétence", "REQUIERT"),
        ("Candidat", "Métier / MEPC", "VISE / CORRESPOND"),
        ("OffreEmploi", "Métier / MEPC", "CORRESPOND_MEPC"),
        ("Candidat", "NCF formation", "A_NIVEAU"),
        ("OffreEmploi", "NCF formation", "REQUIERT_NIVEAU"),
        ("OffreEmploi", "Secteur / Employeur /\nLocalisation", "DANS / PUBLIEE / LOCALISEE"),
        ("DocChunk", "Métier / MEPC", "EXTRAIT_DE / REFERENCES"),
        ("DocChunk", "NCF formation", "EXTRAIT_DE / REFERENCES"),
        ("Métier / MEPC", "Métier / MEPC", "PARTIE_DE"),
        ("NCF formation", "NCF formation", "CONTIENT"),
    ]
    for src, dst, label in arrows:
        x1, y1, _ = nodes[src]
        x2, y2, _ = nodes[dst]
        if src == dst:
            ax.annotate(
                "",
                xy=(x1 + 0.06, y1 + 0.04),
                xytext=(x1 - 0.06, y1 + 0.04),
                arrowprops=dict(arrowstyle="->", color=COLORS["dark"], lw=1.2, connectionstyle="arc3,rad=0.45"),
            )
            ax.text(x1, y1 + 0.11, label, ha="center", fontsize=8, color=COLORS["dark"])
        else:
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=COLORS["dark"], lw=1.3, alpha=0.75),
            )
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.025, label, ha="center", fontsize=8, color=COLORS["dark"])
    ax.set_title("Schéma logique du graphe de connaissances Neo4j", loc="left", fontsize=15, weight="bold")
    save_fig(fig, "11_schema_graphe_connaissances")
    plt.close(fig)


def build_agentic_rag_analysis(plt) -> dict[str, Any]:
    ablation_path = ROOT / "outputs" / "evaluation" / "pgvector_vs_graph" / "ablation_metrics.csv"
    summary_path = ROOT / "outputs" / "evaluation" / "pgvector_vs_graph" / "ablation_summary.json"
    out: dict[str, Any] = {}
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        save_table(ablation, "12_agentic_retrieval_ablation")
        variant_col = "variant" if "variant" in ablation.columns else "system" if "system" in ablation.columns else None
        metrics = [c for c in ["ndcg@10", "recall@10", "precision@1"] if c in ablation.columns]
        if metrics:
            fig, ax = plt.subplots(figsize=(8.8, 5.2))
            x = np.arange(len(metrics))
            width = 0.36
            variants = ablation[variant_col].unique() if variant_col else ["systeme"]
            for i, variant in enumerate(variants):
                rows = ablation[ablation[variant_col] == variant] if variant_col else ablation
                vals = rows[metrics].iloc[0].values
                ax.bar(x + (i - 0.5) * width, vals, width=width, label=variant)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 0.7)
            ax.set_title("Retrieval : effet du reranking graphe")
            ax.legend()
            save_fig(fig, "12_agentic_retrieval_ablation")
            plt.close(fig)
        out["ablation"] = ablation.to_dict("records")
    if summary_path.exists():
        out["ablation_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))

    generation = pd.DataFrame(
        [
            {"bloc": "Analyse requête", "preuve": "Noeud LangGraph analyse_request", "statut": "implémenté"},
            {"bloc": "Routage outils", "preuve": "Tool registry: pgvector, Neo4j, hybrid recommendation, global summary", "statut": "implémenté"},
            {"bloc": "Construction contexte", "preuve": "build_context agrège résultats/outils avant réponse", "statut": "implémenté"},
            {"bloc": "Génération", "preuve": "LLM OpenRouter avec fallback template local", "statut": "implémenté"},
            {"bloc": "Critic", "preuve": "Vérifie appui contextuel et peut élargir la recherche", "statut": "implémenté"},
            {"bloc": "Score génération", "preuve": "À réserver au chapitre évaluation avec jeux de questions annotées/RAGAS", "statut": "non mesuré ici"},
        ]
    )
    save_table(generation, "13_agentic_generation_diagnostic")
    out["generation_diagnostic"] = generation.to_dict("records")
    return out


def copy_appendix_figures() -> dict[str, str]:
    appendix_dir = FIG / "annexes"
    appendix_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for src_dir in [
        ROOT / "rapport" / "figures" / "generated" / "implementation_results",
        ROOT / "rapport" / "figures" / "generated" / "vrai_data_offres",
    ]:
        if src_dir.exists():
            for src in src_dir.glob("*.png"):
                dst = appendix_dir / src.name
                shutil.copy2(src, dst)
                copied[src.name] = rel(dst)
    return copied


def md_cell(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code_cell(text: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def build_notebook(summary: dict[str, Any]) -> None:
    cells = [
        md_cell(
            "# Analyse approfondie du graphe Neo4j et de pgvector\n"
            "\n"
            "Ce notebook accompagne le chapitre d'implémentation. Il ne remplace pas le chapitre d'évaluation : "
            "il décrit les briques effectivement alimentées, leurs volumes, leur structure et leurs premiers indicateurs techniques.\n"
        ),
        code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "ROOT = Path('..').resolve()\n"
            "OUT = ROOT / 'outputs' / 'memoire_stats' / 'implementation_deep_current'\n"
            "FIG = ROOT / 'rapport' / 'figures' / 'generated' / 'implementation_deep_current'\n"
            "pd.set_option('display.max_colwidth', 120)\n"
        ),
        md_cell("## 1. Description des données\n\nLa description doit rester compacte dans le mémoire : trois sous-sections seulement, avec les figures de détail en annexes."),
        md_cell("### 1.1 Sources des offres\n\nCette sous-section contrôle la dépendance du corpus aux plateformes d'origine et vérifie que le nettoyage de casse ne fragmente pas les sources."),
        code_cell("pd.read_csv(OUT / '01_offres_par_source.csv').head(10)"),
        code_cell("from IPython.display import Image, display\n\ndisplay(Image(filename=str(FIG / '01_offres_par_source.png')))"),
        md_cell("### 1.2 Localisation\n\nLa lecture distingue Cameroun, international et localisations non précisées. C'est nécessaire pour ne pas confondre absence d'information et mobilité internationale."),
        code_cell("pd.read_csv(OUT / '02_zones_localisation.csv')"),
        code_cell("display(Image(filename=str(FIG / '02_localisation_mixte.png')))"),
        md_cell("### 1.3 Compétences et contenu descriptif\n\nLes compétences observées mélangent encore compétences, secteurs et familles métiers. Cette limite doit être assumée dans l'interprétation du skill gap."),
        code_cell("pd.read_csv(OUT / '03_competences_frequentes.csv').head(15)"),
        code_cell("display(Image(filename=str(FIG / '03_competences_descriptions.png')))"),
        md_cell("## 2. Fine-tuning du modèle d'embeddings\n\nLe baseline retenu est `all-MiniLM-L6-v2` parce qu'il produit des embeddings denses de 384 dimensions, rapides à indexer, compatibles avec pgvector, et suffisamment légers pour un pipeline local. Le fine-tuning sert à transformer un modèle généraliste en encodeur spécialisé emploi-compétences."),
        code_cell("pd.read_csv(OUT / '04_finetuning_logs.csv').tail(8)"),
        code_cell("display(Image(filename=str(FIG / '04_finetuning_courbes.png')))"),
        md_cell("### Benchmark multi-modèles\n\nLa comparaison suivante justifie empiriquement l'intérêt du fine-tuning : on ne retient pas le modèle parce qu'il est moderne, mais parce qu'il classe mieux les paires du domaine."),
        code_cell("pd.read_csv(OUT / '05_benchmark_modeles_embeddings.csv').sort_values('ndcg@10', ascending=False)"),
        code_cell("display(Image(filename=str(FIG / '05_benchmark_modeles_embeddings.png')))"),
        md_cell("## 3. Espace vectoriel et pgvector\n\nCette section contrôle la composition de la base vectorielle, la synchronisation avec Neo4j, la présence des index et la distribution géométrique des embeddings."),
        code_cell("pd.read_csv(OUT / '06_pgvector_counts.csv')"),
        code_cell("pd.read_csv(OUT / '06_pgvector_indexes.csv')[['tablename', 'indexname', 'indexdef']]"),
        code_cell("pd.read_csv(OUT / '06_pgvector_latency_summary.csv')"),
        code_cell("display(Image(filename=str(FIG / '06_pgvector_indicateurs.png')))\ndisplay(Image(filename=str(FIG / '07_espace_vectoriel_embeddings.png')))"),
        md_cell("## 4. Graphe de connaissances Neo4j\n\nLe graphe est analysé par types de noeuds, types de relations, motifs source-relation-cible, degrés et hubs. Si Neo4j GDS n'est pas installé, les algorithmes avancés type PageRank/Louvain doivent être reportés ou exécutés après installation de GDS."),
        code_cell("pd.read_csv(OUT / '08_neo4j_node_counts.csv').head(15)"),
        code_cell("pd.read_csv(OUT / '08_neo4j_relation_counts.csv').head(15)"),
        code_cell("display(Image(filename=str(FIG / '11_schema_graphe_connaissances.png')))\ndisplay(Image(filename=str(FIG / '08_neo4j_composition.png')))\ndisplay(Image(filename=str(FIG / '09_neo4j_degres_hubs.png')))\ndisplay(Image(filename=str(FIG / '10_neo4j_matrice_liaisons.png')))"),
        md_cell("## 5. Agentic RAG : retrieval et génération\n\nLe chapitre d'implémentation doit séparer ce qui est mesuré côté retrieval de ce qui reste à évaluer côté génération. Les scores de génération relèvent du chapitre d'évaluation, idéalement avec RAGAS et un jeu de questions annotées."),
        code_cell("pd.read_csv(OUT / '12_agentic_retrieval_ablation.csv')"),
        code_cell("display(Image(filename=str(FIG / '12_agentic_retrieval_ablation.png')))\npd.read_csv(OUT / '13_agentic_generation_diagnostic.csv')"),
        md_cell("## 6. Synthèse reproductible\n\nLe résumé JSON ci-dessous donne les chiffres à reporter dans le mémoire. Toute valeur absente doit être recalculée avant rédaction."),
        code_cell("import json\njson.load(open(OUT / '00_resume_deep_implementation.json', encoding='utf-8'))"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "pygments_lexer": "ipython3"}}, "nbformat": 4, "nbformat_minor": 5}
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    plt, fmt_int = setup_plot()
    data = load_current_data()
    summary = {
        "data": build_data_description(data, plt, fmt_int),
        "fine_tuning": build_training_analysis(plt),
        "pgvector": query_pgvector(plt, fmt_int),
        "neo4j": query_neo4j(plt, fmt_int),
        "agentic_rag": build_agentic_rag_analysis(plt),
        "annex_figures": copy_appendix_figures(),
    }
    save_json(summary, "00_resume_deep_implementation")
    build_notebook(summary)
    print(json.dumps({"notebook": rel(NOTEBOOK), "outputs": rel(OUT), "figures": rel(FIG)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
