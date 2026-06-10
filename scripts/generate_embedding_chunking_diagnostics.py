"""Generate thesis diagnostics for embeddings and document chunking.

The script uses only current project artefacts and the local pgvector database:
- embedding benchmark CSV/JSON for model comparison;
- pgvector `embeddings` and `doc_chunks` tables for PCA and chunk statistics;
- project PDFs for an approximate document-coverage diagnostic.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfplumber
import psycopg
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "memoire_stats" / "implementation_deep_current"
FIG = ROOT / "rapport" / "figures" / "generated" / "implementation_deep_current"
BENCH = ROOT / "outputs" / "evaluation" / "embedding_benchmark.csv"
DIRECT_EVAL = ROOT / "models" / "st_finetuned" / "eval_comparatif" / "evaluation_comparatif.json"
CFG_PATH = ROOT / "src" / "04_pgvector" / "config_pgvector.py"
PDFS = {
    "NCF_2017": ROOT / "pdf" / "Nomenclature-Camerounaise-des-Formations-24.01.2017.pdf",
    "MEPC_2013": ROOT / "pdf" / "Nomenclature-camerounaise-des-metiers-_2013.pdf",
    "diplomes": ROOT / "pdf" / "diplome_certificat.pdf",
}


def load_pg_config():
    spec = importlib.util.spec_from_file_location("pg_cfg", CFG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {CFG_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_vector(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return np.fromstring(text, sep=",", dtype=float)


def count_tokens(text: str) -> int:
    return len(re.findall(r"\w+", str(text or ""), flags=re.UNICODE))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+|\n+", str(text or ""))
    return [p.strip() for p in parts if count_tokens(p) >= 5]


def extract_pdf_tokens(path: Path) -> int:
    if not path.exists():
        return 0
    texts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return count_tokens("\n".join(texts))


def fetch_chunks(conn) -> pd.DataFrame:
    rows = conn.execute(
        """
        select
            chunk_id,
            source,
            coalesce(document_title, '') as document_title,
            coalesce(section_title, '') as section_title,
            coalesce(subsection_title, '') as subsection_title,
            coalesce(chunk_text, '') as chunk_text,
            chunk_index,
            chunk_strategy,
            embedding::text as embedding
        from doc_chunks
        order by source, chunk_index
        """
    ).fetchall()
    cols = [
        "chunk_id",
        "source",
        "document_title",
        "section_title",
        "subsection_title",
        "chunk_text",
        "chunk_index",
        "chunk_strategy",
        "embedding",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["tokens"] = df["chunk_text"].map(count_tokens)
    df["chars"] = df["chunk_text"].fillna("").str.len()
    return df


def fetch_embedding_sample(conn, max_per_kind: int = 350) -> tuple[pd.DataFrame, np.ndarray]:
    rows = conn.execute(
        """
        with ranked as (
            select
                entity_kind,
                entity_id,
                label_fr,
                embedding::text as embedding,
                row_number() over (partition by entity_kind order by random()) as rn
            from embeddings
            where embedding is not null
        )
        select entity_kind, entity_id, label_fr, embedding
        from ranked
        where rn <= %s
        """,
        (max_per_kind,),
    ).fetchall()
    df = pd.DataFrame(rows, columns=["entity_kind", "entity_id", "label_fr", "embedding"])
    vectors = np.vstack([parse_vector(v) for v in df["embedding"]])
    return df.drop(columns=["embedding"]), vectors


def model_comparison_figure() -> pd.DataFrame:
    bench = pd.read_csv(BENCH)
    bench = bench.sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    best = bench.iloc[0]
    base = bench[bench["label"].str.contains("MiniLM", case=False, na=False)]
    if base.empty:
        base = bench[bench["label"].str.contains("ML-MiniLM", case=False, na=False)]

    out = bench.copy()
    out["quality_index"] = out[["ndcg@10", "mrr@10", "recall@10", "precision@1"]].mean(axis=1)
    out["latency_efficiency"] = out["ndcg@10"] / out["latency_ms_per_sentence"].clip(lower=1e-9)
    out["gain_ndcg_vs_best_non_finetuned"] = out["ndcg@10"] / bench.loc[bench.index != 0, "ndcg@10"].max()
    out.to_csv(OUT / "05_embedding_benchmark_extended.csv", index=False)

    labels = out["label"].str.replace("ST Fine-tuné", "ST FT", regex=False)
    x = np.arange(len(out))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    metrics = ["ndcg@10", "mrr@10", "recall@10"]
    colors = ["#1f5aa6", "#0f766e", "#b45309"]
    width = 0.24
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, out[metric], width, label=metric.upper(), color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Score")
    ax.set_title("Qualité de classement par modèle")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax = axes[1]
    sizes = np.sqrt(out["size_mb"].clip(lower=5)) * 18
    scatter = ax.scatter(
        out["latency_ms_per_sentence"],
        out["ndcg@10"],
        s=sizes,
        c=out["recall@10"],
        cmap="viridis",
        edgecolor="#111827",
        linewidth=0.6,
    )
    for _, row in out.iterrows():
        short = row["label"].split("(")[0].replace("ST Fine-tuné", "ST FT").strip()
        ax.annotate(short, (row["latency_ms_per_sentence"], row["ndcg@10"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Latence par phrase, ms (échelle log)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Arbitrage qualité, latence et rappel")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Recall@10")
    fig.suptitle("Benchmark embeddings : performance IR et coût opérationnel", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "05_benchmark_modeles_embeddings.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if DIRECT_EVAL.exists():
        payload = json.loads(DIRECT_EVAL.read_text(encoding="utf-8"))
        direct_rows = []
        for key, label in [
            ("test-ir_cosine_ndcg@10", "NDCG@10"),
            ("test-ir_cosine_mrr@10", "MRR@10"),
            ("test-ir_cosine_recall@1", "Recall@1"),
            ("test-ir_cosine_recall@5", "Recall@5"),
            ("test-ir_cosine_recall@10", "Recall@10"),
            ("test-ir_cosine_precision@1", "Precision@1"),
        ]:
            item = payload[key]
            direct_rows.append(
                {
                    "metric": label,
                    "baseline": item["baseline"],
                    "finetuned": item["finetuned"],
                    "gain": item["delta"],
                    "ratio": item["finetuned"] / item["baseline"] if item["baseline"] else np.nan,
                }
            )
        pd.DataFrame(direct_rows).to_csv(OUT / "05_baseline_vs_finetuned_current.csv", index=False)
    return out


def pca_embedding_diagnostics(conn) -> dict[str, float]:
    sample, vectors = fetch_embedding_sample(conn)
    pca = PCA(n_components=5, random_state=42)
    coords = pca.fit_transform(vectors)
    sample["pc1"] = coords[:, 0]
    sample["pc2"] = coords[:, 1]
    sample.to_csv(OUT / "07_espace_vectoriel_pca_current.csv", index=False)

    var = pca.explained_variance_ratio_
    var_df = pd.DataFrame(
        {
            "axis": [f"PC{i}" for i in range(1, len(var) + 1)],
            "explained_variance_ratio": var,
            "cumulative_variance": np.cumsum(var),
        }
    )
    var_df.to_csv(OUT / "07_pca_variance_expliquee.csv", index=False)

    fig = plt.figure(figsize=(13.0, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.8])
    ax = fig.add_subplot(gs[0, 0])
    for kind, group in sample.groupby("entity_kind"):
        ax.scatter(group["pc1"], group["pc2"], s=14, alpha=0.62, label=kind)
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}% de variance)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}% de variance)")
    ax.set_title("Projection PCA des embeddings indexés")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, ncol=2)

    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(var_df["axis"], var_df["explained_variance_ratio"] * 100, color="#1f5aa6")
    ax2.plot(var_df["axis"], var_df["cumulative_variance"] * 100, marker="o", color="#b42318", label="Cumul")
    ax2.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax2.set_ylim(0, max(10, float(var_df["cumulative_variance"].max() * 115)))
    ax2.set_ylabel("Variance expliquée (%)")
    ax2.set_title("Information portée par les axes")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend()
    fig.suptitle("Espace vectoriel : structure 2D et perte d'information PCA", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "07_espace_vectoriel_embeddings.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "pc1": float(var[0]),
        "pc2": float(var[1]),
        "pc1_pc2": float(var[0] + var[1]),
    }


def umap_embedding_diagnostics(conn) -> dict[str, float]:
    sample, vectors = fetch_embedding_sample(conn)
    reducer = UMAP(
        n_components=2,
        n_neighbors=25,
        min_dist=0.12,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(vectors)
    sample["umap_1"] = coords[:, 0]
    sample["umap_2"] = coords[:, 1]
    sample.to_csv(OUT / "07_espace_vectoriel_umap_current.csv", index=False)

    centroids = (
        sample.groupby("entity_kind")[["umap_1", "umap_2"]]
        .mean()
        .reset_index()
        .rename(columns={"umap_1": "centroid_umap_1", "umap_2": "centroid_umap_2"})
    )
    counts = sample["entity_kind"].value_counts().rename_axis("entity_kind").reset_index(name="n_points")
    diagnostic = centroids.merge(counts, on="entity_kind")
    diagnostic["umap_n_neighbors"] = 25
    diagnostic["umap_min_dist"] = 0.12
    diagnostic["umap_metric"] = "cosine"
    diagnostic.to_csv(OUT / "07_umap_diagnostic.csv", index=False)

    fig = plt.figure(figsize=(13.0, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.8])
    ax = fig.add_subplot(gs[0, 0])
    for kind, group in sample.groupby("entity_kind"):
        ax.scatter(group["umap_1"], group["umap_2"], s=14, alpha=0.62, label=kind)
    for row in diagnostic.itertuples(index=False):
        ax.text(
            row.centroid_umap_1,
            row.centroid_umap_2,
            str(row.entity_kind),
            fontsize=8,
            weight="bold",
            color="#111827",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Projection UMAP des embeddings indexés")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, ncol=2)

    ax2 = fig.add_subplot(gs[0, 1])
    order = counts.sort_values("n_points")
    bars = ax2.barh(order["entity_kind"], order["n_points"], color="#1f5aa6")
    ax2.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    ax2.set_xlabel("Points échantillonnés")
    ax2.set_title("Échantillon utilisé")
    ax2.grid(axis="y", alpha=0.25)
    ax2.text(
        0.02,
        -0.16,
        "UMAP rapproche les voisins locaux ; les axes n'ont pas de sens métier direct.",
        transform=ax2.transAxes,
        fontsize=8.5,
        color="#374151",
        va="top",
    )
    fig.suptitle("Espace vectoriel : voisinages locaux révélés par UMAP", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "07_espace_vectoriel_embeddings.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_points": int(len(sample)),
        "n_neighbors": 25,
        "min_dist": 0.12,
        "metric": "cosine",
    }


def pgvector_retrieval_dashboard() -> dict[str, float | bool]:
    counts_path = OUT / "06_pgvector_counts.csv"
    lat_path = OUT / "06_pgvector_latency_summary.csv"
    sizes_path = OUT / "06_pgvector_sizes.csv"
    indexes_path = OUT / "06_pgvector_indexes.csv"
    docs_path = OUT / "06_pgvector_doc_chunks.csv"
    if not all(path.exists() for path in [counts_path, lat_path, sizes_path, indexes_path, docs_path]):
        return {"available": False}

    counts = pd.read_csv(counts_path)
    lat = pd.read_csv(lat_path)
    sizes = pd.read_csv(sizes_path)
    indexes = pd.read_csv(indexes_path)
    docs = pd.read_csv(docs_path)

    counts["sync_pct"] = counts["n_with_neo4j_id"] / counts["n"].replace(0, np.nan) * 100
    index_family = indexes["indexdef"].str.extract(r"USING\s+(\w+)", expand=False).fillna("unknown").str.lower()
    index_counts = index_family.value_counts().rename_axis("famille").reset_index(name="n")
    lat_map = dict(zip(lat["index"], lat["latence_ms_top10"]))
    hnsw_size = sizes.loc[sizes["objet"].str.contains("hnsw", case=False), "total_mb"].sum()
    table_size = sizes.loc[~sizes["objet"].str.contains("hnsw", case=False), "total_mb"].sum()

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0))
    ax = axes[0, 0]
    plot = counts.sort_values("n")
    ax.barh(plot["entity_kind"], plot["n"], color="#1f5aa6")
    ax.set_title("Couverture des entités vectorisées")
    ax.set_xlabel("Embeddings")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[0, 1]
    ax.barh(plot["entity_kind"], plot["sync_pct"], color="#0f766e")
    ax.set_xlim(0, 105)
    ax.set_title("Synchronisation avec Neo4j")
    ax.set_xlabel("% avec identifiant Neo4j")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1, 0]
    latency_rows = pd.DataFrame(
        {
            "stat": ["p50", "p75", "p90", "max"],
            "ms": [
                lat_map.get("50%", np.nan),
                lat_map.get("75%", np.nan),
                lat_map.get("90%", np.nan),
                lat_map.get("max", np.nan),
            ],
        }
    )
    ax.plot(latency_rows["stat"], latency_rows["ms"], marker="o", color="#b45309", linewidth=2.2)
    ax.set_title("Latence top-10 HNSW")
    ax.set_ylabel("ms")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    bars = ax.bar(index_counts["famille"], index_counts["n"], color="#7c3aed")
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_title("Familles d'index disponibles")
    ax.set_ylabel("Nombre d'index")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.42,
        0.92,
        f"Doc chunks synchronisés : {int(docs['n_with_neo4j_id'].sum())}/{int(docs['n'].sum())}\n"
        f"Stockage tables : {table_size:.1f} Mo ; index HNSW : {hnsw_size:.1f} Mo",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#374151",
        va="top",
    )

    fig.suptitle("pgvector : retrieval, synchronisation graphe et coût opérationnel", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "06_pgvector_retrieval_dashboard.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "available": True,
        "n_embeddings": int(counts["n"].sum()),
        "sync_pct_mean": float(counts["sync_pct"].mean()),
        "latency_p50_ms": float(lat_map.get("50%", np.nan)),
        "latency_p90_ms": float(lat_map.get("90%", np.nan)),
    }


def chunk_diagnostics(chunks: pd.DataFrame) -> dict[str, float]:
    stats = {
        "Nombre total de documents": int(chunks["source"].nunique()),
        "Nombre total de chunks": int(len(chunks)),
        "Taille moyenne des chunks (tokens)": float(chunks["tokens"].mean()),
        "Médiane": float(chunks["tokens"].median()),
        "Écart-type": float(chunks["tokens"].std(ddof=1)),
        "Minimum": int(chunks["tokens"].min()),
        "Maximum": int(chunks["tokens"].max()),
        "Coefficient de variation": float(chunks["tokens"].std(ddof=1) / chunks["tokens"].mean()),
    }
    pd.DataFrame({"indicateur": stats.keys(), "valeur": stats.values()}).to_csv(
        OUT / "06_chunking_stats_descriptives.csv",
        index=False,
    )

    bins = [0, 100, 200, 300, 400, math.inf]
    labels = ["0-100", "100-200", "200-300", "300-400", "400+"]
    chunks["classe_tokens"] = pd.cut(chunks["tokens"], bins=bins, labels=labels, right=False)
    classes = chunks["classe_tokens"].value_counts().reindex(labels).fillna(0).astype(int).reset_index()
    classes.columns = ["classe", "nombre"]
    classes.to_csv(OUT / "06_chunking_distribution_classes.csv", index=False)

    by_source = chunks.groupby("source").agg(
        n_chunks=("chunk_id", "count"),
        tokens_total=("tokens", "sum"),
        tokens_moyens=("tokens", "mean"),
        tokens_medians=("tokens", "median"),
        tokens_min=("tokens", "min"),
        tokens_max=("tokens", "max"),
    ).reset_index()
    by_source.to_csv(OUT / "06_chunking_par_source.csv", index=False)

    coverage_rows = []
    for source, pdf_path in PDFS.items():
        pdf_tokens = extract_pdf_tokens(pdf_path)
        chunk_tokens = int(chunks.loc[chunks["source"] == source, "tokens"].sum())
        coverage_rows.append(
            {
                "source": source,
                "pdf_tokens": pdf_tokens,
                "chunk_tokens": chunk_tokens,
                "coverage_ratio": chunk_tokens / pdf_tokens if pdf_tokens else np.nan,
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT / "06_chunking_couverture_documentaire.csv", index=False)

    vectors = np.vstack([parse_vector(v) for v in chunks["embedding"]])
    sim = cosine_similarity(vectors)
    tri = sim[np.triu_indices_from(sim, k=1)]
    duplication_rate = float((tri > 0.95).sum() / len(tri)) if len(tri) else 0.0
    pd.DataFrame(
        [
            {"indicateur": "similarite_max_hors_diagonale", "valeur": float(tri.max()) if len(tri) else np.nan},
            {"indicateur": "similarite_moyenne_hors_diagonale", "valeur": float(tri.mean()) if len(tri) else np.nan},
            {"indicateur": "duplication_rate_cosine_gt_0_95", "valeur": duplication_rate},
        ]
    ).to_csv(OUT / "06_chunking_duplication.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    ax = axes[0, 0]
    ax.hist(chunks["tokens"], bins=22, color="#1f5aa6", edgecolor="white")
    ax.axvline(chunks["tokens"].median(), color="#b42318", linestyle="--", label=f"Médiane {chunks['tokens'].median():.0f}")
    ax.set_title("Distribution des tailles")
    ax.set_xlabel("Tokens par chunk")
    ax.set_ylabel("Nombre de chunks")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.bar(classes["classe"].astype(str), classes["nombre"], color="#0f766e")
    ax.set_title("Classes de taille")
    ax.set_xlabel("Classe de tokens")
    ax.set_ylabel("Nombre")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    ax.bar(by_source["source"], by_source["n_chunks"], color="#b45309")
    ax.set_title("Chunks par source")
    ax.set_ylabel("Nombre de chunks")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.bar(coverage["source"], coverage["coverage_ratio"] * 100, color="#7c3aed")
    ax.axhline(100, color="#111827", linewidth=1, linestyle="--")
    ax.set_title("Couverture documentaire approximative")
    ax.set_ylabel("Chunks / texte PDF extrait (%)")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Diagnostic statistique du chunking documentaire", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "06_chunking_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_chunks": int(len(chunks)),
        "mean_tokens": float(chunks["tokens"].mean()),
        "median_tokens": float(chunks["tokens"].median()),
        "duplication_rate": duplication_rate,
        "mean_coverage": float(coverage["coverage_ratio"].mean()),
    }


def chunk_semantic_coherence(chunks: pd.DataFrame) -> dict[str, float]:
    cfg = load_pg_config()
    model = SentenceTransformer(str(cfg.MODEL_PATH))
    rows = []
    for row in chunks.itertuples(index=False):
        sentences = split_sentences(row.chunk_text)
        if len(sentences) < 2:
            rows.append(
                {
                    "chunk_id": row.chunk_id,
                    "source": row.source,
                    "n_sentences": len(sentences),
                    "semantic_coherence": np.nan,
                }
            )
            continue
        embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        sims = cosine_similarity(embeddings)
        vals = sims[np.triu_indices_from(sims, k=1)]
        rows.append(
            {
                "chunk_id": row.chunk_id,
                "source": row.source,
                "n_sentences": len(sentences),
                "semantic_coherence": float(vals.mean()),
            }
        )

    coh = pd.DataFrame(rows)
    coh.to_csv(OUT / "06_chunking_coherence_semantique.csv", index=False)
    summary = coh.groupby("source").agg(
        n_chunks=("chunk_id", "count"),
        coherence_mean=("semantic_coherence", "mean"),
        coherence_median=("semantic_coherence", "median"),
        coherence_min=("semantic_coherence", "min"),
        coherence_max=("semantic_coherence", "max"),
    ).reset_index()
    summary.to_csv(OUT / "06_chunking_coherence_par_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    data = [coh.loc[coh["source"] == source, "semantic_coherence"].dropna() for source in sorted(coh["source"].unique())]
    ax.boxplot(data, tick_labels=sorted(coh["source"].unique()), patch_artist=True)
    ax.set_title("Cohérence sémantique intra-chunk")
    ax.set_ylabel("Similarité cosinus moyenne entre phrases")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "06_chunking_coherence_semantique.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "coherence_mean": float(coh["semantic_coherence"].mean()),
        "coherence_median": float(coh["semantic_coherence"].median()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = load_pg_config()

    benchmark = model_comparison_figure()
    with psycopg.connect(**cfg.PG_CONN) as conn:
        chunks = fetch_chunks(conn)
        chunks.to_csv(OUT / "06_chunking_chunks_detail.csv", index=False)
        umap_info = umap_embedding_diagnostics(conn)

    chunk_info = chunk_diagnostics(chunks)
    coherence_info = chunk_semantic_coherence(chunks)
    pgvector_info = pgvector_retrieval_dashboard()

    summary = {
        "benchmark_rows": int(len(benchmark)),
        "umap": umap_info,
        "pgvector_retrieval": pgvector_info,
        "chunking": chunk_info,
        "chunking_coherence": coherence_info,
    }
    (OUT / "06_07_embedding_chunking_diagnostics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
