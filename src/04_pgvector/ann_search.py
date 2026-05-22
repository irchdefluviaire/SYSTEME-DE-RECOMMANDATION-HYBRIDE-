"""
ann_search.py
===========================================================================
Module 04 — Fonctions de recherche ANN et évaluation pgvector

Fournit :
  - Recherche ANN optimisée (candidat → offres, compétences similaires)
  - Évaluation du recall ANN vs exact
  - Benchmark de latence
  - Analyse des similarités par secteur

Usage :
  python ann_search.py --eval          # évaluation recall ANN
  python ann_search.py --benchmark     # benchmark latence
  python ann_search.py --query "texte" # requête interactive
===========================================================================
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────────────────────────────────────────────────────────────────
# REQUÊTES ANN PGVECTOR
# ─────────────────────────────────────────────────────────────────────────

SQL_ANN_OFFRES = """
    SELECT
        o.entity_id                              AS offre_id,
        o.label_fr                               AS titre,
        1 - (c.embedding <=> o.embedding)        AS cosine_sim,
        o.neo4j_node_id                          AS neo4j_id
    FROM   embeddings c
    CROSS  JOIN LATERAL (
        SELECT entity_id, label_fr, embedding, neo4j_node_id
        FROM   embeddings
        WHERE  entity_kind = 'OFFRE_EMPLOI'
        ORDER  BY embedding <=> c.embedding
        LIMIT  %s
    ) o
    WHERE  c.entity_kind = 'CANDIDAT'
    AND    c.entity_id   = %s
"""

SQL_ANN_COMPETENCES = """
    SELECT
        s2.entity_id                             AS comp_uri,
        s2.label_fr                              AS label,
        1 - (s1.embedding <=> s2.embedding)      AS cosine_sim
    FROM   embeddings s1, embeddings s2
    WHERE  s1.entity_kind = 'COMPETENCE'
    AND    s1.entity_id   = %s
    AND    s2.entity_kind = 'COMPETENCE'
    AND    s2.entity_id  <> %s
    ORDER  BY s1.embedding <=> s2.embedding
    LIMIT  %s
"""

SQL_ANN_METIERS_POUR_CANDIDAT = """
    SELECT
        m.entity_id                              AS metier_uri,
        m.label_fr                               AS label,
        1 - (c.embedding <=> m.embedding)        AS cosine_sim
    FROM   embeddings c, embeddings m
    WHERE  c.entity_kind = 'CANDIDAT'
    AND    c.entity_id   = %s
    AND    m.entity_kind = 'METIER'
    ORDER  BY c.embedding <=> m.embedding
    LIMIT  %s
"""

SQL_ANN_NCF_POUR_OFFRE = """
    SELECT
        n.entity_id                              AS ncf_code,
        n.label_fr                               AS label,
        1 - (o.embedding <=> n.embedding)        AS cosine_sim
    FROM   embeddings o, embeddings n
    WHERE  o.entity_kind = 'OFFRE_EMPLOI'
    AND    o.entity_id   = %s
    AND    n.entity_kind = 'DOMAINE_DETAILLE_NCF'
    ORDER  BY o.embedding <=> n.embedding
    LIMIT  %s
"""

SQL_STATS_PGVECTOR = """
    SELECT entity_kind, COUNT(*) AS n,
           AVG(LENGTH(text_to_embed))::INT AS moy_chars
    FROM   embeddings
    GROUP  BY entity_kind ORDER BY entity_kind
"""


# ─────────────────────────────────────────────────────────────────────────
# FONCTIONS DE RECHERCHE
# ─────────────────────────────────────────────────────────────────────────

def candidat_top_offres(conn, candidat_id: str, top_k: int = 20) -> list[dict]:
    """Top-K offres pour un candidat par cosine similarity."""
    with conn.cursor() as cur:
        cur.execute(SQL_ANN_OFFRES, (top_k, candidat_id))
        rows = cur.fetchall()
    return [{"offre_id": r[0], "titre": r[1], "cosine_sim": round(r[2], 4),
             "neo4j_id": r[3]} for r in rows]


def competences_similaires(conn, comp_uri: str, top_k: int = 10) -> list[dict]:
    """Compétences les plus proches d'une compétence ESCO."""
    with conn.cursor() as cur:
        cur.execute(SQL_ANN_COMPETENCES, (comp_uri, comp_uri, top_k))
        rows = cur.fetchall()
    return [{"uri": r[0], "label": r[1], "cosine_sim": round(r[2], 4)}
            for r in rows]


def metiers_pour_candidat(conn, candidat_id: str, top_k: int = 10) -> list[dict]:
    """Métiers ESCO les plus proches du profil d'un candidat."""
    with conn.cursor() as cur:
        cur.execute(SQL_ANN_METIERS_POUR_CANDIDAT, (candidat_id, top_k))
        rows = cur.fetchall()
    return [{"uri": r[0], "label": r[1], "cosine_sim": round(r[2], 4)}
            for r in rows]


def ncf_pour_offre(conn, offre_id: str, top_k: int = 5) -> list[dict]:
    """Domaines NCF les plus proches d'une offre (pour la roadmap)."""
    with conn.cursor() as cur:
        cur.execute(SQL_ANN_NCF_POUR_OFFRE, (offre_id, top_k))
        rows = cur.fetchall()
    return [{"code": r[0], "label": r[1], "cosine_sim": round(r[2], 4)}
            for r in rows]


def ann_from_text(conn, model, query_text: str,
                  entity_kind: str = "OFFRE_EMPLOI",
                  top_k: int = 20) -> list[dict]:
    """Recherche ANN depuis un texte libre (encode puis cherche)."""
    vec = model.encode([query_text], normalize_embeddings=True,
                        convert_to_numpy=True)[0]
    emb_str = "[" + ",".join(f"{v:.6f}" for v in vec) + "]"

    sql = """
        SELECT entity_id, label_fr,
               1 - (embedding <=> %s::vector) AS sim
        FROM   embeddings
        WHERE  entity_kind = %s
        ORDER  BY embedding <=> %s::vector
        LIMIT  %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (emb_str, entity_kind, emb_str, top_k))
        rows = cur.fetchall()
    return [{"entity_id": r[0], "label": r[1], "cosine_sim": round(r[2], 4)}
            for r in rows]


# ─────────────────────────────────────────────────────────────────────────
# ÉVALUATION DU RECALL ANN
# ─────────────────────────────────────────────────────────────────────────

def evaluate_ann_recall(conn, model, n_queries: int = 100) -> dict:
    """
    Évalue le recall@K de l'index HNSW par rapport à la recherche exacte.
    Compare les résultats ANN (HNSW) vs recherche exhaustive brute-force.

    Recall@K = |ANN_top_k ∩ Exact_top_k| / K

    Seuil cible : Recall@10 > 0.97
    """
    log.info(f"Évaluation recall ANN sur {n_queries} requêtes...")

    # Charger des candidats aléatoires comme requêtes
    df_cand = pd.read_parquet(ROOT / "data" / "processed" / "candidats_normalized.parquet")
    sample  = df_cand.sample(n=min(n_queries, len(df_cand)), random_state=42)

    recalls_k = {1: [], 5: [], 10: [], 20: []}
    latences  = []

    for _, row in sample.iterrows():
        text = str(row.get("text_to_embed", ""))
        if not text: continue

        # Encoder
        vec = model.encode([text], normalize_embeddings=True,
                            convert_to_numpy=True)[0]
        emb_str = "[" + ",".join(f"{v:.6f}" for v in vec) + "]"

        # ANN (HNSW)
        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_id FROM embeddings
                WHERE entity_kind = 'OFFRE_EMPLOI'
                ORDER BY embedding <=> %s::vector
                LIMIT 20
            """, (emb_str,))
            ann_results = {r[0] for r in cur.fetchall()}
        latences.append((time.time() - t0) * 1000)

        # Exact (sans index — SET LOCAL hnsw.ef_search = 0)
        with conn.cursor() as cur:
            cur.execute("SET LOCAL enable_indexscan = off")
            cur.execute("""
                SELECT entity_id FROM embeddings
                WHERE entity_kind = 'OFFRE_EMPLOI'
                ORDER BY embedding <=> %s::vector
                LIMIT 20
            """, (emb_str,))
            exact_results = [r[0] for r in cur.fetchall()]

        for k in recalls_k:
            exact_k = set(exact_results[:k])
            ann_k   = set(list(ann_results)[:k])
            recalls_k[k].append(len(exact_k & ann_k) / k)

    results = {
        "n_queries":   len(latences),
        "latence_moy": round(np.mean(latences), 2),
        "latence_p95": round(np.percentile(latences, 95), 2),
        "latence_max": round(max(latences), 2),
    }
    for k, vals in recalls_k.items():
        results[f"recall@{k}"] = round(np.mean(vals), 4)

    log.info("\n  Résultats Recall ANN (HNSW m=16, ef=64) :")
    for k in [1, 5, 10, 20]:
        log.info(f"    Recall@{k:<2} = {results[f'recall@{k}']:.4f}")
    log.info(f"  Latence : moy={results['latence_moy']}ms "
             f"p95={results['latence_p95']}ms")

    return results


# ─────────────────────────────────────────────────────────────────────────
# BENCHMARK DE LATENCE
# ─────────────────────────────────────────────────────────────────────────

def benchmark_latency(conn, n_queries: int = 200) -> dict:
    """Benchmark de latence de la recherche ANN sur N requêtes aléatoires."""
    log.info(f"Benchmark latence sur {n_queries} requêtes...")

    df_cand = pd.read_parquet(ROOT / "data" / "processed" / "candidats_normalized.parquet")

    # Pré-calculer quelques vecteurs
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(ROOT / "models" / "st_finetuned" / "final"))

    texts = df_cand["text_to_embed"].dropna().sample(
        n=min(n_queries, len(df_cand)), random_state=42
    ).tolist()

    vecs = model.encode(texts, normalize_embeddings=True,
                         batch_size=32, convert_to_numpy=True)

    latences_ann  = []
    latences_k20  = []

    for vec in vecs:
        emb_str = "[" + ",".join(f"{v:.6f}" for v in vec) + "]"

        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_id FROM embeddings
                WHERE entity_kind = 'OFFRE_EMPLOI'
                ORDER BY embedding <=> %s::vector LIMIT 10
            """, (emb_str,))
            cur.fetchall()
        latences_ann.append((time.time() - t0) * 1000)

        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_id FROM embeddings
                WHERE entity_kind = 'OFFRE_EMPLOI'
                ORDER BY embedding <=> %s::vector LIMIT 20
            """, (emb_str,))
            cur.fetchall()
        latences_k20.append((time.time() - t0) * 1000)

    results = {
        "n_queries": len(latences_ann),
        "top10_moy_ms":    round(np.mean(latences_ann), 2),
        "top10_p50_ms":    round(np.percentile(latences_ann, 50), 2),
        "top10_p95_ms":    round(np.percentile(latences_ann, 95), 2),
        "top20_moy_ms":    round(np.mean(latences_k20), 2),
        "top20_p95_ms":    round(np.percentile(latences_k20, 95), 2),
    }
    log.info(f"\n  Top-10 : moy={results['top10_moy_ms']}ms "
             f"p50={results['top10_p50_ms']}ms "
             f"p95={results['top10_p95_ms']}ms")
    log.info(f"  Top-20 : moy={results['top20_moy_ms']}ms "
             f"p95={results['top20_p95_ms']}ms")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",      action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--query",     type=str, default=None)
    parser.add_argument("--kind",      type=str, default="OFFRE_EMPLOI")
    parser.add_argument("--top-k",     type=int, default=10)
    args = parser.parse_args()

    import psycopg2
    from embed_all_entities import PG_CONN, load_model, DEFAULT_MODEL_PATH
    conn = psycopg2.connect(**PG_CONN)

    if args.eval:
        model = load_model(DEFAULT_MODEL_PATH)
        evaluate_ann_recall(conn, model)
    elif args.benchmark:
        benchmark_latency(conn)
    elif args.query:
        model = load_model(DEFAULT_MODEL_PATH)
        results = ann_from_text(conn, model, args.query, args.kind, args.top_k)
        log.info(f"\nRésultats pour : '{args.query}'")
        for i, r in enumerate(results, 1):
            log.info(f"  {i:2}. {r['label'][:50]:<50} sim={r['cosine_sim']:.4f}")
    else:
        log.info("Usage : --eval | --benchmark | --query 'texte'")

    conn.close()


if __name__ == "__main__":
    main()
