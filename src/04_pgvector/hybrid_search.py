"""
hybrid_search.py
===========================================================================
Module 04 - Recherche hybride dense + lexicale sur PostgreSQL/pgvector.

Cette couche complete l'ANN vectoriel existant par un signal lexical
PostgreSQL full-text. Elle reste volontairement autonome: aucun schema
additionnel n'est requis, meme si des index GIN peuvent etre ajoutes plus tard
pour accelerer les requetes lexicales.
===========================================================================
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _embedding_literal(vec) -> str:
    return "[" + ",".join(f"{float(v):.6f}" for v in vec) + "]"


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    id_key: str = "entity_id",
    k: int = 60,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Fusionne plusieurs classements par Reciprocal Rank Fusion."""

    scores: dict[str, float] = defaultdict(float)
    merged: dict[str, dict[str, Any]] = {}
    for rows in ranked_lists:
        for rank, row in enumerate(rows, 1):
            item_id = str(row.get(id_key, ""))
            if not item_id:
                continue
            scores[item_id] += 1.0 / (k + rank)
            merged.setdefault(item_id, {}).update(row)

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    output = []
    for item_id, score in ordered:
        row = merged[item_id]
        row["rrf_score"] = round(float(score), 6)
        output.append(row)
    return output


def dense_entity_search(conn, model, query_text: str, entity_kind: str, top_k: int = 20) -> list[dict]:
    """Recherche vectorielle depuis un texte libre."""

    vec = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)[0]
    emb = _embedding_literal(vec)
    sql = """
        SELECT entity_id, label_fr, source_system,
               1 - (embedding <=> %s::vector) AS dense_score,
               neo4j_node_id
        FROM embeddings
        WHERE entity_kind = %s::entity_kind
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (emb, entity_kind, emb, top_k))
        rows = cur.fetchall()
    return [
        {
            "entity_id": r[0],
            "label": r[1],
            "source_system": r[2],
            "dense_score": round(float(r[3]), 4),
            "neo4j_node_id": r[4],
            "retriever": "dense",
        }
        for r in rows
    ]


def lexical_entity_search(conn, query_text: str, entity_kind: str, top_k: int = 20) -> list[dict]:
    """Recherche lexicale PostgreSQL full-text sans colonne materialisee."""

    sql = """
        WITH q AS (SELECT websearch_to_tsquery('french', %s) AS query)
        SELECT e.entity_id, e.label_fr, e.source_system,
               ts_rank_cd(
                 to_tsvector('french', coalesce(e.label_fr, '') || ' ' || coalesce(e.text_to_embed, '')),
                 q.query
               ) AS lexical_score,
               e.neo4j_node_id
        FROM embeddings e, q
        WHERE e.entity_kind = %s::entity_kind
          AND to_tsvector('french', coalesce(e.label_fr, '') || ' ' || coalesce(e.text_to_embed, '')) @@ q.query
        ORDER BY lexical_score DESC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (query_text, entity_kind, top_k))
        rows = cur.fetchall()
    return [
        {
            "entity_id": r[0],
            "label": r[1],
            "source_system": r[2],
            "lexical_score": round(float(r[3] or 0.0), 4),
            "neo4j_node_id": r[4],
            "retriever": "lexical",
        }
        for r in rows
    ]


def hybrid_entity_search(conn, model, query_text: str, entity_kind: str, top_k: int = 20) -> list[dict]:
    """Recherche hybride texte libre: dense + lexical + RRF."""

    dense = dense_entity_search(conn, model, query_text, entity_kind, top_k=top_k * 3)
    try:
        lexical = lexical_entity_search(conn, query_text, entity_kind, top_k=top_k * 3)
    except Exception:
        conn.rollback()
        lexical = []
    fused = reciprocal_rank_fusion([dense, lexical], top_k=top_k)

    dense_by_id = {r["entity_id"]: r for r in dense}
    lex_by_id = {r["entity_id"]: r for r in lexical}
    for row in fused:
        item_id = row["entity_id"]
        row["dense_score"] = dense_by_id.get(item_id, {}).get("dense_score", row.get("dense_score", 0.0))
        row["lexical_score"] = lex_by_id.get(item_id, {}).get("lexical_score", row.get("lexical_score", 0.0))
        row["retriever"] = "hybrid_rrf"
    return fused


def candidate_offer_hybrid_search(conn, model, candidat_id: str, top_k: int = 20) -> list[dict]:
    """Recherche hybride candidat -> offres.

    Le signal dense compare le vecteur candidat aux vecteurs d'offres. Le signal
    lexical utilise le texte du candidat comme requete full-text contre les
    offres. Les deux classements sont fusionnes par RRF.
    """

    dense_sql = """
        SELECT o.entity_id, o.label_fr, o.neo4j_node_id,
               1 - (c.embedding <=> o.embedding) AS dense_score
        FROM embeddings c, embeddings o
        WHERE c.entity_kind = 'CANDIDAT'
          AND c.entity_id = %s
          AND o.entity_kind = 'OFFRE_EMPLOI'
        ORDER BY c.embedding <=> o.embedding
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute("SET hnsw.ef_search = 100")
        cur.execute(dense_sql, (candidat_id, top_k * 3))
        dense_rows = cur.fetchall()
        cur.execute(
            """
            SELECT coalesce(label_fr, '') || ' ' || coalesce(text_to_embed, '')
            FROM embeddings
            WHERE entity_kind = 'CANDIDAT' AND entity_id = %s
            LIMIT 1
            """,
            (candidat_id,),
        )
        qrow = cur.fetchone()

    dense = [
        {
            "entity_id": r[0],
            "offre_id": r[0],
            "label": r[1],
            "titre": r[1],
            "neo4j_id": r[2],
            "dense_score": round(float(r[3]), 4),
            "score_sem": round(float(r[3]), 4),
            "retriever": "dense",
        }
        for r in dense_rows
    ]

    query_text = qrow[0] if qrow else candidat_id
    try:
        lexical_raw = lexical_entity_search(conn, query_text, "OFFRE_EMPLOI", top_k=top_k * 3)
    except Exception:
        conn.rollback()
        lexical_raw = []
    lexical = [
        {
            **r,
            "offre_id": r["entity_id"],
            "titre": r.get("label"),
            "neo4j_id": r.get("neo4j_node_id"),
            "score_sem": 0.0,
        }
        for r in lexical_raw
    ]

    fused = reciprocal_rank_fusion([dense, lexical], top_k=top_k, id_key="entity_id")
    dense_by_id = {r["entity_id"]: r for r in dense}
    lex_by_id = {r["entity_id"]: r for r in lexical}
    output = []
    for row in fused:
        item_id = row["entity_id"]
        d = dense_by_id.get(item_id, {})
        l = lex_by_id.get(item_id, {})
        dense_score = d.get("dense_score", row.get("dense_score", 0.0))
        output.append(
            {
                "offre_id": item_id,
                "titre": row.get("titre") or row.get("label"),
                "neo4j_id": row.get("neo4j_id") or row.get("neo4j_node_id"),
                "score_sem": round(float(dense_score or 0.0), 4),
                "score_dense": round(float(dense_score or 0.0), 4),
                "score_lexical": round(float(l.get("lexical_score", row.get("lexical_score", 0.0)) or 0.0), 4),
                "rrf_score": row["rrf_score"],
                "retriever": "hybrid_rrf",
            }
        )
    return output
