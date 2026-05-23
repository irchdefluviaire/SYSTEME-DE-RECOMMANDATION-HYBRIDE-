"""Domain tools exposed to the Agentic GraphRAG workflow.

The functions wrap the existing project modules and require live Neo4j +
PostgreSQL/pgvector connections for retrieval and graph enrichment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

from settings import DEFAULT_TOP_K, USE_REAL_DBS

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC / "05_graphrag"))
sys.path.insert(0, str(SRC / "03_knowledge_graph"))
sys.path.insert(0, str(SRC / "04_pgvector"))

from context_builder import GraphRAGContextBuilder  # noqa: E402
from roadmap_generator import generate_roadmap  # noqa: E402


def _trace(step: str, message: str, status: str = "ok", **details: Any) -> dict:
    return {"step": step, "status": status, "message": message, "details": details}


def load_candidate_profile(candidat_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one normalized candidate profile from local processed data."""

    df = pd.read_parquet(ROOT / "data" / "processed" / "candidats_normalized.parquet")
    row = df[df["candidat_id"].astype(str) == str(candidat_id)]
    if row.empty:
        row = df.iloc[[0]]
        status = "warning"
        message = f"Candidat {candidat_id} introuvable, premier profil utilise."
    else:
        status = "ok"
        message = f"Candidat {candidat_id} charge."
    r = row.iloc[0]
    profile = {
        "candidat_id": str(r.get("candidat_id", candidat_id)),
        "metier_vise": str(r.get("metier_vise", "") or ""),
        "secteur_metier": str(r.get("secteur_metier", "") or ""),
        "secteur_demande": str(r.get("secteur_demande", "") or ""),
        "ncf_niveau_final": (
            int(r["ncf_niveau_final"]) if pd.notna(r.get("ncf_niveau_final")) else None
        ),
        "filiere_specialite": str(r.get("filiere_specialite", "") or ""),
        "diplome_raw": str(r.get("diplome_raw", "") or ""),
        "objectif": str(r.get("objectif", "") or "")[:250],
        "mobilite_geo_bool": bool(r.get("mobilite_geo_bool"))
        if pd.notna(r.get("mobilite_geo_bool"))
        else None,
        "text_to_embed": str(r.get("text_to_embed", "") or ""),
    }
    return profile, _trace("load_candidate", message, status, candidat_id=profile["candidat_id"])


def build_retrieval_context(
    candidat_id: str,
    candidate_profile: dict[str, Any],
    *,
    top_k: int = DEFAULT_TOP_K,
    use_real_dbs: bool = USE_REAL_DBS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run vector retrieval + graph enrichment through the existing builder."""

    if not use_real_dbs:
        raise RuntimeError("Le mode simulation est desactive: Agentic GraphRAG exige Neo4j et pgvector.")
    neo4j_driver, pg_conn = _connect_required_databases()

    builder = GraphRAGContextBuilder(
        neo4j_driver=neo4j_driver,
        pg_conn=pg_conn,
        st_model=None,
        top_k_pgvector=max(20, top_k * 4),
        top_k_final=top_k,
    )
    context = builder.build_context(candidat_id, candidate_profile)
    traces = [
        _trace(
            "retrieval_vector_graph",
            "Recherche vectorielle et enrichissement graphe termines.",
            n_candidates=context.get("n_candidats", 0),
            n_top=len(context.get("top_offres", [])),
            real_databases=True,
        )
    ]
    _close_optional(pg_conn=pg_conn, neo4j_driver=neo4j_driver)
    return context, traces


def extract_skill_gaps(top_offers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict]:
    """Normalize skill-gap fields for the critic and UI."""

    gaps: list[dict[str, Any]] = []
    for offer in top_offers:
        gaps.append(
            {
                "offre_id": offer.get("offre_id"),
                "titre": offer.get("titre"),
                "taux_match": offer.get("taux_match", 0),
                "competences_acquises": offer.get("acquises", []),
                "competences_manquantes": offer.get("manquantes", []),
                "essentielles_manquantes": offer.get("ess_manq", []),
                "nb_manquantes": len(offer.get("manquantes", [])),
                "nb_essentielles_manquantes": len(offer.get("ess_manq", [])),
            }
        )
    return gaps, _trace("skill_gap", "Skill gaps normalises.", n=len(gaps))


def rank_hybrid_offers(top_offers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict]:
    """Sort offers by the hybrid score computed by the GraphRAG builder."""

    ranked = sorted(top_offers, key=lambda item: item.get("score_hybride", 0), reverse=True)
    return ranked, _trace("hybrid_scoring", "Offres classees par score hybride.", n=len(ranked))


def build_training_roadmap(
    candidate_profile: dict[str, Any],
    top_offer: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate a structured training roadmap for the best offer."""

    missing = [
        {"label": label, "importance": "essential" if label in top_offer.get("ess_manq", []) else "optional"}
        for label in top_offer.get("manquantes", [])
    ]
    roadmap = generate_roadmap(
        candidat=candidate_profile,
        top_offre=top_offer,
        competences_manquantes=missing,
        score_actuel=float(top_offer.get("score_hybride", 0.0)),
    )
    return roadmap, _trace("roadmap", "Roadmap de formation generee.", n_missing=len(missing))


def _connect_required_databases():
    """Open and validate required Neo4j and PostgreSQL/pgvector connections."""

    neo4j_driver = None
    pg_conn = None
    try:
        from neo4j import GraphDatabase
        from config_neo4j import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        neo4j_driver.verify_connectivity()
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            session.run("RETURN 1 AS ok").single()
    except Exception as exc:
        _close_optional(pg_conn=pg_conn, neo4j_driver=neo4j_driver)
        raise RuntimeError(f"Connexion Neo4j impossible. Verifie NEO4J_URI/USER/PASSWORD: {exc}") from exc

    try:
        import psycopg
        from config_pgvector import PG_CONN

        pg_conn = psycopg.connect(**PG_CONN)
        with pg_conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_name = 'embeddings'
                )
                """
            )
            has_embeddings = bool(cur.fetchone()[0])
        if not has_embeddings:
            raise RuntimeError("table PostgreSQL 'embeddings' introuvable")
    except Exception as exc:
        _close_optional(pg_conn=pg_conn, neo4j_driver=neo4j_driver)
        raise RuntimeError(f"Connexion pgvector impossible. Verifie PG_HOST/PG_DB/PG_USER/PG_PASSWORD: {exc}") from exc

    return neo4j_driver, pg_conn


def _close_optional(*, pg_conn=None, neo4j_driver=None) -> None:
    for obj in (pg_conn, neo4j_driver):
        if obj is not None:
            try:
                obj.close()
            except Exception:
                pass
