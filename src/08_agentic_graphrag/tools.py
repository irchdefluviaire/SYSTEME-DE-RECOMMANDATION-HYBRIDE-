"""
Tools for the LangGraph Agentic GraphRAG workflow.

The graph database and pgvector database are exposed here as explicit tools.
Each tool returns plain dictionaries so LangGraph Studio, the CLI and Streamlit
can show what was called and what evidence was retrieved.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
SRC_05 = ROOT / "src" / "05_graphrag"
SRC_04 = ROOT / "src" / "04_pgvector"
SRC_03 = ROOT / "src" / "03_knowledge_graph"
for module_path in (SRC_05, SRC_04, SRC_03):
    module_path_str = str(module_path)
    if module_path_str not in sys.path:
        sys.path.insert(0, module_path_str)

load_dotenv(ROOT / ".env")

from document_retriever import ParentDocumentRetriever  # noqa: E402
from hybrid_search import hybrid_entity_search  # noqa: E402
from recommendation_engine import RecommendationEngine  # noqa: E402
from text2cypher import run_text2cypher  # noqa: E402

_PG_CONN = None
_NEO4J_DRIVER = None
_ST_MODEL = None


def _pg_dsn_from_env() -> str:
    if os.getenv("PG_DSN"):
        return str(os.getenv("PG_DSN"))
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db = os.getenv("PG_DB", "test_kmer")
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def get_pg_conn():
    global _PG_CONN
    if _PG_CONN is not None:
        try:
            if not _PG_CONN.closed:
                _PG_CONN.rollback()
                return _PG_CONN
        except Exception:
            _PG_CONN = None
    import psycopg

    _PG_CONN = psycopg.connect(_pg_dsn_from_env(), connect_timeout=5)
    _PG_CONN.autocommit = False
    return _PG_CONN


def get_neo4j_driver():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is not None:
        return _NEO4J_DRIVER
    from neo4j import GraphDatabase

    _NEO4J_DRIVER = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
        connection_timeout=5,
    )
    _NEO4J_DRIVER.verify_connectivity()
    return _NEO4J_DRIVER


def get_st_model():
    global _ST_MODEL
    if _ST_MODEL is not None:
        return _ST_MODEL
    from sentence_transformers import SentenceTransformer

    model_path = Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "st_finetuned" / "final")))
    if model_path.exists() and (model_path / "modules.json").exists():
        _ST_MODEL = SentenceTransformer(str(model_path))
    else:
        _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _ST_MODEL


def tool_service_status(_: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check pgvector, Neo4j and embedding model availability."""

    status: dict[str, str] = {}
    try:
        with get_pg_conn().cursor() as cur:
            cur.execute("SELECT entity_kind::text, count(*) FROM embeddings GROUP BY entity_kind ORDER BY entity_kind")
            rows = cur.fetchall()
        status["pgvector"] = "connected: " + ", ".join(f"{k}={n}" for k, n in rows)
    except Exception as exc:
        status["pgvector"] = f"unavailable: {exc}"

    try:
        with get_neo4j_driver().session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            n_nodes = session.run("MATCH (n) RETURN count(n) AS n").single()["n"]
        status["neo4j"] = f"connected: nodes={n_nodes}"
    except Exception as exc:
        status["neo4j"] = f"unavailable: {exc}"

    model_path = Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "st_finetuned" / "final")))
    if model_path.exists() and (model_path / "modules.json").exists():
        status["st_model"] = f"configured: {model_path}"
    else:
        status["st_model"] = "configured: sentence-transformers/all-MiniLM-L6-v2 fallback"

    or_key = os.getenv("API_KEY_OPEN_ROUTEUR") or os.getenv("OPENROUTER_API_KEY", "")
    or_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
    status["llm_backend"] = f"openrouter:{or_model}" if or_key else "openrouter:missing_api_key"
    return {"tool": "service_status", "status": status}


def tool_pgvector_semantic_search(args: dict[str, Any]) -> dict[str, Any]:
    """Search business entities in PostgreSQL/pgvector."""

    query = str(args.get("query", "")).strip()
    kinds = args.get("kinds") or ["OFFRE_EMPLOI", "METIER", "COMPETENCE"]
    top_k = int(args.get("top_k") or 5)
    conn = get_pg_conn()
    model = get_st_model()
    results = {}
    for kind in kinds:
        results[str(kind)] = hybrid_entity_search(conn, model, query, str(kind), top_k)
    return {"tool": "pgvector_semantic_search", "query": query, "results": results}


def tool_pgvector_document_search(args: dict[str, Any]) -> dict[str, Any]:
    """Search indexed reference documents in PostgreSQL/pgvector."""

    query = str(args.get("query", "")).strip()
    top_k = int(args.get("top_k") or 5)
    conn = get_pg_conn()
    model = get_st_model()
    docs = ParentDocumentRetriever(conn, model, child_k=top_k).retrieve(query, top_k=top_k)
    return {"tool": "pgvector_document_search", "query": query, "documents": docs}


def tool_neo4j_graph_query(args: dict[str, Any]) -> dict[str, Any]:
    """Run a safe read-only text-to-Cypher graph query against Neo4j."""

    query = str(args.get("query", "")).strip()
    result = run_text2cypher(
        get_neo4j_driver(),
        query,
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    return {"tool": "neo4j_graph_query", "query": query, **result}


def tool_hybrid_candidate_recommendation(args: dict[str, Any]) -> dict[str, Any]:
    """Run the existing hybrid recommender as a tool using Neo4j and pgvector."""

    candidat_id = str(args.get("candidat_id", "")).strip()
    top_k = int(args.get("top_k") or 5)
    or_key = os.getenv("API_KEY_OPEN_ROUTEUR") or os.getenv("OPENROUTER_API_KEY", "")
    backend = args.get("backend") or "openrouter"
    engine = RecommendationEngine(
        neo4j_driver=get_neo4j_driver(),
        pg_conn=get_pg_conn(),
        st_model=get_st_model(),
        llm_backend=backend,
        top_k=top_k,
    )
    return {"tool": "hybrid_candidate_recommendation", **engine.recommend(candidat_id)}


def tool_global_graph_summary(args: dict[str, Any]) -> dict[str, Any]:
    """Return macro indicators from Neo4j plus pgvector/doc storage counts."""

    top_k = int(args.get("top_k") or 8)
    status = tool_service_status({}).get("status", {})
    summary: dict[str, Any] = {"status": status}

    try:
        with get_neo4j_driver().session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            summary["graph_counts"] = [
                dict(row)
                for row in session.run(
                    """
                    MATCH (n)
                    RETURN labels(n)[0] AS label, count(n) AS n
                    ORDER BY n DESC
                    LIMIT 20
                    """
                )
            ]
            summary["top_secteurs"] = [
                dict(row)
                for row in session.run(
                    """
                    MATCH (o:OffreEmploi)
                    WITH coalesce(o.secteur_principal, o.secteur, 'non renseigne') AS secteur, count(o) AS n
                    RETURN secteur, n
                    ORDER BY n DESC
                    LIMIT $top_k
                    """,
                    top_k=top_k,
                )
            ]
            summary["top_metiers"] = [
                dict(row)
                for row in session.run(
                    """
                    MATCH (m:Métier)
                    RETURN coalesce(m.preferredLabel, m.label, m.name) AS metier,
                           coalesce(m.iscoCode, m.code_isco, '') AS code_isco
                    ORDER BY metier
                    LIMIT $top_k
                    """,
                    top_k=top_k,
                )
            ]
            summary["top_competences"] = [
                dict(row)
                for row in session.run(
                    """
                    MATCH (c:Compétence)
                    RETURN coalesce(c.preferredLabel, c.label, c.name) AS competence,
                           coalesce(c.conceptUri, '') AS source_uri,
                           coalesce(c.skillType, '') AS type_skill
                    ORDER BY competence
                    LIMIT $top_k
                    """,
                    top_k=top_k,
                )
            ]
    except Exception as exc:
        summary["neo4j_error"] = str(exc)

    try:
        with get_pg_conn().cursor() as cur:
            cur.execute(
                """
                SELECT entity_kind::text, count(*)
                FROM embeddings
                GROUP BY entity_kind
                ORDER BY entity_kind
                """
            )
            summary["pgvector_counts"] = [{"entity_kind": r[0], "n": r[1]} for r in cur.fetchall()]
            cur.execute(
                """
                SELECT source, count(*) AS n_chunks
                FROM doc_chunks
                GROUP BY source
                ORDER BY n_chunks DESC
                LIMIT %s
                """,
                (top_k,),
            )
            summary["document_sources"] = [{"source": r[0], "n_chunks": r[1]} for r in cur.fetchall()]
    except Exception as exc:
        try:
            get_pg_conn().rollback()
        except Exception:
            pass
        summary["pgvector_error"] = str(exc)

    return {"tool": "global_graph_summary", "summary": summary}


TOOL_REGISTRY = {
    "service_status": tool_service_status,
    "pgvector_semantic_search": tool_pgvector_semantic_search,
    "pgvector_document_search": tool_pgvector_document_search,
    "neo4j_graph_query": tool_neo4j_graph_query,
    "hybrid_candidate_recommendation": tool_hybrid_candidate_recommendation,
    "global_graph_summary": tool_global_graph_summary,
}
