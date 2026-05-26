"""
Graphe LangGraph minimal pour orchestrer le moteur GraphRAG non-agentique.

Cette couche fournit une interface agentique stable pour LangGraph Studio,
Streamlit et la CLI `run_agent.py`. Elle encapsule le moteur existant du module
05 au lieu de dupliquer la logique de recherche pgvector/Neo4j.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

ROOT = Path(__file__).resolve().parents[2]
SRC_05 = ROOT / "src" / "05_graphrag"
SRC_04 = ROOT / "src" / "04_pgvector"
SRC_03 = ROOT / "src" / "03_knowledge_graph"
if str(SRC_05) not in sys.path:
    sys.path.insert(0, str(SRC_05))
if str(SRC_04) not in sys.path:
    sys.path.insert(0, str(SRC_04))
if str(SRC_03) not in sys.path:
    sys.path.insert(0, str(SRC_03))

load_dotenv(ROOT / ".env")

from recommendation_engine import RecommendationEngine  # noqa: E402
from ann_search import ann_from_text  # noqa: E402

DEFAULT_CANDIDAT_ID = "PPKOU2501080016340"
_PG_CONN = None
_NEO4J_DRIVER = None
_ST_MODEL = None


class AgentState(MessagesState, total=False):
    candidat_id: str
    top_k: int
    backend: str
    use_case: str
    user_query: str
    result: dict[str, Any]
    traces: list[str]


def _extract_candidat_id(state: AgentState) -> str:
    if state.get("candidat_id"):
        return str(state["candidat_id"])

    messages = state.get("messages", [])
    if messages:
        content = str(getattr(messages[-1], "content", messages[-1]))
        match = re.search(r"\b[A-Z]{2,}[A-Z0-9]{8,}\b", content)
        if match:
            return match.group(0)

    return DEFAULT_CANDIDAT_ID


def _message_text(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return ""
    return str(getattr(messages[-1], "content", messages[-1]))


def _find_candidat_id(text: str) -> str | None:
    match = re.search(r"\b[A-Z]{2,}[A-Z0-9]{8,}\b", text)
    return match.group(0) if match else None


def _infer_use_case(text: str, candidat_id: str | None) -> str:
    q = text.lower()
    if any(w in q for w in ["status", "health", "etat", "diagnostic", "connect"]):
        return "diagnostic"
    if any(w in q for w in ["ncf", "mepc", "diplome", "diplome", "referentiel", "classification"]):
        return "referentiel"
    if candidat_id:
        if any(w in q for w in ["competence", "competences", "skill", "gap", "manquant", "roadmap", "formation"]):
            return "skill_gap_roadmap"
        return "recommendation_candidat"
    if any(w in q for w in ["offre", "poste", "emploi", "recrute", "recrutement"]):
        return "recherche_offres"
    if any(w in q for w in ["devenir", "orientation", "metier", "carriere", "competence", "competences"]):
        return "orientation_metier"
    return "recherche_generale"


def _pg_dsn_from_env() -> str:
    if os.getenv("PG_DSN"):
        return str(os.getenv("PG_DSN"))
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db = os.getenv("PG_DB", "test_kmer")
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _get_pg_conn():
    global _PG_CONN
    if _PG_CONN is not None:
        try:
            if not _PG_CONN.closed:
                _PG_CONN.rollback()
                return _PG_CONN
        except Exception:
            _PG_CONN = None
    import psycopg

    _PG_CONN = psycopg.connect(_pg_dsn_from_env())
    _PG_CONN.autocommit = False
    return _PG_CONN


def _get_neo4j_driver():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is not None:
        return _NEO4J_DRIVER
    from neo4j import GraphDatabase

    _NEO4J_DRIVER = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
    )
    _NEO4J_DRIVER.verify_connectivity()
    return _NEO4J_DRIVER


def _get_st_model():
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


def _service_status() -> dict[str, str]:
    status = {}
    try:
        with _get_pg_conn().cursor() as cur:
            cur.execute("SELECT entity_kind::text, count(*) FROM embeddings GROUP BY entity_kind ORDER BY entity_kind")
            rows = cur.fetchall()
        status["pgvector"] = "connected: " + ", ".join(f"{k}={n}" for k, n in rows)
    except Exception as exc:
        status["pgvector"] = f"unavailable: {exc}"
    try:
        with _get_neo4j_driver().session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            n_nodes = session.run("MATCH (n) RETURN count(n) AS n").single()["n"]
        status["neo4j"] = f"connected: nodes={n_nodes}"
    except Exception as exc:
        status["neo4j"] = f"unavailable: {exc}"
    try:
        model = _get_st_model()
        get_dim = getattr(model, "get_embedding_dimension", model.get_sentence_embedding_dimension)
        status["st_model"] = f"loaded: {get_dim()}d"
    except Exception as exc:
        status["st_model"] = f"unavailable: {exc}"
    status["llm_backend"] = os.getenv("AGENT_LLM_BACKEND", os.getenv("LLM_BACKEND", "simulation"))
    return status


def _build_engine(backend: str, top_k: int) -> RecommendationEngine:
    pg = None
    neo4j = None
    st_model = None
    try:
        pg = _get_pg_conn()
    except Exception:
        pg = None
    try:
        neo4j = _get_neo4j_driver()
    except Exception:
        neo4j = None
    try:
        st_model = _get_st_model()
    except Exception:
        st_model = None
    return RecommendationEngine(
        neo4j_driver=neo4j,
        pg_conn=pg,
        st_model=st_model,
        llm_backend=backend,
        top_k=top_k,
    )


def _search_entities(query: str, kinds: list[str], top_k: int) -> dict[str, list[dict]]:
    conn = _get_pg_conn()
    model = _get_st_model()
    results = {}
    for kind in kinds:
        results[kind] = ann_from_text(conn, model, query, kind, top_k)
    return results


def _search_docs(query: str, top_k: int) -> list[dict]:
    conn = _get_pg_conn()
    model = _get_st_model()
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    emb = "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
    sql = """
        SELECT source, document_title, page_number, section_title, chunk_text,
               1 - (embedding <=> %s::vector) AS sim
        FROM doc_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (emb, emb, top_k))
        rows = cur.fetchall()
    return [
        {
            "source": r[0],
            "document_title": r[1],
            "page_number": r[2],
            "section_title": r[3],
            "chunk_text": str(r[4])[:700],
            "cosine_sim": round(float(r[5]), 4),
        }
        for r in rows
    ]


def analyse_request(state: AgentState) -> AgentState:
    text = _message_text(state)
    explicit_candidat_id = _find_candidat_id(text)
    candidat_id = state.get("candidat_id") or explicit_candidat_id
    top_k = int(state.get("top_k") or os.getenv("AGENT_TOP_K", "5"))
    backend = str(state.get("backend") or os.getenv("AGENT_LLM_BACKEND", "simulation"))
    use_case = _infer_use_case(text, str(candidat_id) if candidat_id else None)

    return {
        **state,
        "candidat_id": candidat_id or "",
        "top_k": top_k,
        "backend": backend,
        "use_case": use_case,
        "user_query": text,
        "traces": [f"analyse_request:{use_case}"],
    }


def run_graphrag(state: AgentState) -> AgentState:
    use_case = state.get("use_case", "recherche_generale")
    query = state.get("user_query", "")
    top_k = int(state["top_k"])
    backend = str(state["backend"])

    if use_case == "diagnostic":
        result = {"status": _service_status()}
    elif use_case in {"recommendation_candidat", "skill_gap_roadmap"}:
        candidat_id = state.get("candidat_id") or _extract_candidat_id(state)
        engine = _build_engine(backend=backend, top_k=top_k)
        result = engine.recommend(str(candidat_id))
    elif use_case == "referentiel":
        result = {"documents": _search_docs(query, top_k)}
    elif use_case == "orientation_metier":
        result = {
            "semantic_results": _search_entities(query, ["METIER", "COMPETENCE", "OFFRE_EMPLOI"], top_k),
            "documents": _search_docs(query, min(top_k, 5)),
        }
    else:
        result = {"semantic_results": _search_entities(query, ["OFFRE_EMPLOI", "METIER", "COMPETENCE"], top_k)}

    traces = [*state.get("traces", []), "recommendation_engine"]
    return {**state, "result": result, "traces": traces}


def generate_final_answer(state: AgentState) -> AgentState:
    result = state.get("result", {})
    use_case = state.get("use_case", "recherche_generale")

    if use_case == "diagnostic":
        lines = ["Diagnostic de l'instance connectee:", ""]
        for key, value in result.get("status", {}).items():
            lines.append(f"- {key}: {value}")
        answer = "\n".join(lines)
        traces = [*state.get("traces", []), "generate_final_answer"]
        return {
            **state,
            "traces": traces,
            "messages": [*state.get("messages", []), AIMessage(content=answer)],
        }

    if use_case == "referentiel":
        docs = result.get("documents", [])
        lines = ["Elements trouves dans les referentiels indexes:", ""]
        for i, doc in enumerate(docs, 1):
            lines.append(
                f"{i}. {doc.get('source')} p.{doc.get('page_number')} "
                f"- sim={doc.get('cosine_sim', 0):.3f}"
            )
            snippet = str(doc.get("chunk_text", "")).replace("\n", " ")
            lines.append(f"   {snippet[:260]}")
        answer = "\n".join(lines)
        traces = [*state.get("traces", []), "generate_final_answer"]
        return {
            **state,
            "traces": traces,
            "messages": [*state.get("messages", []), AIMessage(content=answer)],
        }

    if use_case in {"orientation_metier", "recherche_offres", "recherche_generale"}:
        semantic_results = result.get("semantic_results", {})
        lines = ["Resultats semantiques pgvector pour la requete:", ""]
        for kind, rows in semantic_results.items():
            lines.append(f"{kind}:")
            for i, row in enumerate(rows[: int(state.get("top_k", 5))], 1):
                label = row.get("label", "")
                entity_id = row.get("entity_id", "")
                sim = row.get("cosine_sim", 0)
                lines.append(f"  {i}. {label} [{entity_id}] - sim={sim:.3f}")
            lines.append("")
        docs = result.get("documents") or []
        if docs:
            lines.append("Appuis referentiels:")
            for doc in docs[:3]:
                lines.append(
                    f"- {doc.get('source')} p.{doc.get('page_number')}: "
                    f"{str(doc.get('chunk_text', '')).replace(chr(10), ' ')[:180]}"
                )
        answer = "\n".join(lines).strip()
        traces = [*state.get("traces", []), "generate_final_answer"]
        return {
            **state,
            "traces": traces,
            "messages": [*state.get("messages", []), AIMessage(content=answer)],
        }

    top_offres = result.get("top_offres", [])
    candidat = result.get("candidat", {})

    lines = [
        f"Analyse du candidat {state.get('candidat_id')} - {candidat.get('metier_vise', '')}",
        "",
        "Top offres recommandees:",
    ]
    for i, offre in enumerate(top_offres[: int(state.get("top_k", 5))], 1):
        lines.append(
            f"{i}. {offre.get('titre', offre.get('titre_poste', 'Offre'))} "
            f"- score={offre.get('score_hybride', 0):.3f} "
            f"- verdict={offre.get('verdict_recrutement', 'non precise')}"
        )

    skill_gap = result.get("skill_gap") or {}
    if skill_gap:
        lines.extend([
            "",
            f"Skill gap: {skill_gap.get('niveau_gap', 'non precise')} "
            f"(taux={skill_gap.get('taux_matching', 0)})",
        ])

    roadmap = result.get("roadmap") or {}
    if use_case == "skill_gap_roadmap" and roadmap:
        lines.extend(["", "Roadmap:"])
        for step in roadmap.get("etapes", [])[:5]:
            lines.append(
                f"- P{step.get('priorite', '?')}: {step.get('competence_cible', '')} "
                f"({step.get('delai_acquisition', 'delai non precise')})"
            )

    traces = [*state.get("traces", []), "generate_final_answer"]
    return {
        **state,
        "traces": traces,
        "messages": [*state.get("messages", []), AIMessage(content="\n".join(lines))],
    }


workflow = StateGraph(AgentState)
workflow.add_node("analyse_request", analyse_request)
workflow.add_node("run_graphrag", run_graphrag)
workflow.add_node("generate_final_answer", generate_final_answer)
workflow.add_edge(START, "analyse_request")
workflow.add_edge("analyse_request", "run_graphrag")
workflow.add_edge("run_graphrag", "generate_final_answer")
workflow.add_edge("generate_final_answer", END)

graph = workflow.compile()
