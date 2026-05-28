"""
LangGraph orchestration for the Agentic GraphRAG workflow.

The workflow follows the image supplied by the user:
query -> agent/planner -> tools -> context -> agent/synthesis -> final answer.
Neo4j and PostgreSQL/pgvector are exposed as tools in tools.py.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_05 = ROOT / "src" / "05_graphrag"
if str(SRC_05) not in sys.path:
    sys.path.insert(0, str(SRC_05))
load_dotenv(ROOT / ".env")

from answer_critic import critique_answer  # noqa: E402
from tools import TOOL_REGISTRY  # noqa: E402

DEFAULT_CANDIDAT_ID = "PPKOU2501080016340"


class AgentState(MessagesState, total=False):
    candidat_id: str
    top_k: int
    backend: str
    use_case: str
    user_query: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    context: dict[str, Any]
    result: dict[str, Any]
    critic: dict[str, Any]
    traces: list[str]


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
    if any(w in q for w in ["ncf", "mepc", "diplome", "referentiel", "classification"]):
        return "referentiel"
    if any(w in q for w in ["cypher", "relation", "relations", "graphe", "neo4j", "relie", "lier"]):
        return "graph_query"
    if candidat_id:
        if any(w in q for w in ["competence", "competences", "skill", "gap", "manquant", "roadmap", "formation"]):
            return "skill_gap_roadmap"
        return "recommendation_candidat"
    if any(w in q for w in ["devenir", "orientation", "metier", "carriere", "competence", "competences"]):
        return "orientation_metier"
    return "recherche_generale"


def _tool_call(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args}


def analyse_request(state: AgentState) -> AgentState:
    query = _message_text(state)
    explicit_candidat_id = _find_candidat_id(query)
    candidat_id = state.get("candidat_id") or explicit_candidat_id or ""
    top_k = int(state.get("top_k") or os.getenv("AGENT_TOP_K", "5"))
    backend = "ollama"
    use_case = _infer_use_case(query, candidat_id or None)

    return {
        **state,
        "candidat_id": candidat_id,
        "top_k": top_k,
        "backend": backend,
        "use_case": use_case,
        "user_query": query,
        "traces": [f"analyse_request:{use_case}"],
    }


def plan_tools(state: AgentState) -> AgentState:
    """Select the database tools needed for the request."""

    query = state.get("user_query", "")
    top_k = int(state.get("top_k", 5))
    use_case = state.get("use_case", "recherche_generale")
    candidat_id = state.get("candidat_id") or DEFAULT_CANDIDAT_ID

    if use_case == "diagnostic":
        tool_calls = [_tool_call("service_status")]
    elif use_case in {"recommendation_candidat", "skill_gap_roadmap"}:
        tool_calls = [
            _tool_call(
                "hybrid_candidate_recommendation",
                candidat_id=candidat_id,
                top_k=top_k,
                backend="ollama",
            )
        ]
    elif use_case == "referentiel":
        tool_calls = [_tool_call("pgvector_document_search", query=query, top_k=top_k)]
    elif use_case == "graph_query":
        tool_calls = [_tool_call("neo4j_graph_query", query=query, top_k=top_k)]
    elif use_case == "orientation_metier":
        tool_calls = [
            _tool_call(
                "pgvector_semantic_search",
                query=query,
                kinds=["METIER", "COMPETENCE", "OFFRE_EMPLOI"],
                top_k=top_k,
            ),
            _tool_call("pgvector_document_search", query=query, top_k=min(top_k, 5)),
            _tool_call("neo4j_graph_query", query=query, top_k=top_k),
        ]
    else:
        tool_calls = [
            _tool_call(
                "pgvector_semantic_search",
                query=query,
                kinds=["OFFRE_EMPLOI", "METIER", "COMPETENCE"],
                top_k=top_k,
            )
        ]

    traces = [*state.get("traces", []), "plan_tools:" + ",".join(t["name"] for t in tool_calls)]
    return {**state, "tool_calls": tool_calls, "traces": traces}


def execute_tools(state: AgentState) -> AgentState:
    """Invoke selected tools and keep failures visible in the state."""

    tool_results: list[dict[str, Any]] = []
    traces = [*state.get("traces", [])]
    for call in state.get("tool_calls", []):
        name = call["name"]
        args = call.get("args", {})
        tool = TOOL_REGISTRY[name]
        try:
            result = tool(args)
            tool_results.append(result)
            traces.append(f"tool:{name}:ok")
        except Exception as exc:
            tool_results.append({"tool": name, "error": str(exc)})
            traces.append(f"tool:{name}:error")

    return {**state, "tool_results": tool_results, "traces": traces}


def build_context(state: AgentState) -> AgentState:
    """Normalize raw tool outputs into one context object for synthesis."""

    context: dict[str, Any] = {
        "query": state.get("user_query", ""),
        "use_case": state.get("use_case", ""),
        "tools": state.get("tool_results", []),
    }
    result: dict[str, Any] = {}
    for item in state.get("tool_results", []):
        tool_name = item.get("tool")
        if item.get("error"):
            result.setdefault("errors", []).append(item)
            continue
        if tool_name == "service_status":
            result["status"] = item.get("status", {})
        elif tool_name == "pgvector_semantic_search":
            result["semantic_results"] = item.get("results", {})
        elif tool_name == "pgvector_document_search":
            result["documents"] = item.get("documents", [])
        elif tool_name == "neo4j_graph_query":
            result["graph"] = {"plan": item.get("plan", {}), "rows": item.get("rows", [])}
        elif tool_name == "hybrid_candidate_recommendation":
            result.update(item)

    traces = [*state.get("traces", []), "build_context"]
    return {**state, "context": context, "result": result, "traces": traces}


def generate_final_answer(state: AgentState) -> AgentState:
    result = state.get("result", {})
    use_case = state.get("use_case", "recherche_generale")

    if result.get("errors") and len(result) == 1:
        answer = _render_errors(result["errors"])
        critic = {"decision": "revise", "reason": "all_tools_failed"}
    elif use_case == "diagnostic":
        answer = _render_diagnostic(result)
        critic = {"decision": "accept"}
    elif use_case == "referentiel":
        answer = _render_documents(result.get("documents", []))
        critic = critique_answer(answer, result.get("documents", []))
    elif use_case == "graph_query":
        graph_result = result.get("graph", {})
        answer = _render_graph_rows(graph_result)
        critic = critique_answer(answer, graph_result)
    elif use_case in {"orientation_metier", "recherche_generale"}:
        answer = _render_semantic_and_graph(result, int(state.get("top_k", 5)))
        critic = critique_answer(answer, result)
    else:
        answer = _render_candidate_recommendation(state, result)
        critic = critique_answer(answer, result)

    traces = [*state.get("traces", []), "generate_final_answer", "answer_critic"]
    return {
        **state,
        "critic": critic,
        "traces": traces,
        "messages": [*state.get("messages", []), AIMessage(content=answer)],
    }


def _render_errors(errors: list[dict[str, Any]]) -> str:
    lines = ["Aucun outil n'a pu retourner de contexte exploitable.", ""]
    for err in errors:
        lines.append(f"- {err.get('tool')}: {err.get('error')}")
    return "\n".join(lines)


def _render_diagnostic(result: dict[str, Any]) -> str:
    lines = ["Diagnostic de l'instance connectee:", ""]
    for key, value in result.get("status", {}).items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _render_documents(docs: list[dict[str, Any]]) -> str:
    lines = ["Elements trouves dans les referentiels indexes:", ""]
    for i, doc in enumerate(docs, 1):
        lines.append(
            f"{i}. {doc.get('source')} p.{doc.get('page_number')} "
            f"- sim={doc.get('cosine_sim', 0):.3f}"
        )
        snippet = str(doc.get("parent_context") or doc.get("chunk_text", "")).replace("\n", " ")
        lines.append(f"   {snippet[:260]}")
    if not docs:
        lines.append("Aucun document pertinent trouve dans pgvector.")
    return "\n".join(lines)


def _render_graph_rows(graph_result: dict[str, Any]) -> str:
    rows = graph_result.get("rows", [])
    plan = graph_result.get("plan", {})
    lines = [f"Requete graphe Neo4j executee ({plan.get('intent', 'intent_non_precise')}):", ""]
    for i, row in enumerate(rows, 1):
        rendered = " | ".join(f"{k}={v}" for k, v in row.items())
        lines.append(f"{i}. {rendered}")
    if not rows:
        lines.append("Aucun resultat trouve dans Neo4j pour cette question.")
    return "\n".join(lines)


def _render_semantic_and_graph(result: dict[str, Any], top_k: int) -> str:
    lines = ["Contexte recupere par les tools:", ""]

    semantic_results = result.get("semantic_results", {})
    if semantic_results:
        lines.append("pgvector - recherche semantique hybride:")
        for kind, rows in semantic_results.items():
            lines.append(f"{kind}:")
            for i, row in enumerate(rows[:top_k], 1):
                score = row.get("dense_score", row.get("cosine_sim", row.get("rrf_score", 0)))
                lexical = row.get("lexical_score", 0)
                lines.append(
                    f"  {i}. {row.get('label', '')} [{row.get('entity_id', '')}] "
                    f"- dense={float(score or 0):.3f} lexical={float(lexical or 0):.3f}"
                )

    graph_result = result.get("graph")
    if graph_result:
        lines.extend(["", "Neo4j - relations graphe:"])
        rows = graph_result.get("rows", [])
        for i, row in enumerate(rows[:top_k], 1):
            rendered = " | ".join(f"{k}={v}" for k, v in row.items())
            lines.append(f"  {i}. {rendered}")
        if not rows:
            lines.append("  Aucun lien graphe retourne.")

    docs = result.get("documents") or []
    if docs:
        lines.extend(["", "pgvector - appuis referentiels:"])
        for doc in docs[:3]:
            lines.append(
                f"- {doc.get('source')} p.{doc.get('page_number')}: "
                f"{str(doc.get('chunk_text', '')).replace(chr(10), ' ')[:180]}"
            )

    if result.get("errors"):
        lines.extend(["", "Outils en erreur:"])
        for err in result["errors"]:
            lines.append(f"- {err.get('tool')}: {err.get('error')}")

    return "\n".join(lines).strip()


def _render_candidate_recommendation(state: AgentState, result: dict[str, Any]) -> str:
    candidat = result.get("candidat", {})
    top_offres = result.get("top_offres", [])
    lines = [
        f"Analyse du candidat {state.get('candidat_id')} - {candidat.get('metier_vise', '')}",
        "",
        "Top offres recommandees par le tool hybride Neo4j + pgvector:",
    ]
    for i, offre in enumerate(top_offres[: int(state.get("top_k", 5))], 1):
        lines.append(
            f"{i}. {offre.get('titre', offre.get('titre_poste', 'Offre'))} "
            f"- score={offre.get('score_hybride', 0):.3f} "
            f"- verdict={offre.get('verdict_recrutement', 'non precise')}"
        )

    skill_gap = result.get("skill_gap") or {}
    if skill_gap:
        lines.extend(
            [
                "",
                f"Skill gap: {skill_gap.get('niveau_gap', 'non precise')} "
                f"(taux={skill_gap.get('taux_matching', 0)})",
            ]
        )

    roadmap = result.get("roadmap") or {}
    if state.get("use_case") == "skill_gap_roadmap" and roadmap:
        lines.extend(["", "Roadmap:"])
        for step in roadmap.get("etapes", [])[:5]:
            lines.append(
                f"- P{step.get('priorite', '?')}: {step.get('competence_cible', '')} "
                f"({step.get('delai_acquisition', 'delai non precise')})"
            )

    if result.get("errors"):
        lines.extend(["", "Outils en erreur:"])
        for err in result["errors"]:
            lines.append(f"- {err.get('tool')}: {err.get('error')}")

    return "\n".join(lines)


workflow = StateGraph(AgentState)
workflow.add_node("analyse_request", analyse_request)
workflow.add_node("plan_tools", plan_tools)
workflow.add_node("execute_tools", execute_tools)
workflow.add_node("build_context", build_context)
workflow.add_node("generate_final_answer", generate_final_answer)
workflow.add_edge(START, "analyse_request")
workflow.add_edge("analyse_request", "plan_tools")
workflow.add_edge("plan_tools", "execute_tools")
workflow.add_edge("execute_tools", "build_context")
workflow.add_edge("build_context", "generate_final_answer")
workflow.add_edge("generate_final_answer", END)

graph = workflow.compile()
