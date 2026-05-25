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

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

ROOT = Path(__file__).resolve().parents[2]
SRC_05 = ROOT / "src" / "05_graphrag"
if str(SRC_05) not in sys.path:
    sys.path.insert(0, str(SRC_05))

from recommendation_engine import RecommendationEngine  # noqa: E402

DEFAULT_CANDIDAT_ID = "PPKOU2501080016340"


class AgentState(MessagesState, total=False):
    candidat_id: str
    top_k: int
    backend: str
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


def analyse_request(state: AgentState) -> AgentState:
    candidat_id = _extract_candidat_id(state)
    top_k = int(state.get("top_k") or os.getenv("AGENT_TOP_K", "5"))
    backend = str(state.get("backend") or os.getenv("AGENT_LLM_BACKEND", "simulation"))

    return {
        **state,
        "candidat_id": candidat_id,
        "top_k": top_k,
        "backend": backend,
        "traces": ["analyse_request"],
    }


def run_graphrag(state: AgentState) -> AgentState:
    engine = RecommendationEngine(
        llm_backend=state["backend"],
        top_k=int(state["top_k"]),
    )
    result = engine.recommend(str(state["candidat_id"]))
    traces = [*state.get("traces", []), "recommendation_engine"]
    return {**state, "result": result, "traces": traces}


def generate_final_answer(state: AgentState) -> AgentState:
    result = state.get("result", {})
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
