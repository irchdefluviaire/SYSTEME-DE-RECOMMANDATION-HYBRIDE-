"""LangGraph entrypoint for Agentic GraphRAG.

LangGraph Studio loads the compiled `graph` object from this file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from agent_state import AgentState
from llm import ollama_generate
from prompts import SYSTEM_FINAL, build_final_prompt
from settings import (
    DEFAULT_TOP_K,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING,
    MAX_ESSENTIAL_GAPS,
    MAX_REPLAN,
    MIN_SCORE_FINAL,
    OLLAMA_MODEL,
    USE_OLLAMA,
)
from tools import (
    build_retrieval_context,
    build_training_roadmap,
    extract_skill_gaps,
    load_candidate_profile,
    rank_hybrid_offers,
)


def _append_trace(state: AgentState, trace: dict) -> list[dict]:
    return [*state.get("traces", []), trace]


def _latest_human_text(state: AgentState) -> str:
    messages = state.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
        if getattr(message, "type", None) == "human":
            return str(getattr(message, "content", ""))
    return str(state.get("user_query", "") or "")


def analyse_request(state: AgentState) -> AgentState:
    """Supervisor node: normalize user goal and runtime options."""

    user_query = _latest_human_text(state)
    candidat_id = state.get("candidat_id") or _extract_candidate_id(user_query)
    if not candidat_id:
        candidat_id = "AUTO"
    human_message = HumanMessage(
        content=user_query,
        additional_kwargs={
            "candidat_id": candidat_id,
            "top_k": int(state.get("top_k") or DEFAULT_TOP_K),
            "mode": "real",
        },
    )
    messages_update = [] if state.get("messages") else [human_message]
    trace = {
        "step": "supervisor",
        "status": "ok",
        "message": "HumanMessage transforme en plan Agentic GraphRAG.",
        "details": {
            "human_message": human_message.content,
            "langsmith": {
                "tracing": LANGSMITH_TRACING,
                "project": LANGSMITH_PROJECT if LANGSMITH_TRACING else None,
            },
            "plan": [
                "charger_profil",
                "recherche_vectorielle_pgvector",
                "verification_graphe_neo4j",
                "skill_gap",
                "scoring_hybride",
                "critique",
                "roadmap",
                "generation_finale_ollama",
            ]
        },
    }
    return {
        **state,
        "user_query": user_query,
        "messages": messages_update,
        "human_message": {
            "role": "human",
            "content": human_message.content,
            "metadata": human_message.additional_kwargs,
        },
        "candidat_id": candidat_id,
        "top_k": int(state.get("top_k") or DEFAULT_TOP_K),
        "mode": "real",
        "replan_count": int(state.get("replan_count") or 0),
        "traces": _append_trace(state, trace),
    }


def load_profile(state: AgentState) -> AgentState:
    profile, trace = load_candidate_profile(state["candidat_id"])
    return {**state, "candidate_profile": profile, "candidat_id": profile["candidat_id"], "traces": _append_trace(state, trace)}


def retrieve_and_check_graph(state: AgentState) -> AgentState:
    context, traces = build_retrieval_context(
        state["candidat_id"],
        state["candidate_profile"],
        top_k=int(state.get("top_k") or DEFAULT_TOP_K),
        use_real_dbs=True,
    )
    top_offers = context.get("top_offres", [])
    return {
        **state,
        "vector_results": top_offers,
        "graph_results": top_offers,
        "traces": [*state.get("traces", []), *traces],
    }


def compute_skill_gap(state: AgentState) -> AgentState:
    gaps, trace = extract_skill_gaps(state.get("graph_results", []))
    return {**state, "skill_gaps": gaps, "traces": _append_trace(state, trace)}


def score_and_rank(state: AgentState) -> AgentState:
    ranked, trace = rank_hybrid_offers(state.get("graph_results", []))
    return {**state, "ranked_offers": ranked, "traces": _append_trace(state, trace)}


def critique_recommendations(state: AgentState) -> AgentState:
    ranked = state.get("ranked_offers", [])
    top = ranked[0] if ranked else {}
    score = float(top.get("score_hybride", 0.0) or 0.0)
    essential_gaps = len(top.get("ess_manq", []) or [])
    issues: list[str] = []
    if not ranked:
        issues.append("Aucune offre candidate n'a ete recuperee.")
    if score < MIN_SCORE_FINAL:
        issues.append(f"Score hybride trop faible ({score:.3f} < {MIN_SCORE_FINAL:.3f}).")
    if essential_gaps > MAX_ESSENTIAL_GAPS:
        issues.append(
            f"Trop de competences essentielles manquantes ({essential_gaps} > {MAX_ESSENTIAL_GAPS})."
        )

    should_replan = bool(issues) and int(state.get("replan_count") or 0) < MAX_REPLAN
    critique = {
        "decision": "replan" if should_replan else ("accept_with_reserve" if issues else "accept"),
        "issues": issues,
        "top_score": score,
        "essential_gaps": essential_gaps,
    }
    trace = {
        "step": "critic",
        "status": "warning" if issues else "ok",
        "message": "Critique executee.",
        "details": critique,
    }
    return {
        **state,
        "critique": critique,
        "should_replan": should_replan,
        "traces": _append_trace(state, trace),
    }


def replan_retrieval(state: AgentState) -> AgentState:
    """Simple replan: enlarge top_k once, then retry retrieval and graph checks."""

    new_top_k = max(int(state.get("top_k") or DEFAULT_TOP_K) * 2, DEFAULT_TOP_K + 5)
    trace = {
        "step": "replan",
        "status": "warning",
        "message": "Resultats faibles: elargissement de la recherche.",
        "details": {"old_top_k": state.get("top_k"), "new_top_k": new_top_k},
    }
    return {
        **state,
        "top_k": new_top_k,
        "replan_count": int(state.get("replan_count") or 0) + 1,
        "should_replan": False,
        "traces": _append_trace(state, trace),
    }


def create_roadmap(state: AgentState) -> AgentState:
    ranked = state.get("ranked_offers", [])
    if not ranked:
        trace = {"step": "roadmap", "status": "warning", "message": "Aucune offre pour generer une roadmap.", "details": {}}
        return {**state, "roadmap": {}, "traces": _append_trace(state, trace)}
    roadmap, trace = build_training_roadmap(state.get("candidate_profile", {}), ranked[0])
    return {**state, "roadmap": roadmap, "traces": _append_trace(state, trace)}


def generate_final_answer(state: AgentState) -> AgentState:
    if USE_OLLAMA:
        prompt = build_final_prompt(state)
        answer = ollama_generate(prompt, system=SYSTEM_FINAL)
        mode = "ollama"
    else:
        answer = _deterministic_answer(state)
        mode = "deterministic"
    trace = {
        "step": "generate_final_answer",
        "status": "ok",
        "message": "Reponse finale generee.",
        "details": {"mode": mode, "model": OLLAMA_MODEL if USE_OLLAMA else None},
    }
    return {
        **state,
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
        "traces": _append_trace(state, trace),
    }


def _deterministic_answer(state: AgentState) -> str:
    ranked = state.get("ranked_offers", [])
    profile = state.get("candidate_profile", {})
    critique = state.get("critique", {})
    roadmap = state.get("roadmap", {})
    if not ranked:
        return (
            "Aucune recommandation robuste n'a ete produite. "
            "Le systeme doit elargir la recherche ou verifier les donnees du candidat."
        )
    top = ranked[0]
    issues = critique.get("issues", [])
    reserve = " avec reserve" if issues else ""
    missing = top.get("manquantes", [])[:3]
    return "\n".join(
        [
            f"Recommandation principale{reserve}: {top.get('titre')} ({top.get('offre_id')}).",
            f"Candidat: {profile.get('metier_vise')} | Niveau NCF: {profile.get('ncf_niveau_final')}.",
            (
                f"Score hybride: {top.get('score_hybride')} "
                f"(semantique={top.get('score_sem')}, graphe/skill-gap={top.get('taux_match')})."
            ),
            "Competences manquantes prioritaires: "
            + (", ".join(missing) if missing else "aucune competence critique detectee dans le contexte disponible."),
            "Critique: " + ("; ".join(issues) if issues else "aucun blocage majeur detecte par l'agent critique."),
            (
                f"Roadmap: viser {roadmap.get('poste_cible', top.get('titre'))}, "
                f"score projete {roadmap.get('score_matching_projete', 'non calcule')}."
            ),
            "Cette sortie utilise les bases Neo4j et pgvector; active AGENT_USE_OLLAMA=1 pour la redaction par Llama 3.1.",
        ]
    )


def route_after_critique(state: AgentState) -> Literal["replan", "roadmap"]:
    return "replan" if state.get("should_replan") else "roadmap"


def _extract_candidate_id(text: str) -> str:
    tokens = text.replace(",", " ").replace(";", " ").split()
    for token in tokens:
        if token.upper().startswith(("PP", "CAND", "C")) and len(token) >= 3:
            return token.strip()
    return ""


workflow = StateGraph(AgentState)
workflow.add_node("analyse_request", analyse_request)
workflow.add_node("load_profile", load_profile)
workflow.add_node("retrieve_and_check_graph", retrieve_and_check_graph)
workflow.add_node("compute_skill_gap", compute_skill_gap)
workflow.add_node("score_and_rank", score_and_rank)
workflow.add_node("critique_recommendations", critique_recommendations)
workflow.add_node("replan_retrieval", replan_retrieval)
workflow.add_node("create_roadmap", create_roadmap)
workflow.add_node("generate_final_answer", generate_final_answer)

workflow.add_edge(START, "analyse_request")
workflow.add_edge("analyse_request", "load_profile")
workflow.add_edge("load_profile", "retrieve_and_check_graph")
workflow.add_edge("retrieve_and_check_graph", "compute_skill_gap")
workflow.add_edge("compute_skill_gap", "score_and_rank")
workflow.add_edge("score_and_rank", "critique_recommendations")
workflow.add_conditional_edges(
    "critique_recommendations",
    route_after_critique,
    {"replan": "replan_retrieval", "roadmap": "create_roadmap"},
)
workflow.add_edge("replan_retrieval", "retrieve_and_check_graph")
workflow.add_edge("create_roadmap", "generate_final_answer")
workflow.add_edge("generate_final_answer", END)

graph = workflow.compile()
