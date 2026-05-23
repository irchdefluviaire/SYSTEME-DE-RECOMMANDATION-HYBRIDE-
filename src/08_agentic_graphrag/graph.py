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

from agent_state import AgentInput, AgentState
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
    build_career_competency_guidance,
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
        if isinstance(message, dict):
            role = str(message.get("type") or message.get("role") or "").lower()
            if role in {"human", "user"}:
                return str(message.get("content") or "")
        if getattr(message, "type", None) == "human":
            return str(getattr(message, "content", ""))
    for key in ("message", "message_humain", "question", "input", "user_query"):
        value = state.get(key)
        if value:
            return str(value)
    return ""


def analyse_request(state: AgentState) -> AgentState:
    """Supervisor node: normalize user goal and runtime options."""

    user_query = _latest_human_text(state)
    if not user_query.strip():
        user_query = (
            "Recommande les meilleures offres, explique les gaps de competences "
            "et propose une roadmap."
        )
    candidat_id = state.get("candidat_id") or _extract_candidate_id(user_query)
    intent = _infer_intent(user_query, candidat_id)
    if not candidat_id and intent == "recommendation":
        candidat_id = "AUTO"
    human_message = HumanMessage(
        content=user_query,
        additional_kwargs={
            "candidat_id": candidat_id,
            "top_k": int(state.get("top_k") or DEFAULT_TOP_K),
            "mode": "real",
            "intent": intent,
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
                "orientation_competences"
                if intent == "career_advice"
                else "charger_profil",
                "generation_finale_ollama"
                if intent == "career_advice"
                else "recherche_vectorielle_pgvector",
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
        "intent": intent,
        "top_k": int(state.get("top_k") or DEFAULT_TOP_K),
        "mode": "real",
        "replan_count": int(state.get("replan_count") or 0),
        "traces": _append_trace(state, trace),
    }


def route_after_analysis(state: AgentState) -> Literal["career_advice", "recommendation"]:
    return "career_advice" if state.get("intent") == "career_advice" else "recommendation"


def build_career_guidance(state: AgentState) -> AgentState:
    guidance, trace = build_career_competency_guidance(state.get("user_query", ""))
    return {**state, "career_guidance": guidance, "traces": _append_trace(state, trace)}


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
    verdict = top.get("verdict_recrutement") or top.get("fit_profile", {}).get("verdict")
    blockers = top.get("facteurs_bloquants", []) or top.get("fit_profile", {}).get("facteurs_bloquants", [])
    issues: list[str] = []
    if not ranked:
        issues.append("Aucune offre candidate n'a ete recuperee.")
    if score < MIN_SCORE_FINAL:
        issues.append(f"Score hybride trop faible ({score:.3f} < {MIN_SCORE_FINAL:.3f}).")
    if essential_gaps > MAX_ESSENTIAL_GAPS:
        issues.append(
            f"Trop de competences essentielles manquantes ({essential_gaps} > {MAX_ESSENTIAL_GAPS})."
        )
    if verdict == "hors_cible_actuel":
        issues.append("Verdict recruteur: hors cible actuel.")
    if verdict == "vivier_a_developper":
        issues.append("Verdict recruteur: profil a developper avant candidature forte.")
    for blocker in blockers:
        if blocker not in issues:
            issues.append(str(blocker))

    should_replan = bool(issues) and int(state.get("replan_count") or 0) < MAX_REPLAN
    critique = {
        "decision": "replan" if should_replan else ("accept_with_reserve" if issues else "accept"),
        "issues": issues,
        "top_score": score,
        "essential_gaps": essential_gaps,
        "verdict_recrutement": verdict,
        "facteurs_bloquants": blockers,
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
    if state.get("intent") == "career_advice":
        return _deterministic_career_answer(state)

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
    fit = top.get("fit_profile", {})
    dimensions = top.get("score_components", {}) or fit.get("dimensions", {})
    verdict = top.get("verdict_recrutement") or fit.get("verdict", "non calcule")
    priorities = top.get("priorites_developpement", []) or fit.get("priorites_developpement", [])
    return "\n".join(
        [
            f"Recommandation principale{reserve}: {top.get('titre')} ({top.get('offre_id')}).",
            f"Verdict recruteur: {verdict}.",
            f"Candidat: {profile.get('metier_vise')} | Niveau NCF: {profile.get('ncf_niveau_final')}.",
            (
                f"Score hybride: {top.get('score_hybride')} "
                f"(semantique={dimensions.get('semantique', top.get('score_sem'))}, "
                f"competences={dimensions.get('competences', top.get('taux_match'))}, "
                f"NCF={dimensions.get('niveau_ncf', 'n/a')}, "
                f"secteur/metier={dimensions.get('secteur_metier', 'n/a')})."
            ),
            "Competences manquantes prioritaires: "
            + (", ".join(priorities[:5] or missing) if (priorities or missing) else "aucune competence critique detectee dans le contexte disponible."),
            "Critique: " + ("; ".join(issues) if issues else "aucun blocage majeur detecte par l'agent critique."),
            (
                f"Roadmap: viser {roadmap.get('poste_cible', top.get('titre'))}, "
                f"score projete {roadmap.get('score_matching_projete', 'non calcule')}."
            ),
            "Cette sortie utilise les bases Neo4j et pgvector; active AGENT_USE_OLLAMA=1 pour la redaction par Llama 3.1.",
        ]
    )


def _deterministic_career_answer(state: AgentState) -> str:
    guidance = state.get("career_guidance", {})
    local = guidance.get("local_evidence", {})
    skills = guidance.get("priority_skills", [])
    esco = guidance.get("esco_references", [])
    n_offers = local.get("n_matching_offers", 0)
    top_skills = skills[:6]
    lines = [
        f"Objectif detecte: {guidance.get('target_role', 'Professionnel data')} dans le domaine {guidance.get('target_domain', 'camerounais')}.",
        (
            f"Evidence locale: {n_offers} offre(s) du fichier local correspondent au filtre data/banque-finance."
            if n_offers
            else "Evidence locale limitee: aucune offre locale assez proche n'a ete isolee; la reponse doit rester une orientation, pas une preuve de demande marche."
        ),
        "Competences a mettre en avant:",
    ]
    for item in top_skills:
        evidence = item.get("evidence_locale", 0)
        lines.append(
            f"- {item.get('competence')} ({item.get('priorite')}): {item.get('pourquoi')}."
            + (f" Occurrences locales: {evidence}." if evidence else "")
        )
    if "banque/finance" in str(guidance.get("target_domain", "")):
        lines.extend(
            [
                "Angle bancaire a defendre: montre que tu sais relier un modele a un risque metier, pas seulement entrainer un algorithme.",
                "Exemples de preuves dans ton portfolio: scoring de credit interpretable, segmentation client, detection d'anomalies/fraude, tableau de bord risque ou performance agence.",
            ]
        )
    if esco:
        lines.append("References ESCO proches: " + ", ".join(esco[:5]) + ".")
    lines.append("Sources utilisees: " + ", ".join(guidance.get("sources", [])) + ".")
    return "\n".join(lines)


def route_after_critique(state: AgentState) -> Literal["replan", "roadmap"]:
    return "replan" if state.get("should_replan") else "roadmap"


def _infer_intent(text: str, candidat_id: str) -> Literal["recommendation", "career_advice"]:
    lower = text.lower()
    career_markers = (
        "je veux devenir",
        "devenir",
        "quelles competences",
        "quelles compétences",
        "competences je dois",
        "compétences je dois",
        "mettre en avant",
        "orientation",
        "carriere",
        "carrière",
    )
    if not candidat_id and any(marker in lower for marker in career_markers):
        return "career_advice"
    return "recommendation"


def _extract_candidate_id(text: str) -> str:
    tokens = text.replace(",", " ").replace(";", " ").split()
    for token in tokens:
        cleaned = token.strip().strip(".:()[]{}")
        upper = cleaned.upper()
        if upper.startswith(("PP", "CAND")) and len(cleaned) >= 3:
            return cleaned
        if upper.startswith("C") and len(cleaned) >= 3 and any(ch.isdigit() for ch in cleaned):
            return cleaned
    return ""


workflow = StateGraph(AgentState, input_schema=AgentInput)
workflow.add_node("analyse_request", analyse_request)
workflow.add_node("build_career_guidance", build_career_guidance)
workflow.add_node("load_profile", load_profile)
workflow.add_node("retrieve_and_check_graph", retrieve_and_check_graph)
workflow.add_node("compute_skill_gap", compute_skill_gap)
workflow.add_node("score_and_rank", score_and_rank)
workflow.add_node("critique_recommendations", critique_recommendations)
workflow.add_node("replan_retrieval", replan_retrieval)
workflow.add_node("create_roadmap", create_roadmap)
workflow.add_node("generate_final_answer", generate_final_answer)

workflow.add_edge(START, "analyse_request")
workflow.add_conditional_edges(
    "analyse_request",
    route_after_analysis,
    {"career_advice": "build_career_guidance", "recommendation": "load_profile"},
)
workflow.add_edge("build_career_guidance", "generate_final_answer")
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
