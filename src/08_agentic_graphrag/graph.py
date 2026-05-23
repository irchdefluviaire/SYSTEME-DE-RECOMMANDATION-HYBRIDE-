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
            "Apres analyse de votre profil dans la base vectorielle et le graphe de connaissances, "
            "aucune offre suffisamment proche n'a pu etre identifiee. "
            "Je vous recommande d'elargir vos criteres de recherche ou de verifier que votre profil "
            "est bien renseigne dans le systeme."
        )

    top = ranked[0]
    issues = critique.get("issues", [])
    fit = top.get("fit_profile", {})
    dimensions = top.get("score_components", {}) or fit.get("dimensions", {})
    verdict = top.get("verdict_recrutement") or fit.get("verdict", "non calcule")
    priorities = (top.get("priorites_developpement", []) or fit.get("priorites_developpement", []))[:5]
    missing = top.get("manquantes", [])[:3]
    score_sem = dimensions.get("semantique") or top.get("score_sem", "n/a")
    score_comp = dimensions.get("competences") or top.get("taux_match", "n/a")
    score_ncf = dimensions.get("niveau_ncf", "n/a")
    score_sect = dimensions.get("secteur_metier", "n/a")

    verdict_labels = {
        "pret_a_postuler": "pret a postuler immediatement",
        "postuler_avec_plan": "eligible avec un plan de montee en competences",
        "vivier_a_developper": "a integrer dans le vivier candidats pour une candidature future",
        "hors_cible_actuel": "hors cible pour ce poste dans l'immediat",
    }
    verdict_text = verdict_labels.get(str(verdict), str(verdict))

    paragraphs = [
        f"Suite a votre demande, le systeme a interroge la base vectorielle pgvector "
        f"et le graphe de connaissances Neo4j pour identifier les offres les plus compatibles "
        f"avec votre profil de {profile.get('metier_vise', 'professionnel')} "
        f"(niveau NCF {profile.get('ncf_niveau_final', 'non precise')}).",

        f"L'offre qui ressort en tete est \"{top.get('titre', 'N/A')}\" "
        f"(reference {top.get('offre_id', 'N/A')})"
        + (f", localisee a {top.get('ville')}" if top.get("ville") else "")
        + (f" dans le secteur {top.get('secteur')}" if top.get("secteur") else "")
        + f". Le score hybride obtenu est de {top.get('score_hybride', 'n/a')}, "
        f"compose de la similarite semantique ({score_sem}), "
        f"de la couverture de competences ESCO ({score_comp}), "
        f"de la compatibilite de niveau NCF ({score_ncf}) "
        f"et de l'alignement secteur/metier ({score_sect}).",

        f"Le verdict du systeme pour ce profil face a cette offre est : {verdict_text}.",
    ]

    if priorities or missing:
        comp_list = ", ".join(priorities or missing)
        paragraphs.append(
            f"Pour renforcer votre candidature, les competences prioritaires a developper "
            f"identifiees dans le graphe sont : {comp_list}."
        )

    if roadmap and roadmap.get("poste_cible"):
        etapes = roadmap.get("etapes_court_terme", [])[:2]
        etapes_text = " puis ".join(str(e) for e in etapes) if etapes else "a definir selon votre disponibilite"
        paragraphs.append(
            f"La trajectoire recommandee vise le poste de {roadmap.get('poste_cible')} "
            f"avec un score projete de {roadmap.get('score_matching_projete', 'n/a')} apres montee en competences. "
            f"Prochaine etape concrete : {etapes_text}."
        )

    if issues:
        reserves = " ; ".join(issues)
        paragraphs.append(
            f"L'agent critique a toutefois signale les reserves suivantes : {reserves}. "
            f"Tenez-en compte avant de finaliser votre candidature."
        )

    return "\n\n".join(paragraphs)


def _deterministic_career_answer(state: AgentState) -> str:
    guidance = state.get("career_guidance", {})
    local = guidance.get("local_evidence", {})
    skills = guidance.get("priority_skills", [])
    esco = guidance.get("esco_references", [])
    n_offers = local.get("n_matching_offers", 0)
    top_skills = skills[:6]
    query = state.get("user_query", "votre question")

    paragraphs = [
        f"Vous souhaitez vous orienter vers le metier de "
        f"{guidance.get('target_role', 'professionnel')} dans le domaine "
        f"{guidance.get('target_domain', 'camerounais')}. "
        + (
            f"Le graphe de connaissances local recense {n_offers} offre(s) correspondant "
            f"a ce profil sur le marche camerounais."
            if n_offers
            else
            "Les donnees locales disponibles ne contiennent pas encore suffisamment d'offres "
            "correspondant precisement a ce profil ; la reponse ci-dessous est donc une "
            "orientation generale et non une preuve de demande du marche."
        ),
    ]

    if top_skills:
        skill_texts = []
        for item in top_skills:
            evidence = item.get("evidence_locale", 0)
            line = f"{item.get('competence')} ({item.get('priorite')}) : {item.get('pourquoi')}"
            if evidence:
                line += f" — {evidence} occurrence(s) dans les offres locales"
            skill_texts.append(line)
        paragraphs.append(
            "Les competences a mettre en avant en priorite, identifiees via le graphe ESCO "
            "et les offres locales, sont les suivantes : " + " ; ".join(skill_texts) + "."
        )

    if "banque/finance" in str(guidance.get("target_domain", "")):
        paragraphs.append(
            "Pour le secteur bancaire et financier camerounais, l'angle decisif est de montrer "
            "que vous savez relier un modele a un risque metier concret, pas seulement entrainer "
            "un algorithme. Des preuves convaincantes pour un recruteur : un scoring de credit "
            "interpretable, une segmentation client, une detection d'anomalies ou de fraude, "
            "ou un tableau de bord de performance agence."
        )

    if esco:
        paragraphs.append(
            "Les metiers ESCO les plus proches de votre cible dans le referentiel sont : "
            + ", ".join(esco[:5]) + "."
        )

    sources = guidance.get("sources", [])
    if sources:
        paragraphs.append("Sources interrogees : " + ", ".join(sources) + ".")

    return "\n\n".join(paragraphs)


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
