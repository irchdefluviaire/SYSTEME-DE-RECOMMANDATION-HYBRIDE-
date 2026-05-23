"""Prompts for the Agentic GraphRAG final answer."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_FINAL = """Tu es un agent de recommandation emploi-competences.
Tu expliques des decisions fondees sur des donnees: scores, graphe, skill gap,
NCF/MEPC/ESCO et roadmap. Tu ne dois pas inventer de competences, de scores ou
de formations. Si une information manque, signale-le clairement."""


def build_final_prompt(state: dict[str, Any]) -> str:
    if state.get("intent") == "career_advice":
        return _build_career_prompt(state)

    offers = []
    for offer in state.get("ranked_offers", [])[:3]:
        offers.append(
            {
                "offre_id": offer.get("offre_id"),
                "titre": offer.get("titre"),
                "score_hybride": offer.get("score_hybride"),
                "score_sem": offer.get("score_sem"),
                "taux_match": offer.get("taux_match"),
                "verdict_recrutement": offer.get("verdict_recrutement"),
                "score_components": offer.get("score_components", {}),
                "facteurs_bloquants": offer.get("facteurs_bloquants", []),
                "priorites_developpement": offer.get("priorites_developpement", [])[:5],
                "secteur": offer.get("secteur"),
                "ville": offer.get("ville"),
                "manquantes": offer.get("manquantes", [])[:3],
                "essentielles_manquantes": offer.get("ess_manq", [])[:3],
                "ncf_candidat": offer.get("ncf_cand"),
                "ncf_requis": offer.get("ncf_code"),
            }
        )
    roadmap = state.get("roadmap", {})
    payload = {
        "question_utilisateur": state.get("user_query", ""),
        "candidat": {
            "candidat_id": state.get("candidate_profile", {}).get("candidat_id"),
            "metier_vise": state.get("candidate_profile", {}).get("metier_vise"),
            "secteur_metier": state.get("candidate_profile", {}).get("secteur_metier"),
            "ncf_niveau_final": state.get("candidate_profile", {}).get("ncf_niveau_final"),
            "diplome": state.get("candidate_profile", {}).get("diplome_raw"),
        },
        "offres_classees": offers,
        "critique": state.get("critique", {}),
        "roadmap": {
            "poste_cible": roadmap.get("poste_cible"),
            "score_actuel": roadmap.get("score_matching_actuel"),
            "score_projete": roadmap.get("score_matching_projete"),
            "competences_prioritaires": roadmap.get("competences_prioritaires", [])[:3],
            "etapes_court_terme": roadmap.get("etapes_court_terme", [])[:2],
            "etapes_moyen_terme": roadmap.get("etapes_moyen_terme", [])[:2],
        },
    }
    return f"""
Redige une reponse concise en francais pour le memoire/prototype.

Contraintes:
- expliquer comment la recommandation a ete obtenue;
- distinguer score semantique, competences, niveau NCF et alignement secteur/metier;
- donner un verdict recruteur: pret a postuler, postuler avec plan, vivier a developper ou hors cible;
- citer les limites quand elles existent;
- proposer les competences prioritaires a developper avant une candidature forte;
- proposer une prochaine action concrete au candidat;
- ne pas affirmer qu'une offre est excellente si la critique signale un risque.

Donnees disponibles:
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}
"""


def _build_career_prompt(state: dict[str, Any]) -> str:
    payload = {
        "question_utilisateur": state.get("user_query", ""),
        "orientation": state.get("career_guidance", {}),
    }
    return f"""
Redige une reponse concise en francais pour une question d'orientation metier.

Contraintes:
- ne pas pretendre qu'un candidat precis a ete charge;
- distinguer competences techniques, competences metier bancaire et preuves de portfolio;
- signaler clairement si l'evidence locale est limitee;
- citer les sources locales disponibles dans les donnees;
- ne pas inventer de scores, d'offres ou de formations.

Donnees disponibles:
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}
"""
