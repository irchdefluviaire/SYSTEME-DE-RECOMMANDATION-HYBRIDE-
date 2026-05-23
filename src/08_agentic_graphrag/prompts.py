"""Prompts for the Agentic GraphRAG final answer."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_FINAL = """Tu es un conseiller emploi-competences expert du marche camerounais.
Tu reponds directement au candidat en langage naturel fluide en francais.
Tu te bases exclusivement sur les donnees fournies (scores pgvector, graphe Neo4j,
skill gap, niveaux NCF/MEPC/ESCO, roadmap). Tu n'inventes aucune competence, score
ou formation. Quand une information manque, tu le signales naturellement dans ta reponse.
Ta reponse est un texte continu, sans listes a puces ni JSON."""


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
L'utilisateur a pose la question suivante : "{payload['question_utilisateur']}"

Reponds-lui directement en langage naturel fluide (pas de listes, pas de JSON).
Ta reponse doit en un seul texte continu :
- accuser reception de sa question et presenter le candidat et l'offre principale retenue;
- expliquer en termes simples comment le systeme a calcule le score (similarite semantique
  via pgvector, couverture de competences et contraintes NCF/secteur via le graphe Neo4j);
- donner le verdict recruteur de maniere claire et justifiee;
- mentionner les competences prioritaires a developper si des ecarts existent;
- proposer la prochaine etape concrete a partir de la roadmap;
- si la critique a detecte des reserves, les mentionner honnêtement.

Donnees disponibles:
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}
"""


def _build_career_prompt(state: dict[str, Any]) -> str:
    payload = {
        "question_utilisateur": state.get("user_query", ""),
        "orientation": state.get("career_guidance", {}),
    }
    return f"""
L'utilisateur a pose la question suivante : "{payload['question_utilisateur']}"

Reponds-lui directement en langage naturel fluide (pas de listes, pas de JSON).
Ta reponse doit en un seul texte continu :
- repondre directement a sa question d'orientation;
- distinguer competences techniques et competences metier specifiques au secteur vise;
- signaler si l'evidence locale (offres camerounaises) est limitee;
- suggerer des preuves concretes a mettre en avant dans un portfolio ou entretien;
- ne pas inventer de scores, d'offres ou de formations inexistantes.

Donnees disponibles:
{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}
"""
