"""
prompt_templates.py
===========================================================================
Module 05 — Templates de prompts pour le LLM 2 (GraphRAG)

Trois types de prompts :
  1. RECOMMANDATION  → classement + justification des top-k offres
  2. SKILL_GAP       → analyse détaillée de l'écart de compétences
  3. ROADMAP         → plan d'action personnalisé avec formations NCF

Format : ChatML compatible Mistral-7B-Instruct et GPT-4o
===========================================================================
"""

# ─────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────

SYSTEM_RECOMMANDATION = """Tu es un conseiller emploi expert du marché du travail camerounais, \
spécialisé dans le matching emploi-compétences via les référentiels ESCO, MEPC 2013 et NCF 2017.

Tu reçois un contexte structuré contenant :
- Le profil d'un demandeur d'emploi
- Les top offres recommandées avec leurs scores de matching

Ta mission : générer une analyse de recommandation professionnelle, \
précise, bienveillante et ancrée dans la réalité camerounaise.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT avec un objet JSON valide (aucun texte avant ou après)
2. Les champs texte sont en français courant (pas de jargon technique)
3. Les formations suggérées doivent exister dans le système camerounais (NCF)
4. Ne pas inventer de compétences ou de formations inexistantes
5. Le score de matching est entre 0.0 et 1.0"""

SYSTEM_SKILL_GAP = """Tu es un expert RH et un spécialiste de l'ingénierie des compétences \
pour le marché du travail camerounais.

Tu analyses l'écart de compétences (skill gap) entre un candidat et une offre d'emploi \
en t'appuyant sur les données du graphe de connaissances (Neo4j + pgvector).

RÈGLES :
1. Réponds UNIQUEMENT en JSON valide
2. Distingue les compétences essentielles (bloquantes) des optionnelles
3. Pour chaque compétence manquante, propose une ressource de formation concrète \
   disponible au Cameroun ou en ligne (Coursera, AUF, etc.)
4. Sois précis sur les délais d'acquisition réalistes"""

SYSTEM_ROADMAP = """Tu es un orienteur professionnel expert du marché de l'emploi camerounais \
et du système éducatif national (NCF 2017, établissements locaux).

Tu génères des plans d'action personnalisés (roadmaps) pour améliorer \
l'employabilité d'un candidat sur un poste ciblé.

RÈGLES :
1. Réponds UNIQUEMENT en JSON valide et structuré
2. Les formations doivent être accessibles depuis le Cameroun (présentiel ou en ligne)
3. Les délais doivent être réalistes (ni trop optimistes, ni décourageants)
4. Priorise les compétences à fort impact sur le score de matching
5. Inclure des ressources gratuites ou peu coûteuses en priorité"""


# ─────────────────────────────────────────────────────────────────────────
# USER PROMPTS (templates à remplir avec .format(**variables))
# ─────────────────────────────────────────────────────────────────────────

USER_RECOMMANDATION = """Voici le contexte du système de recommandation :

{context_text}

Génère une analyse de recommandation structurée au format JSON strict :
{{
  "analyse_globale": "string — résumé en 2-3 phrases de la situation du candidat",
  "score_employabilite_global": float,
  "recommandations": [
    {{
      "rang": int,
      "offre_id": "string",
      "titre_poste": "string",
      "score_hybride": float,
      "pourquoi_recommandee": "string — justification en 1-2 phrases",
      "points_forts": ["string"],
      "points_attention": ["string"],
      "verdict": "Recommandation forte | Recommandation modérée | À développer"
    }}
  ],
  "conseil_global": "string — conseil personnalisé en 2-3 phrases pour le candidat",
  "prochaine_action": "string — une action concrète à faire cette semaine"
}}"""

USER_SKILL_GAP = """Analyse l'écart de compétences pour cette paire (candidat, offre) :

CANDIDAT : {candidat_id}
Métier visé : {metier_vise}
Niveau NCF  : {ncf_niveau}

OFFRE : {titre_offre}
Secteur : {secteur}

COMPÉTENCES ACQUISES ({n_acquises}) :
{acquises_list}

COMPÉTENCES MANQUANTES ({n_manquantes}) :
{manquantes_list}

Taux de matching actuel : {taux_match:.0%}

Génère l'analyse du skill gap au format JSON :
{{
  "taux_matching": float,
  "niveau_gap": "faible | modéré | important | critique",
  "competences_critiques": [
    {{
      "label": "string",
      "importance": "essentielle | optionnelle",
      "impact_score": float,
      "formation_recommandee": "string",
      "delai_acquisition": "string (ex: 1-3 mois)",
      "ressource": "string (lien ou établissement)"
    }}
  ],
  "competences_acquises_valeur": "string — valorisation des acquis en 1 phrase",
  "score_projete_apres_formation": float,
  "eligible_maintenant": boolean,
  "message_candidat": "string — message encourageant et réaliste"
}}"""

USER_ROADMAP = """Génère une roadmap personnalisée d'amélioration de profil :

CANDIDAT :
{candidat_profile}

POSTE CIBLE :
{offre_profile}

COMPÉTENCES MANQUANTES PRIORITAIRES :
{competences_manquantes}

Score actuel : {score_actuel:.0%}
Score projeté si formation complétée : {score_projete:.0%}

Génère la roadmap au format JSON :
{{
  "poste_cible": "string",
  "score_matching_actuel": float,
  "score_matching_projete": float,
  "duree_totale_estimee": "string",
  "etapes": [
    {{
      "priorite": int,
      "competence_cible": "string",
      "type": "technique | transversale | connaissance",
      "importance": "essentielle | optionnelle",
      "formation": {{
        "nom": "string",
        "etablissement": "string",
        "duree": "string",
        "cout_estimatif": "string",
        "modalite": "presentiel | en ligne | hybride",
        "lien_info": "string"
      }},
      "impact_score": float,
      "delai_acquisition": "string"
    }}
  ],
  "ressources_gratuites": ["string"],
  "certifications_utiles": ["string"],
  "conseil_candidature_immediate": "string",
  "message_motivation": "string — 2 phrases encourageantes"
}}"""


# ─────────────────────────────────────────────────────────────────────────
# FORMAT CHATML (Mistral-7B-Instruct)
# ─────────────────────────────────────────────────────────────────────────

def format_chatml(system: str, user: str) -> str:
    """Format ChatML pour Mistral-7B-Instruct."""
    return (
        f"<s>[INST] <<SYS>>\n{system.strip()}\n<</SYS>>\n\n"
        f"{user.strip()} [/INST]"
    )


def format_openai_messages(system: str, user: str) -> list[dict]:
    """Format messages pour l'API OpenAI / GPT-4o."""
    return [
        {"role": "system",  "content": system.strip()},
        {"role": "user",    "content": user.strip()},
    ]


# ─────────────────────────────────────────────────────────────────────────
# RESSOURCES DE FORMATION (base camerounaise)
# ─────────────────────────────────────────────────────────────────────────

FORMATIONS_CAMEROUN = {
    "informatique": [
        "Institut Universitaire de Technologie (IUT) de Douala — BTS Informatique",
        "École Nationale Supérieure Polytechnique (ENSP) — Yaoundé",
        "MOOC AUF (Agence Universitaire de la Francophonie) — en ligne",
        "Coursera by Google / IBM — Certifications Data & IT",
    ],
    "finance": [
        "Université de Yaoundé II — École Supérieure des Sciences Économiques",
        "ESSEC Douala — Formation continue Finance",
        "COBAC (Cameroun) — Certifications bancaires",
        "Coursera — Finance for Non-Finance Managers",
    ],
    "marketing": [
        "Institut Supérieur de Commerce (ISC) — Yaoundé",
        "ESSEC Douala — Marketing et Commerce",
        "LinkedIn Learning — Digital Marketing Certificate",
        "Google Digital Skills for Africa — en ligne gratuit",
    ],
    "logistique": [
        "École des Douanes du Cameroun — Formation Transit",
        "IUT Douala — DUT Logistique et Transport",
        "CILT (Chartered Institute of Logistics) — International",
        "Coursera — Supply Chain Management (Rutgers)",
    ],
    "sante": [
        "FMSB Université de Yaoundé I — Formation paramédicale",
        "Centre Pasteur du Cameroun — Formations spécialisées",
        "WHO Academy — en ligne gratuit (anglais/français)",
    ],
    "general": [
        "MOOC AUF (Agence Universitaire de la Francophonie)",
        "Coursera — Certifications reconnues internationalement",
        "LinkedIn Learning — Formation professionnelle continue",
        "YouTube EDU — Ressources pédagogiques gratuites",
        "ENAM (École Nationale d'Administration et de Magistrature)",
    ],
}


def get_formations(secteur: str, competence: str = "") -> list[str]:
    """Retourne les formations recommandées selon le secteur."""
    secteur_lower = str(secteur).lower()
    for key, formations in FORMATIONS_CAMEROUN.items():
        if key in secteur_lower or key in competence.lower():
            return formations[:3]
    return FORMATIONS_CAMEROUN["general"][:3]
