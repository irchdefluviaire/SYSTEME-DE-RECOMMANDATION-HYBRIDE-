"""Agentic GraphRAG — ReAct agent (LangGraph prebuilt).

L'agent llama3.1 décide lui-même quels outils appeler et dans quel ordre.
Entrée : message texte libre. Sortie : réponse en langage naturel.
"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from settings import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE
from tools import (
    generate_roadmap,
    get_candidate_profile,
    get_skill_gap,
    search_offers,
    search_offers_for_candidate,
)

# ── Prompt système ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es un conseiller emploi-compétences expert du marché camerounais.
Tu réponds en français, en langage naturel fluide, en t'adressant directement à l'utilisateur.

Règles d'utilisation des outils :
1. Si l'utilisateur donne un ID candidat (ex: PP001, CAND_042, C001) :
   - Appelle d'abord get_candidate_profile pour charger son profil.
   - Puis search_offers_for_candidate pour obtenir les offres classées par score hybride.
   - Si l'utilisateur veut un plan de formation, appelle get_skill_gap puis generate_roadmap.

2. Si l'utilisateur pose une question générale (sans ID) :
   - Utilise search_offers avec une requête descriptive extraite de sa demande.

3. Ta réponse finale doit :
   - Être un texte continu en français, sans listes JSON ni code.
   - Expliquer comment le score a été calculé (sémantique, compétences, NCF, secteur).
   - Donner le verdict clair : prêt à postuler, postuler avec plan, vivier, hors cible.
   - Proposer une prochaine action concrète.
   - Ne jamais inventer de données absentes des résultats des outils.
"""

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=OLLAMA_TEMPERATURE,
    num_predict=800,
)

# ── Outils disponibles ────────────────────────────────────────────────────────
TOOLS = [
    search_offers,
    get_candidate_profile,
    search_offers_for_candidate,
    get_skill_gap,
    generate_roadmap,
]

# ── Compilation du graph ReAct ────────────────────────────────────────────────
graph = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=SYSTEM_PROMPT,
)
