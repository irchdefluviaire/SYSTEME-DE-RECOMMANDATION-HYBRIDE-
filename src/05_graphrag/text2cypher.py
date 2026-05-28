"""
text2cypher.py
===========================================================================
Text2Cypher sécurisé — modèle GGUF + fallback templates.

Deux modes de génération Cypher :
  1. Modèle GGUF (projectwilsen/llama3.1-8b-text2cypher-neo4j-live-4bit-gguf)
     téléchargé automatiquement via huggingface_hub, exécuté avec llama-cpp-python.
  2. Templates read-only paramétrés (fallback si llama_cpp non disponible).

Dans les deux cas, le Cypher généré est validé (lecture seule, MATCH+RETURN requis).
===========================================================================
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN = (
    "CREATE", "MERGE", "DELETE", "DETACH", "SET ", "DROP",
    "REMOVE", "LOAD CSV", "CALL DBMS",
)

# ── Schéma Neo4j exposé au modèle Text2Cypher ───────────────────────────────

_NEO4J_SCHEMA = """
Node properties:
- Métier {preferredLabel: STRING, conceptUri: STRING, description: STRING, iscoCode: STRING, mepc_base: STRING, mepc_sous: STRING, mepc_grand: STRING}
- Compétence {preferredLabel: STRING, conceptUri: STRING, description: STRING, skillType: STRING, pillar: STRING, isDigital: BOOLEAN, isGreen: BOOLEAN, isTransversal: BOOLEAN}
- OffreEmploi {titre_poste: STRING, secteur_principal: STRING, ville_principale: STRING, source: STRING, employeur: STRING, type_contrat: STRING, niveau_etudes_raw: STRING, experience_min_ans: INTEGER}
- Candidat {id: STRING, metier_vise: STRING, secteur_metier: STRING, ncf_niveau_final: INTEGER, diplome_raw: STRING, filiere_specialite: STRING, objectif: STRING}
- GroupeCompétences {preferredLabel: STRING, conceptUri: STRING}
- GroupeISCO {code: STRING, preferredLabel: STRING, niveau: INTEGER}
- DocChunk {chunk_id: STRING, source: STRING, page_number: INTEGER, section_title: STRING, subsection_title: STRING, chunk_text_preview: STRING}
- DocumentReferentiel {source: STRING, title: STRING}
- Secteur {label: STRING}
- Localisation {ville: STRING}

Relationships:
(:OffreEmploi)-[:REQUIERT]->(:Compétence)
(:Candidat)-[:POSSEDE]->(:Compétence)
(:Métier)-[:CLASSIFIE_DANS]->(:GroupeISCO)
(:Compétence)-[:PARTIE_DE]->(:GroupeCompétences)
(:OffreEmploi)-[:DANS_SECTEUR]->(:Secteur)
(:OffreEmploi)-[:LOCALISEE_A]->(:Localisation)
(:DocChunk)-[:EXTRAIT_DE]->(:DocumentReferentiel)
"""

_TEXT2CYPHER_PROMPT = """### Task:
Generate a Cypher statement to query a graph database.

### Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
The query must be read-only: only MATCH and RETURN clauses, no CREATE, MERGE, DELETE, SET.
Always add LIMIT 25 at the end unless another limit is explicitly requested.
Return only the Cypher query, no explanation.

### Schema:
{schema}

### Question:
{question}

### Cypher query:
"""


# ── Dataclass plan ───────────────────────────────────────────────────────────

@dataclass
class CypherPlan:
    intent: str
    cypher: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Validation sécurité ──────────────────────────────────────────────────────

def validate_readonly_cypher(cypher: str) -> str:
    """Valide que le Cypher est en lecture seule et ajoute LIMIT si absent."""
    upper = " ".join(cypher.upper().split())
    if any(token in upper for token in FORBIDDEN):
        raise ValueError("Cypher non autorisé: opération d'écriture détectée")
    if "MATCH" not in upper or "RETURN" not in upper:
        raise ValueError("Cypher non autorisé: MATCH et RETURN sont obligatoires")
    if "LIMIT" not in upper:
        cypher = cypher.rstrip() + "\nLIMIT 25"
    return cypher


# ── Modèle GGUF (llama-cpp-python) ──────────────────────────────────────────

_llama_instance: Any = None
_llama_load_attempted: bool = False


def _get_llama_model() -> Any | None:
    """Charge le modèle GGUF (lazy, singleton). Retourne None si non disponible."""
    global _llama_instance, _llama_load_attempted
    if _llama_load_attempted:
        return _llama_instance
    _llama_load_attempted = True

    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError:
        log.info(
            "llama-cpp-python non installé — Text2Cypher LLM désactivé. "
            "Installer avec: pip install llama-cpp-python"
        )
        return None

    try:
        from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore
    except ImportError:
        log.info("huggingface_hub non disponible — Text2Cypher LLM désactivé.")
        return None

    # Chemin local configuré manuellement
    model_path_env = os.getenv("TEXT2CYPHER_MODEL_PATH", "")
    if model_path_env and Path(model_path_env).exists():
        model_file = model_path_env
        log.info("Text2Cypher: chargement depuis TEXT2CYPHER_MODEL_PATH=%s", model_file)
    else:
        repo_id = os.getenv(
            "TEXT2CYPHER_REPO",
            "projectwilsen/llama3.1-8b-text2cypher-neo4j-live-4bit-gguf",
        )
        cache_dir = ROOT / "models" / "text2cypher"
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            files = list(list_repo_files(repo_id))
            gguf_files = sorted(f for f in files if f.endswith(".gguf"))
            if not gguf_files:
                log.warning("Aucun fichier .gguf trouvé dans %s — fallback templates", repo_id)
                return None
            # Préférer Q4_K_M pour ratio qualité/vitesse optimal
            filename = next(
                (f for f in gguf_files if "Q4_K_M" in f.upper()),
                gguf_files[0],
            )
            log.info("Text2Cypher: téléchargement %s / %s ...", repo_id, filename)
            model_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(cache_dir),
            )
            log.info("Text2Cypher: modèle prêt → %s", model_file)
        except Exception as exc:
            log.warning("Échec téléchargement modèle Text2Cypher (%s) — fallback templates", exc)
            return None

    try:
        n_ctx = int(os.getenv("TEXT2CYPHER_CTX", "4096"))
        n_gpu_layers = int(os.getenv("TEXT2CYPHER_GPU_LAYERS", "0"))
        _llama_instance = Llama(
            model_path=str(model_file),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        log.info("Modèle Text2Cypher chargé (ctx=%s, gpu_layers=%s)", n_ctx, n_gpu_layers)
        return _llama_instance
    except Exception as exc:
        log.warning("Échec chargement Llama (%s) — fallback templates", exc)
        return None


def _llm_generate_cypher(question: str) -> str | None:
    """Génère du Cypher via le modèle GGUF. Retourne None si non disponible."""
    llm = _get_llama_model()
    if llm is None:
        return None

    prompt = _TEXT2CYPHER_PROMPT.format(schema=_NEO4J_SCHEMA, question=question)
    try:
        output = llm(
            prompt,
            max_tokens=256,
            temperature=0.0,
            stop=["###", "\n\n", "Question:", "Task:", "Instructions:"],
        )
        cypher = output["choices"][0]["text"].strip()
        return cypher if cypher else None
    except Exception as exc:
        log.warning("Erreur inférence Text2Cypher LLM: %s", exc)
        return None


# ── OpenRouter (fallback entre GGUF et templates) ───────────────────────────

def _openrouter_generate_cypher(question: str) -> str | None:
    """Génère du Cypher via OpenRouter. Retourne None si non configuré ou en erreur."""
    api_key = os.getenv("API_KEY_OPEN_ROUTEUR") or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return None

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    prompt = _TEXT2CYPHER_PROMPT.format(schema=_NEO4J_SCHEMA, question=question)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 300,
        "stop": ["###", "Question:", "Task:"],
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/irchdefluviaire",
            "X-Title": "Text2Cypher-GraphRAG",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
        cypher = body["choices"][0]["message"]["content"].strip()
        log.debug("Text2Cypher via OpenRouter (model=%s): %s", model, cypher[:120])
        return cypher if cypher else None
    except Exception as exc:
        log.warning("Text2Cypher OpenRouter indisponible (%s) — fallback templates", exc)
        return None


# ── Fallback templates ───────────────────────────────────────────────────────

def build_cypher_plan(question: str) -> CypherPlan:
    """Construit un plan Cypher par templates paramétrés (sans LLM).

    N'utilise que les relations vérifiées saines dans le graphe :
      CLASSIFIE_DANS, EXTRAIT_DE, DECRIT, ALIGNE_AVEC, PREPARE_POUR.
    Les relations REQUIERT, POSSEDE, LOCALISEE_A, DANS_SECTEUR sont
    actuellement corrompues et exclues des templates.
    """
    q = question.lower()
    quoted = re.findall(r'"([^"]+)"|' r"'([^']+)'", question)
    terms = [a or b for a, b in quoted if (a or b)]
    term = terms[0] if terms else _keyword_tail(question)

    # Compétences — lookup direct sans traversal de relation corrompue
    if any(w in q for w in ["competence", "competences", "skill", "liees", "requiert"]):
        cypher = """
        MATCH (c:Compétence)
        WHERE toLower(coalesce(c.preferredLabel, '')) CONTAINS toLower($term)
           OR toLower(coalesce(c.description, '')) CONTAINS toLower($term)
        RETURN c.preferredLabel AS competence,
               c.conceptUri AS source_uri,
               c.skillType AS type_skill,
               c.pillar AS pillar,
               c.isDigital AS is_digital,
               c.isGreen AS is_green,
               c.description AS description
        ORDER BY competence
        LIMIT 15
        """
        return CypherPlan("competence_lookup", validate_readonly_cypher(cypher), {"term": term})

    # Référentiels NCF / formations / documents — EXTRAIT_DE est sain
    if any(w in q for w in ["ncf", "formation", "domaine", "referentiel", "mepc", "diplome", "nomenclature"]):
        cypher = """
        MATCH (chunk:DocChunk)-[:EXTRAIT_DE]->(doc:DocumentReferentiel)
        WHERE toLower(coalesce(chunk.section_title, '')) CONTAINS toLower($term)
           OR toLower(coalesce(chunk.subsection_title, '')) CONTAINS toLower($term)
           OR toLower(coalesce(chunk.chunk_text_preview, '')) CONTAINS toLower($term)
           OR toLower(coalesce(doc.title, '')) CONTAINS toLower($term)
        RETURN doc.title AS document,
               doc.source AS source_id,
               chunk.chunk_id AS chunk_id,
               chunk.section_title AS section,
               chunk.page_number AS page,
               chunk.chunk_text_preview AS apercu
        ORDER BY doc.source, chunk.page_number
        LIMIT 15
        """
        return CypherPlan("referentiel_lookup", validate_readonly_cypher(cypher), {"term": term})

    # Candidat — lookup direct sans POSSEDE (corrompu)
    if any(w in q for w in ["candidat", "profil", "skill gap", "roadmap"]):
        cypher = """
        MATCH (cand:Candidat)
        WHERE toLower(coalesce(cand.id, '')) CONTAINS toLower($term)
           OR toLower(coalesce(cand.metier_vise, '')) CONTAINS toLower($term)
           OR toLower(coalesce(cand.secteur_metier, '')) CONTAINS toLower($term)
        RETURN cand.id AS candidat_id,
               cand.metier_vise AS metier_vise,
               cand.secteur_metier AS secteur,
               cand.ncf_niveau_final AS niveau_ncf,
               cand.diplome_raw AS diplome,
               cand.filiere_specialite AS filiere,
               cand.objectif AS objectif
        ORDER BY candidat_id
        LIMIT 10
        """
        return CypherPlan("candidat_lookup", validate_readonly_cypher(cypher), {"term": term})

    # Offres d'emploi — sans les relations corrompues
    if any(w in q for w in ["offre", "offres", "emploi", "poste", "recrutement", "job"]):
        cypher = """
        MATCH (o:OffreEmploi)
        WHERE toLower(coalesce(o.titre_poste, '')) CONTAINS toLower($term)
           OR toLower(coalesce(o.secteur_principal, '')) CONTAINS toLower($term)
           OR toLower(coalesce(o.employeur, '')) CONTAINS toLower($term)
        RETURN o.titre_poste AS titre,
               o.employeur AS employeur,
               o.source AS source_id,
               o.secteur_principal AS secteur,
               o.ville_principale AS ville,
               o.type_contrat AS contrat,
               o.niveau_etudes_raw AS niveau_etudes
        ORDER BY titre
        LIMIT 15
        """
        return CypherPlan("offre_lookup", validate_readonly_cypher(cypher), {"term": term})

    # Métier avec classification ISCO — CLASSIFIE_DANS est sain
    if any(w in q for w in ["metier", "orientation", "carriere", "devenir", "profession"]):
        cypher = """
        MATCH (m:Métier)
        WHERE toLower(coalesce(m.preferredLabel, '')) CONTAINS toLower($term)
        OPTIONAL MATCH (m)-[:CLASSIFIE_DANS]->(g:GroupeISCO)
        RETURN m.preferredLabel AS metier,
               m.conceptUri AS source_uri,
               m.iscoCode AS code_isco,
               m.mepc_grand AS mepc_grand,
               g.preferredLabel AS groupe_isco,
               g.code AS code_groupe_isco,
               m.description AS description
        ORDER BY metier
        LIMIT 15
        """
        return CypherPlan("metier_lookup", validate_readonly_cypher(cypher), {"term": term})

    # Domaines NCF → Métiers via PREPARE_POUR (sain)
    if any(w in q for w in ["isco", "groupe", "classification", "mepc"]):
        cypher = """
        MATCH (d:DomaineDétailléNCF)-[:PREPARE_POUR]->(m:Métier)
        WHERE toLower(coalesce(d.intitule, '')) CONTAINS toLower($term)
           OR toLower(coalesce(m.preferredLabel, '')) CONTAINS toLower($term)
        OPTIONAL MATCH (m)-[:CLASSIFIE_DANS]->(g:GroupeISCO)
        RETURN d.code AS code_ncf,
               d.intitule AS domaine_ncf,
               m.preferredLabel AS metier,
               m.conceptUri AS source_uri,
               g.preferredLabel AS groupe_isco
        ORDER BY d.code
        LIMIT 15
        """
        return CypherPlan("ncf_metier_lookup", validate_readonly_cypher(cypher), {"term": term})

    # Métier par défaut (fallback)
    cypher = """
    MATCH (m:Métier)
    WHERE toLower(coalesce(m.preferredLabel, '')) CONTAINS toLower($term)
    OPTIONAL MATCH (m)-[:CLASSIFIE_DANS]->(g:GroupeISCO)
    RETURN m.preferredLabel AS metier,
           m.conceptUri AS source_uri,
           m.description AS description,
           g.preferredLabel AS groupe_isco,
           g.code AS code_isco
    ORDER BY metier
    LIMIT 15
    """
    return CypherPlan("metier_lookup", validate_readonly_cypher(cypher), {"term": term})


# ── Point d'entrée principal ─────────────────────────────────────────────────

def run_text2cypher(driver: Any, question: str, *, database: str = "neo4j") -> dict[str, Any]:
    """
    Exécute une requête Text2Cypher contre Neo4j.

    Essaie le modèle GGUF en premier ; bascule sur les templates si :
      - llama-cpp-python n'est pas installé
      - le modèle n'est pas encore téléchargé
      - le Cypher généré est invalide ou produit une erreur Neo4j
    """
    # 1. Modèle GGUF local (priorité si disponible)
    cypher_raw = _llm_generate_cypher(question)
    source = "gguf_llm"

    # 2. OpenRouter (si GGUF absent ou en erreur)
    if not cypher_raw:
        cypher_raw = _openrouter_generate_cypher(question)
        source = "openrouter"

    if cypher_raw:
        try:
            cypher = validate_readonly_cypher(cypher_raw)
            with driver.session(database=database) as session:
                rows = [dict(r) for r in session.run(cypher)]
            plan = CypherPlan("llm_generated", cypher, {})
            log.debug("Text2Cypher %s: %d résultats", source, len(rows))
            return {"plan": plan.to_dict(), "rows": rows, "source": source}
        except Exception as exc:
            log.warning(
                "Cypher %s invalide ou erreur Neo4j (%s) — fallback template", source, exc
            )

    # 3. Templates paramétrés (fallback final)
    plan = build_cypher_plan(question)
    with driver.session(database=database) as session:
        rows = [dict(r) for r in session.run(plan.cypher, **plan.params)]
    log.debug("Text2Cypher template '%s': %d résultats", plan.intent, len(rows))
    return {"plan": plan.to_dict(), "rows": rows, "source": "template"}


# ── Utilitaire ───────────────────────────────────────────────────────────────

def _keyword_tail(question: str) -> str:
    cleaned = re.sub(r"[^\w\s'-]", " ", question, flags=re.UNICODE)
    tokens = [t for t in cleaned.split() if len(t) >= 4]
    stop = {
        "quels", "quelle", "quelles", "montre", "donne", "liste", "avec",
        "pour", "dans", "competences", "competence", "sont", "liees",
        "lies", "relie", "reliees", "graphe", "neo4j", "relation",
        "relations",
    }
    kept = [t for t in tokens if t.lower() not in stop]
    return " ".join(kept[-3:]) if kept else question[:80]
