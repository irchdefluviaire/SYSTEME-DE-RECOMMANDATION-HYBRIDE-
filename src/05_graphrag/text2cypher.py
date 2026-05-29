"""Secure Text2Cypher with a local Hugging Face model plus safe templates.

Primary model:
    neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1

The generated Cypher is always validated as read-only before execution. If the
Hugging Face model cannot be loaded or generates invalid Cypher, deterministic
read-only templates are used.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

TEXT2CYPHER_MODEL = "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"
FORBIDDEN = (
    "CREATE", "MERGE", "DELETE", "DETACH", "SET ", "DROP",
    "REMOVE", "LOAD CSV", "CALL DBMS",
)

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

_hf_tokenizer: Any | None = None
_hf_model: Any | None = None
_hf_load_attempted = False


@dataclass
class CypherPlan:
    intent: str
    cypher: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_readonly_cypher(cypher: str) -> str:
    upper = " ".join(str(cypher).upper().split())
    if any(token in upper for token in FORBIDDEN):
        raise ValueError("Cypher non autorise: operation d'ecriture detectee")
    if "MATCH" not in upper or "RETURN" not in upper:
        raise ValueError("Cypher non autorise: MATCH et RETURN sont obligatoires")
    if "LIMIT" not in upper:
        cypher = str(cypher).rstrip() + "\nLIMIT 25"
    return str(cypher)


def _extract_cypher(text: str) -> str | None:
    cleaned = re.sub(r"```(?:cypher)?|```", "", str(text), flags=re.IGNORECASE).strip()
    match = re.search(r"(?is)\bMATCH\b.+", cleaned)
    if not match:
        return None
    cypher = match.group(0).strip()
    cut_markers = ["\n###", "\nQuestion:", "\nExplanation:", "\nNotes:"]
    for marker in cut_markers:
        pos = cypher.find(marker)
        if pos >= 0:
            cypher = cypher[:pos].strip()
    return cypher or None


def _get_hf_model() -> tuple[Any, Any] | tuple[None, None]:
    global _hf_tokenizer, _hf_model, _hf_load_attempted
    if _hf_load_attempted:
        return _hf_tokenizer, _hf_model
    _hf_load_attempted = True

    model_id = os.getenv("TEXT2CYPHER_MODEL", TEXT2CYPHER_MODEL)
    if os.getenv("TEXT2CYPHER_DISABLE_HF", "0") == "1":
        log.info("Text2Cypher Hugging Face desactive par TEXT2CYPHER_DISABLE_HF=1")
        return None, None

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        log.warning("transformers/torch indisponibles pour Text2Cypher HF: %s", exc)
        return None, None

    try:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        dtype_name = os.getenv("TEXT2CYPHER_TORCH_DTYPE", "auto")
        torch_dtype = "auto"
        if dtype_name in {"float16", "fp16"}:
            torch_dtype = torch.float16
        elif dtype_name in {"bfloat16", "bf16"}:
            torch_dtype = torch.bfloat16

        log.info("Chargement Text2Cypher HF: %s", model_id)
        _hf_tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        _hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            torch_dtype=torch_dtype,
            device_map=os.getenv("TEXT2CYPHER_DEVICE_MAP", "auto"),
            low_cpu_mem_usage=True,
        )
        return _hf_tokenizer, _hf_model
    except Exception as exc:
        log.warning("Modele Text2Cypher HF indisponible (%s) - fallback templates", exc)
        _hf_tokenizer = None
        _hf_model = None
        return None, None


def _hf_generate_cypher(question: str) -> str | None:
    tokenizer, model = _get_hf_model()
    if tokenizer is None or model is None:
        return None

    prompt = _TEXT2CYPHER_PROMPT.format(schema=_NEO4J_SCHEMA, question=question)
    max_new_tokens = int(os.getenv("TEXT2CYPHER_MAX_NEW_TOKENS", "256"))
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return _extract_cypher(generated)
    except Exception as exc:
        log.warning("Erreur inference Text2Cypher HF: %s", exc)
        return None


def build_cypher_plan(question: str) -> CypherPlan:
    q = question.lower()
    quoted = re.findall(r'"([^"]+)"|' r"'([^']+)'", question)
    terms = [a or b for a, b in quoted if (a or b)]
    term = terms[0] if terms else _keyword_tail(question)

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


def run_text2cypher(driver: Any, question: str, *, database: str = "neo4j") -> dict[str, Any]:
    cypher_raw = _hf_generate_cypher(question)
    source = "hf_text2cypher_gemma"

    if cypher_raw:
        try:
            cypher = validate_readonly_cypher(cypher_raw)
            with driver.session(database=database) as session:
                rows = [dict(r) for r in session.run(cypher)]
            plan = CypherPlan("hf_generated", cypher, {})
            log.debug("Text2Cypher HF: %d resultats", len(rows))
            return {"plan": plan.to_dict(), "rows": rows, "source": source}
        except Exception as exc:
            log.warning("Cypher HF invalide ou erreur Neo4j (%s) - fallback template", exc)

    plan = build_cypher_plan(question)
    with driver.session(database=database) as session:
        rows = [dict(r) for r in session.run(plan.cypher, **plan.params)]
    log.debug("Text2Cypher template '%s': %d resultats", plan.intent, len(rows))
    return {"plan": plan.to_dict(), "rows": rows, "source": "template"}


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
