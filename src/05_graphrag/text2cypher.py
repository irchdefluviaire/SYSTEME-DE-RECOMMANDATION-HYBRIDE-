"""
text2cypher.py
===========================================================================
Text2Cypher securise par templates.

Ce module evite de laisser un LLM produire du Cypher arbitraire. Il route des
questions relationnelles frequentes vers des templates read-only, limites et
parametres. Les requetes generees sont validees avant execution.
===========================================================================
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

FORBIDDEN = ("CREATE", "MERGE", "DELETE", "DETACH", "SET ", "DROP", "REMOVE", "LOAD CSV", "CALL DBMS")


@dataclass
class CypherPlan:
    intent: str
    cypher: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_readonly_cypher(cypher: str) -> str:
    upper = " ".join(cypher.upper().split())
    if any(token in upper for token in FORBIDDEN):
        raise ValueError("Cypher non autorise: operation d'ecriture detectee")
    if "MATCH" not in upper or "RETURN" not in upper:
        raise ValueError("Cypher non autorise: MATCH et RETURN sont obligatoires")
    if "LIMIT" not in upper:
        cypher = cypher.rstrip() + "\nLIMIT 25"
    return cypher


def build_cypher_plan(question: str) -> CypherPlan:
    q = question.lower()
    quoted = re.findall(r'"([^"]+)"|' r"'([^']+)'", question)
    terms = [a or b for a, b in quoted if (a or b)]
    term = terms[0] if terms else _keyword_tail(question)

    if any(w in q for w in ["competence", "competences", "skill"]):
        cypher = """
        MATCH (s:Compétence)
        WHERE toLower(coalesce(s.preferredLabel, s.label, '')) CONTAINS toLower($term)
        OPTIONAL MATCH (o:OffreEmploi)-[:REQUIERT]->(s)
        OPTIONAL MATCH (c:Candidat)-[:POSSEDE]->(s)
        RETURN coalesce(s.preferredLabel, s.label) AS competence,
               s.conceptUri AS uri,
               count(DISTINCT o) AS nb_offres,
               count(DISTINCT c) AS nb_candidats
        ORDER BY nb_offres DESC
        LIMIT 15
        """
        return CypherPlan("competence_lookup", validate_readonly_cypher(cypher), {"term": term})

    if any(w in q for w in ["ncf", "formation", "domaine"]):
        cypher = """
        MATCH (d:DomaineDétailléNCF)
        WHERE toLower(coalesce(d.intitule, '')) CONTAINS toLower($term)
        OPTIONAL MATCH (parent)-[:CONTIENT]->(d)
        RETURN d.code AS code,
               d.intitule AS domaine,
               labels(parent)[0] AS parent_type,
               coalesce(parent.intitule, parent.label, parent.code) AS parent
        ORDER BY code
        LIMIT 20
        """
        return CypherPlan("ncf_lookup", validate_readonly_cypher(cypher), {"term": term})

    cypher = """
    MATCH (m:Métier)
    WHERE toLower(coalesce(m.preferredLabel, m.label, '')) CONTAINS toLower($term)
    OPTIONAL MATCH (m)<-[:CORRESPOND_METIER]-(o:OffreEmploi)
    RETURN coalesce(m.preferredLabel, m.label) AS metier,
           m.conceptUri AS uri,
           count(DISTINCT o) AS nb_offres
    ORDER BY nb_offres DESC
    LIMIT 15
    """
    return CypherPlan("metier_lookup", validate_readonly_cypher(cypher), {"term": term})


def run_text2cypher(driver, question: str, *, database: str = "neo4j") -> dict[str, Any]:
    plan = build_cypher_plan(question)
    with driver.session(database=database) as session:
        rows = [dict(r) for r in session.run(plan.cypher, **plan.params)]
    return {"plan": plan.to_dict(), "rows": rows}


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
