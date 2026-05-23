"""Domain tools exposed to the Agentic GraphRAG workflow.

The functions wrap the existing project modules and require live Neo4j +
PostgreSQL/pgvector connections for retrieval and graph enrichment.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from settings import DEFAULT_TOP_K, USE_REAL_DBS

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC / "05_graphrag"))
sys.path.insert(0, str(SRC / "03_knowledge_graph"))
sys.path.insert(0, str(SRC / "04_pgvector"))

from context_builder import GraphRAGContextBuilder  # noqa: E402
from roadmap_generator import generate_roadmap  # noqa: E402


CAREER_KEYWORDS = {
    "python": "Python",
    "sql": "SQL",
    "statistique": "Statistiques appliquees",
    "statistiques": "Statistiques appliquees",
    "econometrie": "Econometrie",
    "machine learning": "Machine learning",
    "apprentissage automatique": "Machine learning",
    "power bi": "Power BI",
    "tableau": "Tableau",
    "excel": "Excel avance",
    "base de donnees": "Bases de donnees",
    "bases de donnees": "Bases de donnees",
    "data warehouse": "Data warehouse",
    "etl": "ETL",
    "visualisation": "Visualisation de donnees",
    "dashboard": "Dashboarding",
    "credit scoring": "Credit scoring",
    "score de credit": "Credit scoring",
    "risque": "Gestion du risque",
    "fraude": "Detection de fraude",
    "reglementaire": "Reporting reglementaire",
    "conformite": "Conformite",
}


def _trace(step: str, message: str, status: str = "ok", **details: Any) -> dict:
    return {"step": step, "status": status, "message": message, "details": details}


def load_candidate_profile(candidat_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one normalized candidate profile from local processed data."""

    df = pd.read_parquet(ROOT / "data" / "processed" / "candidats_normalized.parquet")
    row = df[df["candidat_id"].astype(str) == str(candidat_id)]
    if row.empty:
        row = df.iloc[[0]]
        status = "warning"
        message = f"Candidat {candidat_id} introuvable, premier profil utilise."
    else:
        status = "ok"
        message = f"Candidat {candidat_id} charge."
    r = row.iloc[0]
    profile = {
        "candidat_id": str(r.get("candidat_id", candidat_id)),
        "metier_vise": str(r.get("metier_vise", "") or ""),
        "secteur_metier": str(r.get("secteur_metier", "") or ""),
        "secteur_demande": str(r.get("secteur_demande", "") or ""),
        "ncf_niveau_final": (
            int(r["ncf_niveau_final"]) if pd.notna(r.get("ncf_niveau_final")) else None
        ),
        "filiere_specialite": str(r.get("filiere_specialite", "") or ""),
        "diplome_raw": str(r.get("diplome_raw", "") or ""),
        "objectif": str(r.get("objectif", "") or "")[:250],
        "mobilite_geo_bool": bool(r.get("mobilite_geo_bool"))
        if pd.notna(r.get("mobilite_geo_bool"))
        else None,
        "text_to_embed": str(r.get("text_to_embed", "") or ""),
    }
    return profile, _trace("load_candidate", message, status, candidat_id=profile["candidat_id"])


def build_career_competency_guidance(user_query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build career guidance without pretending that a candidate profile exists."""

    target_role = _infer_target_role(user_query)
    target_domain = _infer_target_domain(user_query)
    offers_path = ROOT / "data" / "processed" / "offres_normalized.parquet"
    esco_path = ROOT / "data" / "raw" / "esco" / "skills_fr.csv"

    local_evidence = _summarize_local_offer_evidence(offers_path, target_role, target_domain)
    esco_skills = _find_esco_skill_labels(esco_path)
    priority_skills = _rank_priority_skills(local_evidence["keyword_counts"], target_domain)

    guidance = {
        "intent": "career_advice",
        "target_role": target_role,
        "target_domain": target_domain,
        "local_evidence": local_evidence,
        "priority_skills": priority_skills,
        "esco_references": esco_skills,
        "sources": [
            "data/processed/offres_normalized.parquet",
            "data/raw/esco/skills_fr.csv",
        ],
    }
    trace = _trace(
        "career_guidance",
        "Orientation metier generee sans profil candidat force.",
        target_role=target_role,
        target_domain=target_domain,
        n_local_offers=local_evidence["n_matching_offers"],
    )
    return guidance, trace


def build_retrieval_context(
    candidat_id: str,
    candidate_profile: dict[str, Any],
    *,
    top_k: int = DEFAULT_TOP_K,
    use_real_dbs: bool = USE_REAL_DBS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run vector retrieval + graph enrichment through the existing builder."""

    if not use_real_dbs:
        raise RuntimeError("Le mode simulation est desactive: Agentic GraphRAG exige Neo4j et pgvector.")
    neo4j_driver, pg_conn = _connect_required_databases()

    builder = GraphRAGContextBuilder(
        neo4j_driver=neo4j_driver,
        pg_conn=pg_conn,
        st_model=None,
        top_k_pgvector=max(20, top_k * 4),
        top_k_final=top_k,
    )
    context = builder.build_context(candidat_id, candidate_profile)
    traces = [
        _trace(
            "retrieval_vector_graph",
            "Recherche vectorielle et enrichissement graphe termines.",
            n_candidates=context.get("n_candidats", 0),
            n_top=len(context.get("top_offres", [])),
            real_databases=True,
        )
    ]
    _close_optional(pg_conn=pg_conn, neo4j_driver=neo4j_driver)
    return context, traces


def _infer_target_role(user_query: str) -> str:
    text = user_query.lower()
    if "data scientist" in text:
        return "Data scientist"
    if "data analyst" in text or "analyste" in text:
        return "Data analyst"
    if "statisticien" in text:
        return "Statisticien data"
    return "Professionnel data"


def _infer_target_domain(user_query: str) -> str:
    text = user_query.lower()
    if any(word in text for word in ("banque", "bancaire", "microfinance", "finance")):
        return "banque/finance au Cameroun"
    return "marche camerounais"


def _summarize_local_offer_evidence(path: Path, target_role: str, target_domain: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "n_matching_offers": 0,
            "sample_titles": [],
            "keyword_counts": {},
            "note": f"Fichier introuvable: {path.as_posix()}",
        }

    df = pd.read_parquet(path)
    text_cols = [col for col in ("titre_poste", "secteur_principal", "skills_raw", "details_clean", "text_to_embed") if col in df.columns]
    combined = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()

    role_terms = ("data scientist", "data analyst", "analyste", "donnees", "données", "statisticien", "business intelligence", " bi ")
    role_mask = combined.apply(lambda value: any(term in value for term in role_terms))
    if "banque/finance" in target_domain:
        domain_terms = ("banque", "bancaire", "microfinance", "finance", "financier", "assurance", "credit", "crédit")
        domain_mask = combined.apply(lambda value: any(term in value for term in domain_terms))
        filtered = df[role_mask & domain_mask].copy()
        if filtered.empty:
            filtered = df[role_mask].copy()
    else:
        filtered = df[role_mask].copy()

    keyword_counts: Counter[str] = Counter()
    filtered_text = filtered[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    for text in filtered_text:
        for raw, label in CAREER_KEYWORDS.items():
            if raw in text:
                keyword_counts[label] += 1

    title_col = "titre_poste" if "titre_poste" in filtered.columns else filtered.columns[0]
    sample_titles = [str(value) for value in filtered[title_col].dropna().head(5).tolist()]
    return {
        "n_matching_offers": int(len(filtered)),
        "sample_titles": sample_titles,
        "keyword_counts": dict(keyword_counts.most_common(12)),
        "note": "Filtre local sur offres data puis banque/finance quand disponible.",
    }


def _find_esco_skill_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    labels = pd.read_csv(path, usecols=["preferredLabel"]).dropna()["preferredLabel"].astype(str)
    wanted = (
        "analyser des donnees",
        "analyser des données",
        "science des donnees",
        "science des données",
        "statistiques",
        "apprentissage automatique",
        "bases de donnees",
        "bases de données",
        "visualisation",
        "risque",
    )
    matches = []
    for label in labels:
        low = label.lower()
        if any(term in low for term in wanted):
            matches.append(label)
        if len(matches) >= 8:
            break
    return matches


def _rank_priority_skills(keyword_counts: dict[str, int], target_domain: str) -> list[dict[str, Any]]:
    baseline = [
        ("SQL", "extraire, joindre et auditer les donnees bancaires"),
        ("Python", "automatiser les analyses, modeliser et produire des pipelines reproductibles"),
        ("Statistiques appliquees", "quantifier l'incertitude et eviter les conclusions fragiles"),
        ("Machine learning", "scoring, segmentation, prevision et detection d'anomalies"),
        ("Visualisation de donnees", "transformer les resultats en decisions lisibles"),
        ("ETL", "nettoyer et fiabiliser les donnees operationnelles"),
    ]
    if "banque/finance" in target_domain:
        baseline.extend(
            [
                ("Credit scoring", "relier les modeles aux decisions de credit"),
                ("Gestion du risque", "parler le langage metier banque et portefeuille"),
                ("Conformite", "tenir compte des contraintes de donnees sensibles et reporting"),
            ]
        )
    rows = []
    for label, reason in baseline:
        rows.append(
            {
                "competence": label,
                "priorite": "forte" if label in {"SQL", "Python", "Statistiques appliquees", "Machine learning"} else "moyenne",
                "evidence_locale": int(keyword_counts.get(label, 0)),
                "pourquoi": reason,
            }
        )
    return rows


def extract_skill_gaps(top_offers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict]:
    """Normalize skill-gap fields for the critic and UI."""

    gaps: list[dict[str, Any]] = []
    for offer in top_offers:
        gaps.append(
            {
                "offre_id": offer.get("offre_id"),
                "titre": offer.get("titre"),
                "taux_match": offer.get("taux_match", 0),
                "competences_acquises": offer.get("acquises", []),
                "competences_manquantes": offer.get("manquantes", []),
                "essentielles_manquantes": offer.get("ess_manq", []),
                "nb_manquantes": len(offer.get("manquantes", [])),
                "nb_essentielles_manquantes": len(offer.get("ess_manq", [])),
            }
        )
    return gaps, _trace("skill_gap", "Skill gaps normalises.", n=len(gaps))


def rank_hybrid_offers(top_offers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict]:
    """Sort offers by the hybrid score computed by the GraphRAG builder."""

    ranked = sorted(top_offers, key=lambda item: item.get("score_hybride", 0), reverse=True)
    return ranked, _trace("hybrid_scoring", "Offres classees par score hybride.", n=len(ranked))


def build_training_roadmap(
    candidate_profile: dict[str, Any],
    top_offer: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate a structured training roadmap for the best offer."""

    missing = [
        {"label": label, "importance": "essential" if label in top_offer.get("ess_manq", []) else "optional"}
        for label in top_offer.get("manquantes", [])
    ]
    roadmap = generate_roadmap(
        candidat=candidate_profile,
        top_offre=top_offer,
        competences_manquantes=missing,
        score_actuel=float(top_offer.get("score_hybride", 0.0)),
    )
    return roadmap, _trace("roadmap", "Roadmap de formation generee.", n_missing=len(missing))


def _connect_required_databases():
    """Open and validate required Neo4j and PostgreSQL/pgvector connections."""

    neo4j_driver = None
    pg_conn = None
    try:
        from neo4j import GraphDatabase
        from config_neo4j import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        neo4j_driver.verify_connectivity()
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            session.run("RETURN 1 AS ok").single()
    except Exception as exc:
        _close_optional(pg_conn=pg_conn, neo4j_driver=neo4j_driver)
        raise RuntimeError(f"Connexion Neo4j impossible. Verifie NEO4J_URI/USER/PASSWORD: {exc}") from exc

    try:
        import psycopg
        from config_pgvector import PG_CONN

        pg_conn = psycopg.connect(**PG_CONN)
        with pg_conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_name = 'embeddings'
                )
                """
            )
            has_embeddings = bool(cur.fetchone()[0])
        if not has_embeddings:
            raise RuntimeError("table PostgreSQL 'embeddings' introuvable")
    except Exception as exc:
        _close_optional(pg_conn=pg_conn, neo4j_driver=neo4j_driver)
        raise RuntimeError(f"Connexion pgvector impossible. Verifie PG_HOST/PG_DB/PG_USER/PG_PASSWORD: {exc}") from exc

    return neo4j_driver, pg_conn


def _close_optional(*, pg_conn=None, neo4j_driver=None) -> None:
    for obj in (pg_conn, neo4j_driver):
        if obj is not None:
            try:
                obj.close()
            except Exception:
                pass
