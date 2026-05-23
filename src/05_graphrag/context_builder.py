"""
context_builder.py
===========================================================================
Module 05 — Construction du contexte GraphRAG

Assemble le contexte structuré à partir de :
  1. pgvector ANN → top-k offres sémantiquement proches
  2. Neo4j Cypher → skill gap + compatibilité + chemins NCF
  3. Score collaboratif → historique candidats similaires

Le contexte produit est injecté dans le prompt du LLM 2 (Mistral-7B / GPT-4o)
pour la génération des recommandations et roadmaps en français.

Architecture GraphRAG :
  retrieval (Neo4j + pgvector) → augmentation (contexte structuré) → generation (LLM 2)
===========================================================================
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)

# Chemins inter-modules
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "03_knowledge_graph"))
sys.path.insert(0, str(ROOT / "src" / "04_pgvector"))


# ─────────────────────────────────────────────────────────────────────────
# REQUÊTES CYPHER INTÉGRÉES
# (version autonome, sans import depuis module 03)
# ─────────────────────────────────────────────────────────────────────────

Q_GRAPH_ENRICHMENT = """
MATCH (c:Candidat {id: $cid})
MATCH (o:OffreEmploi {id: $oid})
OPTIONAL MATCH (c)-[:A_NIVEAU]->(cn:NiveauFormationNCF)
OPTIONAL MATCH (o)-[:REQUIERT_NIVEAU_NCF|REQUIERT_NIVEAU]->(on:NiveauFormationNCF)
OPTIONAL MATCH (c)-[:A_FORMATION]->(cf:DomaineDétailléNCF)
OPTIONAL MATCH (o)-[:DANS_SECTEUR]->(sect:Secteur)
OPTIONAL MATCH (o)-[:LOCALISEE_A]->(loc:Localisation)
RETURN
  c.ncf_niveau_final AS candidat_ncf,
  c.metier_vise AS metier_vise,
  c.secteur_metier AS secteur_metier,
  c.filiere_specialite AS filiere_specialite,
  cn.code AS candidat_ncf_code,
  cn.intitule AS candidat_ncf_label,
  on.code AS offre_ncf_code,
  on.intitule AS offre_ncf_label,
  collect(DISTINCT cf.intitule)[0..5] AS formations_candidat,
  sect.label AS secteur,
  loc.ville AS ville,
  o.type_contrat AS type_contrat,
  o.ncf_niveau_code AS ncf_requis,
  o.skills_raw AS skills_raw,
  o.details_clean AS details
"""

Q_SKILL_GAP_EXACT = """
MATCH (o:OffreEmploi {id: $oid})
OPTIONAL MATCH (c:Candidat {id: $cid})-[cp:POSSEDE]->(sc:Compétence)
OPTIONAL MATCH (o)-[rq:REQUIERT]->(sr:Compétence)
WITH
  collect(DISTINCT sc.conceptUri) AS cand_uris,
  collect(DISTINCT {
    uri: sr.conceptUri,
    label: sr.preferredLabel,
    type: coalesce(rq.relationType, 'essential'),
    confidence: coalesce(rq.confidence, 0.0)
  }) AS offre_skills
WITH cand_uris, [x IN offre_skills WHERE x.uri IS NOT NULL] AS offre_skills
RETURN
  [x IN offre_skills WHERE x.uri IN cand_uris] AS acquises,
  [x IN offre_skills WHERE NOT x.uri IN cand_uris] AS manquantes,
  [x IN offre_skills WHERE NOT x.uri IN cand_uris AND x.type = 'essential'] AS essentielles_manquantes,
  size(offre_skills) AS n_total,
  toFloat(size([x IN offre_skills WHERE x.uri IN cand_uris])) /
    CASE WHEN size(offre_skills) > 0 THEN size(offre_skills) ELSE 1 END AS taux
"""

Q_OFFRE_DETAILS = """
MATCH (o:OffreEmploi {id: $oid})
OPTIONAL MATCH (o)-[:DANS_SECTEUR]->(s:Secteur)
OPTIONAL MATCH (o)-[:LOCALISEE_A]->(l:Localisation)
RETURN o.titre_poste AS titre,
       o.employeur   AS employeur,
       o.type_contrat AS contrat,
       o.ncf_niveau_code AS ncf_requis,
       o.details_clean AS details,
       s.label AS secteur,
       l.ville AS ville
"""

Q_NCF_CHEMIN = """
MATCH (c:Candidat {id: $cid})-[:A_NIVEAU]->(n:NiveauFormationNCF)
RETURN n.code AS ncf_code, n.intitule AS ncf_label
"""


# ─────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────

class GraphRAGContextBuilder:
    """
    Construit le contexte structuré pour le LLM 2 (GraphRAG pattern).

    Utilise :
      - neo4j_driver  : connexion Neo4j (optionnel — mode simulation si None)
      - pg_conn       : connexion psycopg2 pgvector (optionnel)
      - st_model      : SentenceTransformer fine-tuné (optionnel)
    """

    def __init__(
        self,
        neo4j_driver=None,
        pg_conn=None,
        st_model=None,
        top_k_pgvector: int = 20,
        top_k_final: int = 5,
    ):
        self.driver     = neo4j_driver
        self.pg         = pg_conn
        self.model      = st_model
        self.top_k_pgv  = top_k_pgvector
        self.top_k_fin  = top_k_final

    # ── Étape 1 : ANN pgvector ──────────────────────────────────────────
    def _ann_search(self, candidat_id: str) -> list[dict]:
        """
        Requête ANN pgvector : top-k offres sémantiquement proches.
        Opérateur <=> (cosine distance) avec index HNSW.
        """
        if self.pg is None:
            # Mode simulation
            return self._simulate_ann(candidat_id)

        sql = """
        SELECT o.entity_id, o.label_fr, o.neo4j_node_id,
               1 - (c.embedding <=> o.embedding) AS cosine_sim
        FROM   embeddings c, embeddings o
        WHERE  c.entity_kind = 'CANDIDAT'
          AND  c.entity_id   = %s
          AND  o.entity_kind = 'OFFRE_EMPLOI'
        ORDER  BY c.embedding <=> o.embedding
        LIMIT  %s;
        """
        with self.pg.cursor() as cur:
            cur.execute("SET hnsw.ef_search = 100")
            cur.execute(sql, (candidat_id, self.top_k_pgv))
            rows = cur.fetchall()

        return [
            {"offre_id": r[0], "titre": r[1], "neo4j_id": r[2],
             "score_sem": round(float(r[3]), 4)}
            for r in rows
        ]

    def _simulate_ann(self, candidat_id: str) -> list[dict]:
        """Simulation ANN sans pgvector (utilise les données réelles)."""
        import pandas as pd, hashlib
        from pathlib import Path as P

        PROC = P(__file__).resolve().parent.parent.parent / "data" / "processed"
        df_o = pd.read_parquet(PROC / "offres_normalized.parquet")
        df_c = pd.read_parquet(PROC / "candidats_normalized.parquet")

        # Trouver le candidat
        cand_row = df_c[df_c["candidat_id"].astype(str) == str(candidat_id)]
        if cand_row.empty:
            cand_row = df_c.iloc[[0]]

        # Encodage simulé déterministe
        def encode_sim(text, dim=384):
            h = int(hashlib.md5(str(text).encode()).hexdigest(), 16)
            rng = np.random.default_rng(h % (2**32))
            v = rng.standard_normal(dim).astype(np.float32)
            return v / np.linalg.norm(v)

        cand_vec  = encode_sim(cand_row["text_to_embed"].iloc[0])
        offre_vecs = np.array([encode_sim(t) for t in df_o["text_to_embed"].fillna("").tolist()[:200]])
        sims = offre_vecs @ cand_vec
        top_idx = np.argsort(-sims)[:self.top_k_pgv]

        return [
            {
                "offre_id":  df_o.iloc[i]["offre_id"],
                "titre":     df_o.iloc[i]["titre_poste"],
                "neo4j_id":  df_o.iloc[i]["offre_id"],
                "score_sem": round(float(sims[i]), 4),
                "secteur":   df_o.iloc[i].get("secteur_principal", ""),
                "ville":     df_o.iloc[i].get("ville_principale", ""),
                "type_contrat": df_o.iloc[i].get("type_contrat_norm", ""),
                "ncf_code":  df_o.iloc[i].get("ncf_niveau_code"),
                "skills":    df_o.iloc[i].get("skills_raw", ""),
                "details":   str(df_o.iloc[i].get("details_clean", ""))[:200],
            }
            for i in top_idx
        ]

    # ── Étape 2 : Enrichissement Neo4j ──────────────────────────────────
    def _enrich_with_neo4j(self, candidat_id: str, offre: dict) -> dict:
        """Enrichit une offre candidate avec les données Neo4j."""
        if self.driver is None:
            return self._simulate_neo4j(candidat_id, offre)

        with self.driver.session() as session:
            graph_row = session.run(
                Q_GRAPH_ENRICHMENT,
                cid=candidat_id,
                oid=offre["offre_id"],
            ).single()
            skill_gap = session.run(
                Q_SKILL_GAP_EXACT,
                cid=candidat_id,
                oid=offre["offre_id"],
            ).single()
            ncf = session.run(Q_NCF_CHEMIN, cid=candidat_id).single()

        graph = dict(graph_row) if graph_row else {}
        if skill_gap and int(skill_gap.get("n_total", 0) or 0) > 0:
            acquired = [item.get("label", "") for item in (skill_gap.get("acquises") or []) if item.get("label")]
            missing = [item.get("label", "") for item in (skill_gap.get("manquantes") or []) if item.get("label")]
            essential_missing = [
                item.get("label", "")
                for item in (skill_gap.get("essentielles_manquantes") or [])
                if item.get("label")
            ]
            taux = float(skill_gap.get("taux", 0.0) or 0.0)
            graph_schema = "ESCO_SKILL_GAP"
        else:
            acquired, missing, essential_missing, taux = self._score_existing_graph(
                candidat_id=candidat_id,
                offre={**offre, **graph},
            )
            graph_schema = "NCF_SECTEUR_FORMATION_FALLBACK"

        return {
            **offre,
            "secteur": graph.get("secteur") or offre.get("secteur"),
            "ville": graph.get("ville") or offre.get("ville"),
            "type_contrat": graph.get("type_contrat") or offre.get("type_contrat"),
            "ncf_code": graph.get("offre_ncf_code") or graph.get("ncf_requis") or offre.get("ncf_code"),
            "skills": graph.get("skills_raw") or offre.get("skills", ""),
            "details": str(graph.get("details") or offre.get("details", ""))[:200],
            "acquises":    acquired[:5],
            "manquantes":  missing[:5],
            "ess_manq":    essential_missing[:3],
            "taux_match":  round(float(taux), 3),
            "ncf_cand":    ncf.get("ncf_code") if ncf else None,
            "graph_signals": {
                "schema": graph_schema,
                "offre_ncf": graph.get("offre_ncf_code") or graph.get("ncf_requis"),
                "candidat_ncf": graph.get("candidat_ncf") or graph.get("candidat_ncf_code"),
                "secteur_offre": graph.get("secteur"),
                "formations_candidat": graph.get("formations_candidat") or [],
                "n_competences_requises": int(skill_gap.get("n_total", 0) or 0) if skill_gap else 0,
            },
        }

    @staticmethod
    def _as_int(value) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        stop = {"et", "de", "du", "des", "la", "le", "les", "en", "a", "à", "and", "&"}
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text or ""))
        return {token for token in cleaned.split() if len(token) >= 4 and token not in stop}

    def _score_existing_graph(self, *, candidat_id: str, offre: dict) -> tuple[list[str], list[str], list[str], float]:
        """Score graph compatibility with the relations that are actually loaded.

        Current Neo4j does not materialize Candidat-[:POSSEDE]->Compétence nor
        OffreEmploi-[:REQUIERT]->Compétence. The honest graph score therefore
        uses NCF level, sector lexical overlap, and candidate formation labels.
        """

        cand_ncf = self._as_int(offre.get("candidat_ncf") or offre.get("candidat_ncf_code"))
        offer_ncf = self._as_int(offre.get("offre_ncf_code") or offre.get("ncf_requis") or offre.get("ncf_code"))
        if cand_ncf is None or offer_ncf is None:
            ncf_score = 0.5
            ncf_label = "Niveau NCF non totalement renseigne"
            ncf_missing = []
        elif cand_ncf >= offer_ncf:
            ncf_score = 1.0
            ncf_label = f"Niveau NCF compatible ({cand_ncf} >= {offer_ncf})"
            ncf_missing = []
        else:
            gap = offer_ncf - cand_ncf
            ncf_score = max(0.0, 1.0 - 0.25 * gap)
            ncf_label = f"Niveau NCF partiellement compatible ({cand_ncf} < {offer_ncf})"
            ncf_missing = [f"Monter du niveau NCF {cand_ncf} vers {offer_ncf}"]

        sector_tokens = self._tokenize(offre.get("secteur"))
        candidate_tokens = self._tokenize(
            " ".join(
                [
                    str(offre.get("metier_vise", "")),
                    str(offre.get("secteur_metier", "")),
                    str(offre.get("filiere_specialite", "")),
                    str(offre.get("formations_candidat", "")),
                ]
            )
        )
        # Candidate fields are not in the Neo4j row, so fallback to id-based
        # scoring through offer terms only when no candidate tokens are present.
        overlap = sector_tokens & candidate_tokens
        sector_score = len(overlap) / max(len(sector_tokens), 1) if sector_tokens else 0.5

        skill_tokens = self._tokenize(offre.get("skills_raw") or offre.get("skills"))
        formation_tokens = self._tokenize(" ".join(offre.get("formations_candidat") or []))
        skill_overlap = skill_tokens & formation_tokens
        formation_score = len(skill_overlap) / max(len(skill_tokens), 1) if skill_tokens else 0.5

        taux = 0.55 * ncf_score + 0.25 * sector_score + 0.20 * formation_score
        acquired = [ncf_label]
        if overlap:
            acquired.append("Secteur rapproche: " + ", ".join(sorted(overlap)[:3]))
        if skill_overlap:
            acquired.append("Formation rapprochee: " + ", ".join(sorted(skill_overlap)[:3]))

        missing = ncf_missing
        if sector_score == 0 and sector_tokens:
            missing.append("Justifier l'experience dans le secteur " + str(offre.get("secteur")))
        if formation_score == 0 and skill_tokens:
            missing.append("Renforcer: " + ", ".join(sorted(skill_tokens)[:3]))

        essential_missing = ncf_missing[:1]
        return acquired, missing, essential_missing, taux

    def _simulate_neo4j(self, candidat_id: str, offre: dict) -> dict:
        """Simulation Neo4j sans driver (données synthétiques cohérentes)."""
        import pandas as pd
        from pathlib import Path as P

        PROC = P(__file__).resolve().parent.parent.parent / "data" / "processed"
        df_c = pd.read_parquet(PROC / "candidats_normalized.parquet")
        cand_row = df_c[df_c["candidat_id"].astype(str) == str(candidat_id)]
        if cand_row.empty:
            cand_row = df_c.iloc[[0]]

        cand = cand_row.iloc[0]
        skills_offre = [s.strip() for s in str(offre.get("skills","")).split(",") if s.strip()]
        metier = str(cand.get("metier_vise", ""))
        secteur = str(cand.get("secteur_metier", ""))

        # Simuler correspondances partielles
        np.random.seed(hash(str(candidat_id) + offre["offre_id"]) % 2**32)
        n_skills = len(skills_offre)
        n_acquis = max(1, int(n_skills * np.random.uniform(0.3, 0.8)))
        acquises  = skills_offre[:n_acquis]
        manquantes = skills_offre[n_acquis:]

        return {
            **offre,
            "acquises":   acquises[:4],
            "manquantes": manquantes[:4],
            "ess_manq":   manquantes[:2],
            "taux_match": round(n_acquis / max(n_skills, 1), 3),
            "ncf_cand":   int(cand.get("ncf_niveau_final", 0) or 0),
        }

    # ── Étape 3 : Score hybride ──────────────────────────────────────────
    @staticmethod
    def _clamp(value, low: float = 0.0, high: float = 1.0) -> float:
        try:
            return max(low, min(high, float(value)))
        except (TypeError, ValueError):
            return low

    @staticmethod
    def _first_present(*values):
        for value in values:
            if value is None:
                continue
            try:
                if value != value:
                    continue
            except TypeError:
                pass
            if str(value) in {"", "<NA>", "nan", "NaN", "None"}:
                continue
            return value
        return None

    def _ncf_fit(self, offre: dict) -> tuple[float, int | None, str]:
        cand_ncf = self._as_int(
            self._first_present(
                offre.get("ncf_cand"),
                offre.get("candidat_ncf"),
                offre.get("candidat_ncf_code"),
                offre.get("graph_signals", {}).get("candidat_ncf"),
            )
        )
        offer_ncf = self._as_int(
            self._first_present(
                offre.get("ncf_code"),
                offre.get("ncf_requis"),
                offre.get("offre_ncf_code"),
                offre.get("graph_signals", {}).get("offre_ncf"),
            )
        )
        if cand_ncf is None or offer_ncf is None:
            return 0.55, None, "niveau formation incomplet"
        gap = offer_ncf - cand_ncf
        if gap <= 0:
            return 1.0, gap, "niveau formation compatible"
        if gap == 1:
            return 0.72, gap, "niveau juste inferieur au requis"
        if gap == 2:
            return 0.45, gap, "ecart de niveau formation important"
        return 0.20, gap, "ecart de niveau formation bloquant"

    def _sector_fit(self, offre: dict) -> tuple[float, list[str]]:
        candidate_text = " ".join(
            [
                str(offre.get("metier_vise", "")),
                str(offre.get("secteur_metier", "")),
                str(offre.get("filiere_specialite", "")),
                str(offre.get("formations_candidat", "")),
            ]
        )
        offer_text = " ".join(
            [
                str(offre.get("titre", "")),
                str(offre.get("secteur", "")),
                str(offre.get("skills", "")),
                str(offre.get("details", "")),
            ]
        )
        cand_tokens = self._tokenize(candidate_text)
        offer_tokens = self._tokenize(offer_text)
        if not cand_tokens or not offer_tokens:
            return 0.50, []
        overlap = sorted(cand_tokens & offer_tokens)
        return self._clamp(len(overlap) / max(min(len(cand_tokens), len(offer_tokens)), 1)), overlap[:5]

    def _recruitment_verdict(self, score: float, n_essential: int, ncf_gap: int | None) -> tuple[str, list[str]]:
        blockers: list[str] = []
        if n_essential >= 4:
            blockers.append(f"{n_essential} competences essentielles manquantes")
        elif n_essential >= 2:
            blockers.append(f"{n_essential} competences essentielles a combler")
        if ncf_gap is not None and ncf_gap >= 2:
            blockers.append(f"niveau NCF inferieur de {ncf_gap} niveau(x)")

        if score >= 0.72 and n_essential <= 1 and not (ncf_gap is not None and ncf_gap > 0):
            verdict = "pret_a_postuler"
        elif score >= 0.55 and n_essential <= 3 and not (ncf_gap is not None and ncf_gap >= 3):
            verdict = "postuler_avec_plan_de_montee_en_competence"
        elif score >= 0.42:
            verdict = "vivier_a_developper"
        else:
            verdict = "hors_cible_actuel"
        return verdict, blockers

    def _compute_fit_profile(self, offre_enriched: dict) -> dict:
        semantic_fit = self._clamp(offre_enriched.get("score_sem", 0.5))
        raw_skill_fit = self._clamp(offre_enriched.get("taux_match", 0.5))
        n_essential = len(offre_enriched.get("ess_manq", []) or [])
        essential_penalty = min(0.45, n_essential * 0.12)
        skill_fit = self._clamp(raw_skill_fit - essential_penalty)
        ncf_fit, ncf_gap, ncf_message = self._ncf_fit(offre_enriched)
        sector_fit, sector_overlap = self._sector_fit(offre_enriched)

        score = (
            0.35 * semantic_fit
            + 0.30 * skill_fit
            + 0.20 * ncf_fit
            + 0.15 * sector_fit
        )
        if ncf_gap is not None and ncf_gap >= 3:
            score *= 0.75
        if n_essential >= 4:
            score *= 0.80
        score = round(self._clamp(score), 4)
        verdict, blockers = self._recruitment_verdict(score, n_essential, ncf_gap)

        strengths = []
        if semantic_fit >= 0.65:
            strengths.append("proximite semantique forte avec le profil")
        if raw_skill_fit >= 0.70:
            strengths.append("bonne couverture des competences")
        if ncf_fit >= 0.90:
            strengths.append(ncf_message)
        if sector_fit >= 0.50 and sector_overlap:
            strengths.append("alignement sectoriel: " + ", ".join(sector_overlap[:3]))

        development_priorities = []
        development_priorities.extend(offre_enriched.get("ess_manq", [])[:3])
        if ncf_gap is not None and ncf_gap > 0:
            development_priorities.append(f"combler le gap NCF de {ncf_gap} niveau(x)")
        for item in offre_enriched.get("manquantes", []):
            if item not in development_priorities:
                development_priorities.append(item)
            if len(development_priorities) >= 5:
                break

        return {
            "score": score,
            "verdict": verdict,
            "dimensions": {
                "semantique": round(semantic_fit, 4),
                "competences": round(skill_fit, 4),
                "competences_brut": round(raw_skill_fit, 4),
                "niveau_ncf": round(ncf_fit, 4),
                "secteur_metier": round(sector_fit, 4),
            },
            "poids": {
                "semantique": 0.35,
                "competences": 0.30,
                "niveau_ncf": 0.20,
                "secteur_metier": 0.15,
            },
            "ncf_gap": ncf_gap,
            "ncf_message": ncf_message,
            "facteurs_bloquants": blockers,
            "points_forts": strengths,
            "priorites_developpement": development_priorities[:5],
        }

    def _compute_hybrid_score(self, offre_enriched: dict) -> float:
        """Recruitment-oriented score with explicit fit dimensions."""

        fit_profile = self._compute_fit_profile(offre_enriched)
        offre_enriched["fit_profile"] = fit_profile
        offre_enriched["score_components"] = fit_profile["dimensions"]
        offre_enriched["verdict_recrutement"] = fit_profile["verdict"]
        offre_enriched["facteurs_bloquants"] = fit_profile["facteurs_bloquants"]
        offre_enriched["priorites_developpement"] = fit_profile["priorites_developpement"]
        return fit_profile["score"]

    # ── Assemblage du contexte ───────────────────────────────────────────
    def build_context(
        self, candidat_id: str, candidat_profile: dict
    ) -> dict:
        """
        Pipeline complet : ANN → Cypher → scoring → contexte structuré.

        Returns:
            dict avec :
              - top_offres     : liste ordonnée par score_hybride
              - candidat       : profil résumé
              - context_text   : texte prêt à injecter dans le prompt LLM
        """
        log.info(f"GraphRAG context pour candidat {candidat_id}")

        # Étape 1 : ANN pgvector
        candidates = self._ann_search(candidat_id)
        log.info(f"  ANN pgvector → {len(candidates)} offres candidates")

        # Étape 2 : Enrichissement Neo4j + scoring
        enriched = []
        for offre in candidates:
            e = self._enrich_with_neo4j(candidat_id, offre)
            e["score_hybride"] = self._compute_hybrid_score(e)
            enriched.append(e)

        # Trier par score hybride
        enriched.sort(key=lambda x: x["score_hybride"], reverse=True)
        top = enriched[:self.top_k_fin]

        # Étape 3 : Assembler le texte de contexte
        context_text = self._format_context(candidat_profile, top)

        return {
            "candidat_id":   candidat_id,
            "candidat":      candidat_profile,
            "top_offres":    top,
            "n_candidats":   len(candidates),
            "context_text":  context_text,
        }

    def _format_context(self, candidat: dict, top_offres: list) -> str:
        """Formate le contexte en texte structuré pour le prompt LLM."""
        lines = [
            "=== PROFIL DU CANDIDAT ===",
            f"Métier visé     : {candidat.get('metier_vise', 'Non précisé')}",
            f"Secteur souhaité: {candidat.get('secteur_metier', 'Non précisé')}",
            f"Niveau NCF      : {candidat.get('ncf_niveau_final', 'Non précisé')}",
            f"Filière         : {candidat.get('filiere_specialite', 'Non précisé')}",
            f"Objectif        : {str(candidat.get('objectif', ''))[:150]}",
            "",
            f"=== TOP {len(top_offres)} OFFRES RECOMMANDÉES ===",
        ]

        for i, o in enumerate(top_offres, 1):
            lines += [
                f"\n--- Offre {i} [score hybride : {o['score_hybride']:.3f}] ---",
                f"Poste     : {o.get('titre', '')}",
                f"Secteur   : {o.get('secteur', '')}  |  Ville : {o.get('ville', '')}",
                f"Contrat   : {o.get('type_contrat', o.get('contrat', ''))}",
                f"Match     : {o.get('taux_match', 0):.0%}  "
                f"(sem={o['score_sem']:.3f})",
                f"Verdict   : {o.get('verdict_recrutement', 'non calcule')}",
                f"Dimensions: {json.dumps(o.get('score_components', {}), ensure_ascii=False)}",
                f"Acquises  : {', '.join(o.get('acquises', [])[:3]) or 'Aucune déclarée'}",
                f"Manquantes: {', '.join(o.get('manquantes', [])[:3]) or 'Aucune'}",
                f"Essentielles manquantes: {', '.join(o.get('ess_manq', [])) or 'Aucune'}",
                f"Priorites developpement: {', '.join(o.get('priorites_developpement', [])[:3]) or 'Aucune'}",
            ]

        return "\n".join(lines)
