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

Q_SKILL_GAP = """
MATCH (c:Candidat   {id: $cid})-[:POSSEDE]->(sc:Compétence)
MATCH (o:OffreEmploi{id: $oid})-[r:REQUIERT]->(sr:Compétence)
WITH collect(DISTINCT sc.conceptUri) AS cand_uris,
     collect(DISTINCT {uri:sr.conceptUri, label:sr.preferredLabel,
             type:r.relationType}) AS offre_skills
RETURN
  [x IN offre_skills WHERE x.uri IN cand_uris] AS acquises,
  [x IN offre_skills WHERE NOT x.uri IN cand_uris] AS manquantes,
  size(offre_skills) AS n_total,
  toFloat(size([x IN offre_skills WHERE x.uri IN cand_uris])) /
    CASE WHEN size(offre_skills)>0 THEN size(offre_skills) ELSE 1 END AS taux
"""

Q_OFFRE_DETAILS = """
MATCH (o:OffreEmploi {id: $oid})
OPTIONAL MATCH (o)-[:DANS_SECTEUR]->(s:Secteur)
OPTIONAL MATCH (o)-[:LOCALISEE_A]->(l:Localisation)
OPTIONAL MATCH (o)-[:CORRESPOND_METIER]->(m:Métier)
RETURN o.titre_poste AS titre,
       o.employeur   AS employeur,
       o.type_contrat AS contrat,
       o.ncf_niveau_code AS ncf_requis,
       o.details_clean AS details,
       s.label AS secteur,
       l.ville AS ville,
       m.preferredLabel AS metier_esco
"""

Q_COMPETENCES_MANQUANTES = """
MATCH (o:OffreEmploi{id: $oid})-[r:REQUIERT]->(sr:Compétence)
WHERE NOT EXISTS { MATCH (c:Candidat{id: $cid})-[:POSSEDE]->(sr) }
RETURN sr.preferredLabel AS label,
       sr.description    AS description,
       sr.isDigital      AS is_digital,
       sr.isGreen        AS is_green,
       r.relationType    AS importance
ORDER BY CASE r.relationType WHEN 'essential' THEN 1 ELSE 2 END
LIMIT 10
"""

Q_NCF_CHEMIN = """
MATCH (c:Candidat {id: $cid})-[:A_NIVEAU]->(n:NiveauFormationNCF)
RETURN n.code AS ncf_code, n.intitule AS ncf_label
"""

Q_CANDIDATS_SIMILAIRES = """
MATCH (c1:Candidat {id: $cid})-[:POSSEDE]->(s:Compétence)
MATCH (c2:Candidat)-[:POSSEDE]->(s)
WHERE c2.id <> $cid
WITH c2, count(s) AS n_commun
MATCH (c2)-[p:POSTULE]->(o:OffreEmploi)
WHERE p.score_hybride >= 0.6
RETURN o.id AS offre_id, avg(p.score_hybride) AS score_moy, count(c2) AS n_sim
ORDER BY score_moy DESC LIMIT 10
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
        SET hnsw.ef_search = 100;
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
            # Skill gap
            sg = session.run(Q_SKILL_GAP,
                             cid=candidat_id, oid=offre["offre_id"]).single()
            # Compétences manquantes avec descriptions
            cm = session.run(Q_COMPETENCES_MANQUANTES,
                             cid=candidat_id, oid=offre["offre_id"]).data()
            # Chemin NCF
            ncf = session.run(Q_NCF_CHEMIN, cid=candidat_id).single()

        if sg:
            acquises  = sg.get("acquises", [])
            manquantes = sg.get("manquantes", [])
            taux = sg.get("taux", 0.0)
        else:
            acquises = manquantes = []
            taux = 0.0

        return {
            **offre,
            "acquises":    [c.get("label","") for c in (acquises or [])[:5]],
            "manquantes":  [c.get("label","") for c in (manquantes or [])[:5]],
            "ess_manq":    [c["label"] for c in (cm or []) if c.get("importance")=="essential"][:3],
            "taux_match":  round(float(taux), 3),
            "ncf_cand":    ncf.get("ncf_code") if ncf else None,
        }

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
    def _compute_hybrid_score(self, offre_enriched: dict) -> float:
        """
        Score hybride = α·sémantique + β·graphe + γ·collaboratif
        α=0.40, β=0.35, γ=0.25
        """
        s_sem  = offre_enriched.get("score_sem", 0.5)
        taux   = offre_enriched.get("taux_match", 0.5)
        n_ess  = len(offre_enriched.get("ess_manq", []))
        penalite = min(1.0, n_ess * 0.15)
        s_graph = taux * (1 - penalite)
        s_collab = offre_enriched.get("score_collab", 0.5)
        return round(0.40 * s_sem + 0.35 * s_graph + 0.25 * s_collab, 4)

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
            e["score_collab"] = 0.5   # baseline (remplacé par le vrai score collab)
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
                f"Acquises  : {', '.join(o.get('acquises', [])[:3]) or 'Aucune déclarée'}",
                f"Manquantes: {', '.join(o.get('manquantes', [])[:3]) or 'Aucune'}",
                f"Essentielles manquantes: {', '.join(o.get('ess_manq', [])) or 'Aucune'}",
            ]

        return "\n".join(lines)
