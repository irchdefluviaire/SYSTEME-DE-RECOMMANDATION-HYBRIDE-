"""
Audit Neo4j skill-match scores for candidate-offer pairs.

For each pair, this script recomputes the graph score directly from:
  (c:Candidat)-[:POSSEDE]->(s:Compétence)
  (o:OffreEmploi)-[:REQUIERT]->(s:Compétence)

It compares the recomputed score with the taux_match stored in evaluation
outputs and exports both row-level and aggregate diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
SRC_08 = ROOT / "src" / "08_agentic_graphrag"
if str(SRC_08) not in sys.path:
    sys.path.insert(0, str(SRC_08))

load_dotenv(ROOT / ".env")

from tools import get_neo4j_driver, get_pg_conn  # noqa: E402


DEFAULT_INPUT = ROOT / "outputs" / "evaluation" / "pgvector_vs_graph" / "ablation_graph_rerank_details.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "evaluation" / "neo4j_match_score_audit"


Q_AUDIT_PAIR = """
OPTIONAL MATCH (c:Candidat {id: $cid})
OPTIONAL MATCH (o:OffreEmploi {id: $oid})
WITH c, o
OPTIONAL MATCH (c)-[:POSSEDE]->(sc:Compétence)
OPTIONAL MATCH (o)-[rq:REQUIERT]->(sr:Compétence)
WITH
  c IS NOT NULL AS candidat_found,
  o IS NOT NULL AS offre_found,
  collect(DISTINCT sc.conceptUri) AS cand_uris,
  collect(DISTINCT {
    uri: sr.conceptUri,
    label: sr.preferredLabel,
    relation_type: coalesce(rq.relationType, 'essential'),
    confidence: coalesce(rq.confidence, 0.0)
  }) AS offre_skills
WITH candidat_found, offre_found, cand_uris, [x IN offre_skills WHERE x.uri IS NOT NULL] AS offre_skills
WITH
  candidat_found,
  offre_found,
  cand_uris,
  offre_skills,
  [x IN offre_skills WHERE x.uri IN cand_uris] AS acquired,
  [x IN offre_skills WHERE NOT x.uri IN cand_uris] AS missing,
  [x IN offre_skills WHERE NOT x.uri IN cand_uris AND x.relation_type = 'essential'] AS essential_missing
RETURN
  candidat_found,
  offre_found,
  size(cand_uris) AS n_candidat_skills,
  size(offre_skills) AS n_offre_skills,
  size(acquired) AS n_acquises,
  size(missing) AS n_manquantes,
  size(essential_missing) AS n_essentielles_manquantes,
  CASE WHEN size(offre_skills) > 0
       THEN toFloat(size(acquired)) / size(offre_skills)
       ELSE 0.0
  END AS taux_match_neo4j,
  [x IN acquired | x.label][0..8] AS competences_acquises,
  [x IN missing | x.label][0..8] AS competences_manquantes,
  [x IN essential_missing | x.label][0..8] AS essentielles_manquantes
"""


Q_GLOBAL_COUNTS = """
RETURN
  EXISTS { MATCH (:Candidat)-[:POSSEDE]->(:Compétence) } AS has_possede,
  EXISTS { MATCH (:OffreEmploi)-[:REQUIERT]->(:Compétence) } AS has_requiert,
  COUNT { (:Candidat)-[:POSSEDE]->(:Compétence) } AS n_possede,
  COUNT { (:OffreEmploi)-[:REQUIERT]->(:Compétence) } AS n_requiert
"""

Q_ID_COVERAGE = """
CALL () {
  MATCH (c:Candidat)
  WHERE c.id IN $candidate_ids
  RETURN
    count(DISTINCT c) AS n_candidates_found,
    count(DISTINCT CASE WHEN EXISTS { (c)-[:POSSEDE]->(:Compétence) } THEN c END) AS n_candidates_with_possede
}
CALL () {
  MATCH (o:OffreEmploi)
  WHERE o.id IN $offer_ids
  RETURN
    count(DISTINCT o) AS n_offers_found,
    count(DISTINCT CASE WHEN EXISTS { (o)-[:REQUIERT]->(:Compétence) } THEN o END) AS n_offers_with_requiert
}
RETURN
  size($candidate_ids) AS n_candidate_ids_requested,
  n_candidates_found,
  n_candidates_with_possede,
  size($offer_ids) AS n_offer_ids_requested,
  n_offers_found,
  n_offers_with_requiert
"""

Q_SAMPLE_OFFER_IDS = """
MATCH (o:OffreEmploi)
WHERE o.id IN $offer_ids
RETURN o.id AS offre_id,
       o.titre_poste AS titre,
       COUNT { (o)-[:REQUIERT]->(:Compétence) } AS n_requiert
LIMIT 10
"""


def audit(input_path: Path, output_dir: Path, limit: int | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    if limit:
        df = df.head(limit).copy()

    required = {"candidat_id", "offre_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    driver = get_neo4j_driver()
    pg_conn = get_pg_conn()
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    rows: list[dict[str, Any]] = []
    offer_ids = sorted(df["offre_id"].astype(str).unique().tolist())
    candidate_ids = sorted(df["candidat_id"].astype(str).unique().tolist())

    with pg_conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              count(DISTINCT entity_id) FILTER (WHERE entity_kind = 'OFFRE_EMPLOI') AS n_offer_entity_ids,
              count(DISTINCT neo4j_node_id) FILTER (
                WHERE entity_kind = 'OFFRE_EMPLOI' AND neo4j_node_id IS NOT NULL
              ) AS n_offer_neo4j_node_ids,
              count(DISTINCT entity_id) FILTER (
                WHERE entity_kind = 'OFFRE_EMPLOI' AND entity_id = ANY(%s)
              ) AS n_eval_offer_ids_as_entity_id,
              count(DISTINCT neo4j_node_id) FILTER (
                WHERE entity_kind = 'OFFRE_EMPLOI' AND neo4j_node_id = ANY(%s)
              ) AS n_eval_offer_ids_as_neo4j_node_id
            FROM embeddings
            """,
            (offer_ids, offer_ids),
        )
        pgvector_coverage = dict(zip([desc[0] for desc in cur.description], cur.fetchone()))

        cur.execute(
            """
            SELECT entity_id, neo4j_node_id, label_fr
            FROM embeddings
            WHERE entity_kind = 'OFFRE_EMPLOI'
              AND (entity_id = ANY(%s) OR neo4j_node_id = ANY(%s))
            LIMIT 10
            """,
            (offer_ids, offer_ids),
        )
        pgvector_offer_hits = [
            {"entity_id": row[0], "neo4j_node_id": row[1], "label_fr": row[2]}
            for row in cur.fetchall()
        ]

        cur.execute(
            """
            SELECT entity_kind::text AS entity_kind, count(*) AS n
            FROM embeddings
            GROUP BY entity_kind
            ORDER BY entity_kind
            """
        )
        pgvector_counts = [{"entity_kind": row[0], "n": row[1]} for row in cur.fetchall()]

    with driver.session(database=database) as session:
        global_counts = dict(session.run(Q_GLOBAL_COUNTS).single())
        id_coverage = dict(
            session.run(
                Q_ID_COVERAGE,
                candidate_ids=candidate_ids,
                offer_ids=offer_ids,
            ).single()
        )
        sample_offer_hits = [
            dict(row)
            for row in session.run(
                Q_SAMPLE_OFFER_IDS,
                offer_ids=offer_ids[:20],
            )
        ]
        for _, row in df.iterrows():
            cid = str(row["candidat_id"])
            oid = str(row["offre_id"])
            rec = session.run(Q_AUDIT_PAIR, cid=cid, oid=oid).single()
            out = {
                "candidat_id": cid,
                "offre_id": oid,
                "titre": row.get("titre"),
                "taux_match_stocke": row.get("taux_match"),
            }
            if rec is None:
                out.update(
                    {
                        "pair_found": False,
                        "candidat_found": False,
                        "offre_found": False,
                        "n_candidat_skills": 0,
                        "n_offre_skills": 0,
                        "n_acquises": 0,
                        "n_manquantes": 0,
                        "n_essentielles_manquantes": 0,
                        "taux_match_neo4j": None,
                        "delta": None,
                    }
                )
            else:
                data = dict(rec)
                stored = pd.to_numeric(pd.Series([row.get("taux_match")]), errors="coerce").iloc[0]
                recomputed = data.get("taux_match_neo4j")
                delta = None
                if pd.notna(stored) and recomputed is not None:
                    delta = float(recomputed) - float(stored)
                out.update(
                    {
                        "pair_found": bool(data.get("candidat_found")) and bool(data.get("offre_found")),
                        **data,
                        "taux_match_neo4j": None if recomputed is None else round(float(recomputed), 6),
                        "delta": None if delta is None else round(float(delta), 6),
                    }
                )
            rows.append(out)

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(output_dir / "neo4j_match_score_audit_details.csv", index=False, encoding="utf-8")

    comparable = audit_df[audit_df["taux_match_neo4j"].notna()].copy()
    if "delta" in comparable.columns:
        comparable["abs_delta"] = comparable["delta"].abs()

    summary = {
        "input_path": str(input_path.relative_to(ROOT)),
        "n_pairs": int(len(audit_df)),
        "global_counts": global_counts,
        "pgvector_counts": pgvector_counts,
        "pgvector_coverage_for_eval_offer_ids": pgvector_coverage,
        "pgvector_offer_hits": pgvector_offer_hits,
        "id_coverage": id_coverage,
        "sample_offer_hits": sample_offer_hits,
        "pairs_found": int(audit_df["pair_found"].sum()),
        "pairs_with_offre_skills": int((audit_df["n_offre_skills"] > 0).sum()),
        "pairs_with_candidat_skills": int((audit_df["n_candidat_skills"] > 0).sum()),
        "pairs_with_common_skills": int((audit_df["n_acquises"] > 0).sum()),
        "mean_offre_skills": round(float(audit_df["n_offre_skills"].mean()), 4),
        "mean_candidat_skills": round(float(audit_df["n_candidat_skills"].mean()), 4),
        "mean_common_skills": round(float(audit_df["n_acquises"].mean()), 4),
        "mean_taux_match_neo4j": round(float(comparable["taux_match_neo4j"].mean()), 6) if len(comparable) else None,
        "mean_taux_match_stocke": round(float(pd.to_numeric(comparable["taux_match_stocke"], errors="coerce").mean()), 6)
        if len(comparable)
        else None,
        "max_abs_delta": round(float(comparable["abs_delta"].max()), 6) if len(comparable) else None,
        "mean_abs_delta": round(float(comparable["abs_delta"].mean()), 6) if len(comparable) else None,
        "n_delta_gt_001": int((comparable["abs_delta"] > 0.01).sum()) if len(comparable) else 0,
        "outputs": {
            "details_csv": str((output_dir / "neo4j_match_score_audit_details.csv").relative_to(ROOT)),
            "summary_json": str((output_dir / "neo4j_match_score_audit_summary.json").relative_to(ROOT)),
        },
    }
    (output_dir / "neo4j_match_score_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Neo4j taux_match for candidate-offer pairs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit(args.input, args.output_dir, args.limit)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
