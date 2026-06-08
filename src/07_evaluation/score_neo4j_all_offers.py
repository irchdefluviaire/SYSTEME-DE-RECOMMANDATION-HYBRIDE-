"""
Compute and audit Neo4j skill-match scores for all normalized job offers.

The graph score is candidate-dependent:

    taux_match = |skills required by offer and owned by candidate|
                 / |skills required by offer|

This script is intentionally separate from pgvector retrieval. It checks that
every offer in data/processed/offres_normalized.parquet can be resolved in
Neo4j and that the exact skill-overlap score can be computed without falling
back to heuristic metadata.
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

from tools import get_neo4j_driver  # noqa: E402


OFFRES_PARQUET = ROOT / "data" / "processed" / "offres_normalized.parquet"
CANDIDATS_PARQUET = ROOT / "data" / "processed" / "candidats_normalized.parquet"
OUT_DIR = ROOT / "outputs" / "evaluation" / "neo4j_all_offer_scores"


Q_SCORE_OFFERS = """
MATCH (c:Candidat {id: $candidate_id})
OPTIONAL MATCH (c)-[:POSSEDE]->(cs:Compétence)
WITH c, collect(DISTINCT cs.conceptUri) AS cand_uris
UNWIND $offer_ids AS offer_id
OPTIONAL MATCH (o:OffreEmploi {id: offer_id})
OPTIONAL MATCH (o)-[rq:REQUIERT]->(os:Compétence)
WITH
  offer_id,
  o,
  cand_uris,
  collect(DISTINCT {
    uri: os.conceptUri,
    label: os.preferredLabel,
    relation_type: coalesce(rq.relationType, 'essential'),
    confidence: coalesce(rq.confidence, 0.0)
  }) AS raw_offer_skills
WITH
  offer_id,
  o IS NOT NULL AS offre_found,
  cand_uris,
  [x IN raw_offer_skills WHERE x.uri IS NOT NULL] AS offer_skills
WITH
  offer_id,
  offre_found,
  size(cand_uris) AS n_candidat_skills,
  size(offer_skills) AS n_offre_skills,
  [x IN offer_skills WHERE x.uri IN cand_uris] AS acquired,
  [x IN offer_skills WHERE NOT x.uri IN cand_uris] AS missing,
  [x IN offer_skills WHERE NOT x.uri IN cand_uris AND x.relation_type = 'essential'] AS essential_missing
RETURN
  offer_id AS offre_id,
  offre_found,
  n_candidat_skills,
  n_offre_skills,
  size(acquired) AS n_acquises,
  size(missing) AS n_manquantes,
  size(essential_missing) AS n_essentielles_manquantes,
  CASE WHEN n_offre_skills > 0 THEN toFloat(size(acquired)) / n_offre_skills ELSE 0.0 END AS taux_match_neo4j,
  [x IN acquired | x.label][0..6] AS competences_acquises,
  [x IN missing | x.label][0..6] AS competences_manquantes
"""


Q_GRAPH_OFFER_COUNTS = """
CALL () {
  MATCH (o:OffreEmploi)
  RETURN count(o) AS n_offres_neo4j
}
CALL () {
  MATCH (:OffreEmploi)-[r:REQUIERT]->(:Compétence)
  RETURN count(r) AS n_requiert_relations
}
CALL () {
  MATCH (o:OffreEmploi)
  WHERE EXISTS { (o)-[:REQUIERT]->(:Compétence) }
  RETURN count(o) AS n_offres_with_requiert
}
RETURN n_offres_neo4j, n_requiert_relations, n_offres_with_requiert
"""


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _default_candidate_id() -> str:
    df = pd.read_parquet(CANDIDATS_PARQUET, columns=["candidat_id"])
    if df.empty:
        raise ValueError("Aucun candidat dans candidats_normalized.parquet")
    return str(df["candidat_id"].iloc[0])


def compute_scores(candidate_id: str, batch_size: int = 1000) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    offres = pd.read_parquet(OFFRES_PARQUET, columns=["offre_id", "titre_poste"])
    offres["offre_id"] = offres["offre_id"].astype(str)
    offer_ids = offres["offre_id"].dropna().astype(str).unique().tolist()

    driver = get_neo4j_driver()
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    rows: list[dict[str, Any]] = []

    with driver.session(database=database) as session:
        graph_counts = dict(session.run(Q_GRAPH_OFFER_COUNTS).single())
        for batch in _chunks(offer_ids, batch_size):
            rows.extend(dict(row) for row in session.run(Q_SCORE_OFFERS, candidate_id=candidate_id, offer_ids=batch))

    scores = pd.DataFrame(rows)
    scores = scores.merge(offres.drop_duplicates("offre_id"), on="offre_id", how="left")
    scores["taux_match_neo4j"] = pd.to_numeric(scores["taux_match_neo4j"], errors="coerce")

    out_csv = OUT_DIR / f"neo4j_scores_all_offers_{candidate_id}.csv"
    scores.to_csv(out_csv, index=False, encoding="utf-8")

    found = scores["offre_found"].astype(bool)
    with_skills = scores["n_offre_skills"].fillna(0).astype(int) > 0
    comparable = scores[scores["taux_match_neo4j"].notna()].copy()

    summary = {
        "candidate_id": candidate_id,
        "n_offres_data": int(len(offer_ids)),
        "graph_counts": graph_counts,
        "n_offres_found_in_neo4j": int(found.sum()),
        "n_offres_missing_in_neo4j": int((~found).sum()),
        "n_offres_with_requiert": int((found & with_skills).sum()),
        "n_offres_without_requiert": int((found & ~with_skills).sum()),
        "coverage_offres_found_pct": round(float(found.mean() * 100), 3) if len(scores) else 0.0,
        "coverage_requiert_pct": round(float((found & with_skills).sum() / max(found.sum(), 1) * 100), 3),
        "mean_required_skills_per_offer": round(float(scores.loc[found, "n_offre_skills"].mean()), 4) if found.any() else None,
        "median_required_skills_per_offer": round(float(scores.loc[found, "n_offre_skills"].median()), 4) if found.any() else None,
        "mean_taux_match_neo4j": round(float(comparable["taux_match_neo4j"].mean()), 6) if len(comparable) else None,
        "median_taux_match_neo4j": round(float(comparable["taux_match_neo4j"].median()), 6) if len(comparable) else None,
        "n_offres_with_positive_match": int((scores["n_acquises"].fillna(0).astype(int) > 0).sum()),
        "output_csv": str(out_csv.relative_to(ROOT)),
    }

    out_json = OUT_DIR / f"neo4j_scores_all_offers_{candidate_id}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["output_summary"] = str(out_json.relative_to(ROOT))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Neo4j scores for all normalized offers.")
    parser.add_argument("--candidat-id", default=None, help="Candidate id used to compute candidate-offer scores.")
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_id = str(args.candidat_id or _default_candidate_id())
    summary = compute_scores(candidate_id, batch_size=args.batch_size)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
