"""
Materialise les relations :SIMILAIRE_A dans Neo4j depuis pgvector.

Le script cherche, pour chaque entite encodee dans la table `embeddings`, ses
plus proches voisins du meme type par similarite cosinus, puis cree une relation
ponderee dans Neo4j. Les entites ESCO sont reliees via leur `conceptUri`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PG = ROOT / "src" / "04_pgvector"
SRC_NEO = ROOT / "src" / "03_knowledge_graph"
for p in (str(SRC_PG), str(SRC_NEO)):
    if p not in sys.path:
        sys.path.insert(0, p)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


@dataclass(frozen=True)
class EntityConfig:
    entity_kind: str
    label: str
    match_property: str


ENTITY_CONFIGS = {
    "skills": EntityConfig("COMPETENCE", "Compétence", "conceptUri"),
    "metiers": EntityConfig("METIER", "Métier", "conceptUri"),
}


SQL_SIMILAR_PAIRS = """
    SELECT
        e1.entity_id AS source_id,
        e2.entity_id AS target_id,
        e1.label_fr  AS source_label,
        e2.label_fr  AS target_label,
        e2.cosine_sim
    FROM embeddings e1
    CROSS JOIN LATERAL (
        SELECT
            entity_id,
            label_fr,
            1 - (embedding <=> e1.embedding) AS cosine_sim
        FROM embeddings
        WHERE entity_kind = %(entity_kind)s
          AND entity_id <> e1.entity_id
          AND embedding IS NOT NULL
        ORDER BY embedding <=> e1.embedding
        LIMIT %(top_k)s
    ) e2
    WHERE e1.entity_kind = %(entity_kind)s
      AND e1.embedding IS NOT NULL
      AND e2.cosine_sim >= %(threshold)s
"""

SQL_LIMIT_ENTITIES = """
      AND e1.entity_id IN (
          SELECT entity_id
          FROM embeddings
          WHERE entity_kind = %(entity_kind)s
          ORDER BY entity_id
          LIMIT %(limit_entities)s
      )
"""


def fetch_similar_pairs(conn, cfg: EntityConfig, threshold: float,
                        top_k: int, limit_entities: int | None) -> list[dict]:
    params = {
        "entity_kind": cfg.entity_kind,
        "threshold": threshold,
        "top_k": top_k,
    }
    sql = SQL_SIMILAR_PAIRS
    if limit_entities is not None:
        params["limit_entities"] = limit_entities
        sql += SQL_LIMIT_ENTITIES

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    best_by_pair: dict[tuple[str, str], dict] = {}
    for source_id, target_id, source_label, target_label, cosine_sim in rows:
        left, right = sorted((source_id, target_id))
        key = (left, right)
        current = best_by_pair.get(key)
        record = {
            "source_id": left,
            "target_id": right,
            "source_label": source_label,
            "target_label": target_label,
            "cosine_sim": float(cosine_sim),
        }
        if current is None or record["cosine_sim"] > current["cosine_sim"]:
            best_by_pair[key] = record

    return list(best_by_pair.values())


def upsert_similarity_relations(driver, database: str, cfg: EntityConfig,
                                pairs: list[dict], batch_size: int) -> int:
    cypher = f"""
    UNWIND $rows AS row
    MATCH (a:`{cfg.label}` {{{cfg.match_property}: row.source_id}})
    MATCH (b:`{cfg.label}` {{{cfg.match_property}: row.target_id}})
    MERGE (a)-[r:SIMILAIRE_A]-(b)
    SET r.weight = row.cosine_sim,
        r.similarity = row.cosine_sim,
        r.metric = 'cosine',
        r.source = 'pgvector',
        r.updated_at = datetime()
    RETURN count(r) AS n
    """
    total = 0
    with driver.session(database=database) as session:
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            result = session.run(cypher, rows=batch).single()
            total += int(result["n"]) if result else 0
    return total


def count_existing(driver, database: str, cfg: EntityConfig) -> int:
    cypher = f"""
    MATCH (a:`{cfg.label}`)-[r:SIMILAIRE_A]-(b:`{cfg.label}`)
    RETURN count(DISTINCT r) AS n
    """
    with driver.session(database=database) as session:
        record = session.run(cypher).single()
    return int(record["n"]) if record else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialise :SIMILAIRE_A dans Neo4j depuis pgvector."
    )
    parser.add_argument(
        "--entity",
        required=True,
        choices=sorted(ENTITY_CONFIGS),
        help="Famille d'entites a traiter.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Seuil minimal de similarite cosinus.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Nombre de voisins candidats par entite avant filtrage.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Nombre de relations envoyees par transaction Neo4j.",
    )
    parser.add_argument(
        "--limit-entities",
        type=int,
        default=None,
        help="Limite de diagnostic sur les premieres entites pgvector.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calcule les paires sans ecrire dans Neo4j.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ENTITY_CONFIGS[args.entity]

    if not (0 < args.threshold <= 1):
        raise ValueError("--threshold doit etre dans ]0, 1].")
    if args.top_k < 1:
        raise ValueError("--top-k doit etre >= 1.")

    import psycopg
    from neo4j import GraphDatabase
    from config_pgvector import PG_CONN
    from config_neo4j import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

    t0 = time.time()
    log.info("=" * 70)
    log.info("MATERIALISATION SIMILAIRE_A")
    log.info(f"  Entite      : {args.entity} ({cfg.entity_kind})")
    log.info(f"  Seuil       : {args.threshold}")
    log.info(f"  Top-k       : {args.top_k}")
    log.info(f"  Dry-run     : {args.dry_run}")
    log.info("=" * 70)

    with psycopg.connect(**PG_CONN) as conn:
        pairs = fetch_similar_pairs(
            conn,
            cfg,
            threshold=args.threshold,
            top_k=args.top_k,
            limit_entities=args.limit_entities,
        )

    log.info(f"Paires candidates uniques : {len(pairs):,}")
    if pairs:
        sims = [p["cosine_sim"] for p in pairs]
        log.info(
            "Similarite cosine : min=%.4f  moy=%.4f  max=%.4f",
            min(sims),
            sum(sims) / len(sims),
            max(sims),
        )

    if args.dry_run:
        log.info("Dry-run: aucune relation Neo4j creee.")
        return

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver:
        before = count_existing(driver, NEO4J_DATABASE, cfg)
        written = upsert_similarity_relations(
            driver,
            NEO4J_DATABASE,
            cfg,
            pairs,
            batch_size=args.batch_size,
        )
        after = count_existing(driver, NEO4J_DATABASE, cfg)

    elapsed = time.time() - t0
    log.info(f"Relations traitees : {written:,}")
    log.info(f"Relations existantes avant/apres : {before:,} -> {after:,}")
    log.info(f"Termine en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
