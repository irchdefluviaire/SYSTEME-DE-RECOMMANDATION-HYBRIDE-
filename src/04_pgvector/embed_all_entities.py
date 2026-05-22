"""
embed_all_entities.py
===========================================================================
Module 04 — Encodage vectoriel de toutes les entités du système

Encode en vecteurs 384d toutes les entités du graphe de connaissances
avec le SentenceTransformer fine-tuné (Module 02) et insère dans pgvector.

Entités encodées (6 familles) :
  1. Offres d'emploi        (7 861 — côté corpus, text_to_embed)
  2. Candidats              (1 105 — côté requête, metadata_str)
  3. Compétences ESCO       (13 939 — label + altLabels + description)
  4. Métiers ESCO           (3 039 — label + altLabels + description)
  5. Domaines NCF détaillés (201 — intitule + explication)
  6. Groupes de base MEPC   (209 — intitule + notes_explicatives)

Usage :
  python embed_all_entities.py                   # tout encoder
  python embed_all_entities.py --entity offres   # une entité
  python embed_all_entities.py --dry-run         # test sans DB
  python embed_all_entities.py --batch-size 64
  python embed_all_entities.py --model ./models/st_finetuned/final

Dépendances :
  pip install sentence-transformers psycopg2-binary pandas pyarrow
===========================================================================
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Chemins ────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent.parent
PROC      = ROOT / "data" / "processed"
ESCO_DIR  = Path("/mnt/user-data/uploads")
MODEL_DIR = ROOT / "models" / "st_finetuned" / "final"
MODEL_FALLBACK = "sentence-transformers/all-MiniLM-L6-v2"

# ── Config PostgreSQL (adapter selon installation) ─────────────────────────
PG_CONN = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "recommandation",
    "user":     "postgres",
    "password": "postgres",
}

# ── Paramètres d'encodage ──────────────────────────────────────────────────
DEFAULT_BATCH_SIZE    = 64
DEFAULT_MODEL_PATH    = str(MODEL_DIR)
NORMALIZE_EMBEDDINGS  = True   # vecteurs unitaires → cosine = dot product
MAX_TEXT_CHARS        = 512    # troncature avant encodage


# ─────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """
    Charge le SentenceTransformer fine-tuné (Module 02).
    Fallback sur le modèle de base si le fine-tuné n'est pas disponible.
    """
    from sentence_transformers import SentenceTransformer

    path = Path(model_path)
    if path.exists():
        log.info(f"Chargement modèle fine-tuné : {model_path}")
    else:
        log.warning(f"Modèle fine-tuné absent ({model_path})")
        log.warning(f"  → Fallback sur : {MODEL_FALLBACK}")
        model_path = MODEL_FALLBACK

    model = SentenceTransformer(model_path)
    dim   = model.get_sentence_embedding_dimension()
    n_par = sum(p.numel() for p in model.parameters())
    log.info(f"  Dimension   : {dim}d")
    log.info(f"  Paramètres  : {n_par:,}")
    return model


# ─────────────────────────────────────────────────────────────────────────
# CONNEXION POSTGRESQL
# ─────────────────────────────────────────────────────────────────────────

def get_pg_conn():
    import psycopg2
    conn = psycopg2.connect(**PG_CONN)
    log.info("Connexion PostgreSQL OK")
    return conn


def create_schema(conn):
    """Crée le schéma pgvector si non existant."""
    sql_path = Path(__file__).parent / "schema_pgvector.sql"
    with open(sql_path, encoding="utf-8") as f:
        sql = f.read()
    with conn.cursor() as cur:
        # Exécuter statement par statement (ignorer commentaires)
        statements = [s.strip() for s in sql.split(";")
                      if s.strip() and not s.strip().startswith("--")]
        for stmt in statements:
            try:
                cur.execute(stmt)
            except Exception as e:
                conn.rollback()
                if "already exists" not in str(e).lower():
                    log.warning(f"Schema warning: {e}")
    conn.commit()
    log.info("Schéma pgvector créé/vérifié")


# ─────────────────────────────────────────────────────────────────────────
# FONCTIONS D'ENCODAGE PAR ENTITÉ
# ─────────────────────────────────────────────────────────────────────────

def encode_batch(model, texts: list[str], batch_size: int) -> np.ndarray:
    """Encode une liste de textes en vecteurs, avec barre de progression."""
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def build_insert_rows(
    entity_kind: str,
    entity_ids:  list[str],
    labels:      list[str],
    texts:       list[str],
    embeddings:  np.ndarray,
    source:      str,
    neo4j_ids:   Optional[list[str]] = None,
) -> list[tuple]:
    """Construit les tuples d'insertion pour la table embeddings."""
    rows = []
    for i, (eid, lbl, txt, emb) in enumerate(zip(entity_ids, labels, texts, embeddings)):
        neo4j_id = neo4j_ids[i] if neo4j_ids else eid
        rows.append((
            entity_kind,
            eid,
            source,
            lbl[:200] if lbl else "",
            txt[:MAX_TEXT_CHARS],
            emb.tolist(),    # VECTOR(384) → liste Python
            "all-MiniLM-L6-v2-ft-offres-cm",
            neo4j_id,
        ))
    return rows


def upsert_embeddings(conn, rows: list[tuple]) -> int:
    """
    INSERT ... ON CONFLICT DO UPDATE (upsert idempotent).
    Clé de conflit : (entity_kind, entity_id, model_id).
    """
    sql = """
        INSERT INTO embeddings
            (entity_kind, entity_id, source_system, label_fr,
             text_to_embed, embedding, model_id, neo4j_node_id)
        VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s)
        ON CONFLICT (entity_kind, entity_id, model_id)
        DO UPDATE SET
            label_fr      = EXCLUDED.label_fr,
            text_to_embed = EXCLUDED.text_to_embed,
            embedding     = EXCLUDED.embedding,
            neo4j_node_id = EXCLUDED.neo4j_node_id,
            updated_at    = NOW()
    """
    with conn.cursor() as cur:
        # Convertir les embeddings en format pgvector
        pg_rows = []
        for row in rows:
            entity_kind, entity_id, source, label, text, emb_list, model_id, neo4j_id = row
            # Format pgvector : '[0.1,0.2,...]'
            emb_str = "[" + ",".join(f"{v:.6f}" for v in emb_list) + "]"
            pg_rows.append((entity_kind, entity_id, source, label, text,
                            emb_str, model_id, neo4j_id))
        cur.executemany(sql, pg_rows)
    conn.commit()
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────
# ENCODEURS PAR TYPE D'ENTITÉ
# ─────────────────────────────────────────────────────────────────────────

def embed_offres(model, conn, batch_size: int, dry_run: bool) -> int:
    """Encode les 7 861 offres (côté corpus — description)."""
    log.info("[1/6] Encodage des Offres d'emploi...")
    df = pd.read_parquet(PROC / "offres_normalized.parquet")

    entity_ids = df["offre_id"].tolist()
    labels     = df["titre_poste"].fillna("").tolist()
    texts      = df["text_to_embed"].fillna("").str[:MAX_TEXT_CHARS].tolist()

    log.info(f"  {len(texts):,} offres | moy={np.mean([len(t) for t in texts]):.0f} chars")
    embeddings = encode_batch(model, texts, batch_size)

    if not dry_run:
        rows = build_insert_rows("OFFRE_EMPLOI", entity_ids, labels, texts,
                                  embeddings, "OFFRES")
        n = upsert_embeddings(conn, rows)
        log.info(f"  → {n:,} vecteurs insérés/mis à jour dans pgvector")
    else:
        log.info(f"  [DRY-RUN] {len(embeddings)} vecteurs {embeddings.shape[1]}d calculés")
    return len(texts)


def embed_candidats(model, conn, batch_size: int, dry_run: bool) -> int:
    """Encode les 1 105 candidats (côté requête — metadata structurées)."""
    log.info("[2/6] Encodage des Candidats...")
    df = pd.read_parquet(PROC / "candidats_normalized.parquet")

    entity_ids = df["candidat_id"].astype(str).tolist()
    labels     = df["metier_vise"].fillna("").tolist()
    texts      = df["text_to_embed"].fillna("").str[:MAX_TEXT_CHARS].tolist()

    log.info(f"  {len(texts):,} candidats | moy={np.mean([len(t) for t in texts]):.0f} chars")
    embeddings = encode_batch(model, texts, batch_size)

    if not dry_run:
        rows = build_insert_rows("CANDIDAT", entity_ids, labels, texts,
                                  embeddings, "CANDIDATS")
        n = upsert_embeddings(conn, rows)
        log.info(f"  → {n:,} vecteurs insérés")
    return len(texts)


def embed_esco_skills(model, conn, batch_size: int, dry_run: bool) -> int:
    """Encode les 13 939 compétences ESCO."""
    log.info("[3/6] Encodage des Compétences ESCO...")

    entity_ids, labels, texts = [], [], []
    with open(ESCO_DIR / "skills_fr.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uri  = row.get("conceptUri", "")
            lbl  = row.get("preferredLabel", "")
            alt  = row.get("altLabels", "").replace("\n", " ")[:100]
            desc = row.get("description", "")[:300]
            text = f"{lbl} {alt}. {desc}".strip()
            entity_ids.append(uri)
            labels.append(lbl)
            texts.append(text[:MAX_TEXT_CHARS])

    log.info(f"  {len(texts):,} compétences | moy={np.mean([len(t) for t in texts]):.0f} chars")
    embeddings = encode_batch(model, texts, batch_size)

    if not dry_run:
        rows = build_insert_rows("COMPETENCE", entity_ids, labels, texts,
                                  embeddings, "ESCO")
        n = upsert_embeddings(conn, rows)
        log.info(f"  → {n:,} vecteurs insérés")
    return len(texts)


def embed_esco_occupations(model, conn, batch_size: int, dry_run: bool) -> int:
    """Encode les 3 039 métiers ESCO."""
    log.info("[4/6] Encodage des Métiers ESCO...")

    entity_ids, labels, texts = [], [], []
    with open(ESCO_DIR / "occupations_fr.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uri  = row.get("conceptUri", "")
            lbl  = row.get("preferredLabel", "")
            alt  = row.get("altLabels", "").replace("\n", " ")[:100]
            desc = row.get("description", "")[:300]
            text = f"{lbl} {alt}. {desc}".strip()
            entity_ids.append(uri)
            labels.append(lbl)
            texts.append(text[:MAX_TEXT_CHARS])

    log.info(f"  {len(texts):,} métiers | moy={np.mean([len(t) for t in texts]):.0f} chars")
    embeddings = encode_batch(model, texts, batch_size)

    if not dry_run:
        rows = build_insert_rows("METIER", entity_ids, labels, texts,
                                  embeddings, "ESCO")
        n = upsert_embeddings(conn, rows)
        log.info(f"  → {n:,} vecteurs insérés")
    return len(texts)


def embed_ncf_detailles(model, conn, batch_size: int, dry_run: bool) -> int:
    """Encode les 201 domaines détaillés NCF."""
    log.info("[5/6] Encodage des Domaines Détaillés NCF...")
    df = pd.read_parquet(PROC / "ncf_dom_detailles.parquet")

    entity_ids = df["code"].astype(str).tolist()
    labels     = df["intitule"].fillna("").tolist()
    texts      = df["text_to_embed"].fillna("").str[:MAX_TEXT_CHARS].tolist()

    log.info(f"  {len(texts):,} domaines NCF | moy={np.mean([len(t) for t in texts]):.0f} chars")
    embeddings = encode_batch(model, texts, batch_size)

    if not dry_run:
        rows = build_insert_rows("DOMAINE_DETAILLE_NCF", entity_ids, labels, texts,
                                  embeddings, "NCF")
        n = upsert_embeddings(conn, rows)
        log.info(f"  → {n:,} vecteurs insérés")
    return len(texts)


def embed_mepc_base(model, conn, batch_size: int, dry_run: bool) -> int:
    """Encode les 209 groupes de base MEPC."""
    log.info("[6/6] Encodage des Groupes de Base MEPC...")
    df = pd.read_parquet(PROC / "mepc_groupes_base.parquet")

    entity_ids = df["code"].astype(str).tolist()
    labels     = df["intitule"].fillna("").tolist()
    texts      = df["text_to_embed"].fillna("").str[:MAX_TEXT_CHARS].tolist()

    log.info(f"  {len(texts):,} groupes MEPC | moy={np.mean([len(t) for t in texts]):.0f} chars")
    embeddings = encode_batch(model, texts, batch_size)

    if not dry_run:
        rows = build_insert_rows("GROUPE_BASE_MEPC", entity_ids, labels, texts,
                                  embeddings, "MEPC")
        n = upsert_embeddings(conn, rows)
        log.info(f"  → {n:,} vecteurs insérés")
    return len(texts)


# ─────────────────────────────────────────────────────────────────────────
# RECHERCHE ANN (cosine similarity)
# ─────────────────────────────────────────────────────────────────────────

def ann_search(
    conn,
    model,
    query_text:    str,
    entity_kind:   str = "OFFRE_EMPLOI",
    top_k:         int = 20,
) -> list[dict]:
    """
    Recherche ANN : encode le texte requête et retourne les top-k entités
    les plus proches par cosine similarity depuis pgvector.
    """
    # Encoder la requête
    query_vec = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    emb_str = "[" + ",".join(f"{v:.6f}" for v in query_vec) + "]"

    sql = """
        SELECT
            entity_id,
            label_fr,
            1 - (embedding <=> %s::vector) AS cosine_sim
        FROM   embeddings
        WHERE  entity_kind = %s
        ORDER  BY embedding <=> %s::vector
        LIMIT  %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (emb_str, entity_kind, emb_str, top_k))
        rows = cur.fetchall()

    return [
        {"entity_id": r[0], "label": r[1], "cosine_sim": round(r[2], 4)}
        for r in rows
    ]


def ann_search_filtered(
    conn,
    model,
    candidat_text: str,
    top_k:         int = 20,
) -> list[dict]:
    """
    Recherche ANN candidat → offres (requête cross-entité optimisée).
    Utilise l'index HNSW sur OFFRE_EMPLOI uniquement.
    """
    return ann_search(conn, model, candidat_text, "OFFRE_EMPLOI", top_k)


# ─────────────────────────────────────────────────────────────────────────
# VALIDATION DU CONTENU PGVECTOR
# ─────────────────────────────────────────────────────────────────────────

def validate_pgvector(conn) -> dict:
    """Retourne les statistiques de la table embeddings."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT entity_kind, COUNT(*) AS n,
                   AVG(LENGTH(text_to_embed))::INT AS moy_chars
            FROM   embeddings
            GROUP  BY entity_kind
            ORDER  BY entity_kind
        """)
        rows = cur.fetchall()

    stats = {}
    log.info("\n" + "="*55)
    log.info("VALIDATION PGVECTOR")
    log.info("="*55)
    log.info(f"{'Type':<30} {'N vecteurs':>12} {'Moy chars':>10}")
    log.info("-"*55)
    for kind, n, moy in rows:
        log.info(f"  {kind:<28} {n:>12,} {moy:>10}")
        stats[kind] = {"n": n, "moy_chars": moy}

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM embeddings")
        total = cur.fetchone()[0]
    log.info(f"\n  TOTAL : {total:,} vecteurs 384d")
    return stats


# ─────────────────────────────────────────────────────────────────────────
# ORCHESTRATEUR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────

ENTITY_FUNCS = {
    "offres":     embed_offres,
    "candidats":  embed_candidats,
    "skills":     embed_esco_skills,
    "metiers":    embed_esco_occupations,
    "ncf":        embed_ncf_detailles,
    "mepc":       embed_mepc_base,
}


def run(
    entity:     Optional[str] = None,
    batch_size: int  = DEFAULT_BATCH_SIZE,
    model_path: str  = DEFAULT_MODEL_PATH,
    dry_run:    bool = False,
):
    log.info("=" * 65)
    log.info("MODULE 04 — ENCODAGE VECTORIEL (pgvector)")
    log.info("=" * 65)
    log.info(f"  Modèle     : {model_path}")
    log.info(f"  Batch size : {batch_size}")
    log.info(f"  Dry-run    : {dry_run}")

    model = load_model(model_path)
    conn  = None if dry_run else get_pg_conn()

    if not dry_run:
        create_schema(conn)

    t0    = time.time()
    total = 0

    entities = [entity] if entity else list(ENTITY_FUNCS.keys())
    for ent in entities:
        fn = ENTITY_FUNCS.get(ent)
        if not fn:
            log.error(f"Entité inconnue : {ent}")
            continue
        t1 = time.time()
        n  = fn(model, conn, batch_size, dry_run)
        total += n
        log.info(f"  ✓ {ent:<12} : {n:,} entités en {time.time()-t1:.1f}s")

    if not dry_run and conn:
        stats = validate_pgvector(conn)
        conn.close()

    elapsed = time.time() - t0
    log.info(f"\n✓ Module 04 terminé — {total:,} entités en {elapsed:.1f}s")
    log.info(f"  Débit moyen : {total/elapsed:.0f} entités/s")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Module 04 — Encodage vectoriel pgvector"
    )
    parser.add_argument("--entity", choices=list(ENTITY_FUNCS.keys()),
                        help="Encoder une seule entité")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model",  type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="Calculer les embeddings sans insérer en base")
    args = parser.parse_args()
    run(entity=args.entity, batch_size=args.batch_size,
        model_path=args.model, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
