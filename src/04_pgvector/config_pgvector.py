"""config_pgvector.py — Configuration Module 04"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

PG_HOST     = os.getenv("PG_HOST", "localhost")
PG_PORT     = int(os.getenv("PG_PORT", "5432"))
PG_DB       = os.getenv("PG_DB", "test_kmer")
PG_USER     = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_DSN      = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
PG_CONN     = {
    "host": PG_HOST,
    "port": PG_PORT,
    "dbname": PG_DB,
    "user": PG_USER,
    "password": PG_PASSWORD,
}

DATA_PROC  = ROOT / "data" / "processed"
ESCO_DIR   = ROOT / "data" / "raw" / "esco"
MODEL_PATH = ROOT / "models" / "st_finetuned" / "final"
MODEL_BASE = "sentence-transformers/all-MiniLM-L6-v2"

OFFRES_PARQUET    = DATA_PROC / "offres_normalized.parquet"
CANDIDATS_PARQUET = DATA_PROC / "candidats_normalized.parquet"
MEPC_BASE_PARQUET = DATA_PROC / "mepc_groupes_base.parquet"
NCF_DET_PARQUET   = DATA_PROC / "ncf_dom_detailles.parquet"
ESCO_SKILLS_CSV   = ESCO_DIR  / "skills_fr.csv"
ESCO_OCC_CSV      = ESCO_DIR  / "occupations_fr.csv"
EMBEDDINGS_DIR    = DATA_PROC / "embeddings"

DIM_EMBEDDING       = 384
NORMALIZE           = True
BATCH_SIZE_ENCODE   = 128
MODEL_ID            = "all-MiniLM-L6-v2-ft-offres-cm"

HNSW_M               = 16
HNSW_EF_CONSTRUCTION = 64

ENTITY_KINDS = [
    "OFFRE_EMPLOI", "CANDIDAT", "COMPETENCE",
    "METIER", "GROUPE_BASE_MEPC", "DOMAINE_DETAILLE_NCF",
]
