"""
dependencies.py — Dépendances FastAPI (singletons partagés)
Module 06

Gère le cycle de vie des connexions :
  - SentenceTransformer fine-tuné (chargé une seule fois au démarrage)
  - Connexion Neo4j (pool de sessions)
  - Connexion PostgreSQL/pgvector (pool de connexions)
  - Moteur de recommandation

Injection via Depends(get_engine) dans chaque route.
"""

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent

# ─── État des singletons ────────────────────────────────────────────────
_st_model          = None
_neo4j_driver      = None
_pg_conn           = None
_recommendation_engine = None


# ─────────────────────────────────────────────────────────────────────────
# SENTENCETRANSFORMER — chargé une seule fois (lifespan)
# ─────────────────────────────────────────────────────────────────────────

def init_st_model(model_path: Optional[str] = None) -> bool:
    """
    Charge le SentenceTransformer fine-tuné au démarrage de l'API.
    Pattern singleton : appel unique via lifespan, partagé entre toutes les routes.
    Retourne True si chargé depuis le modèle fine-tuné, False si fallback baseline.
    """
    global _st_model

    ft_path = model_path or str(ROOT / "models" / "st_finetuned" / "final")
    base_model = "sentence-transformers/all-MiniLM-L6-v2"

    try:
        from sentence_transformers import SentenceTransformer
        if Path(ft_path).exists() and any(Path(ft_path).iterdir()):
            _st_model = SentenceTransformer(ft_path)
            log.info(f"ST fine-tuné chargé : {ft_path}")
            return True
        else:
            log.warning(f"Fine-tuned model absent ({ft_path}), chargement baseline...")
            _st_model = SentenceTransformer(base_model)
            log.info(f"ST baseline chargé : {base_model}")
            return False
    except Exception as e:
        log.error(f"Impossible de charger le ST model : {e}")
        _st_model = None
        return False


def get_st_model():
    """Dépendance FastAPI → injecte le ST model dans les routes."""
    return _st_model


# ─────────────────────────────────────────────────────────────────────────
# NEO4J — pool de sessions
# ─────────────────────────────────────────────────────────────────────────

def init_neo4j(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
) -> bool:
    global _neo4j_driver
    try:
        from neo4j import GraphDatabase
        _neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
        _neo4j_driver.verify_connectivity()
        log.info(f"Neo4j connecté : {uri}")
        return True
    except Exception as e:
        log.warning(f"Neo4j indisponible ({e}) — mode dégradé sans graphe")
        _neo4j_driver = None
        return False


def get_neo4j():
    return _neo4j_driver


# ─────────────────────────────────────────────────────────────────────────
# POSTGRESQL / PGVECTOR
# ─────────────────────────────────────────────────────────────────────────

def init_pgvector(
    dsn: str = "postgresql://postgres:password@localhost:5432/recommandation",
) -> bool:
    global _pg_conn
    try:
        import psycopg
        _pg_conn = psycopg.connect(dsn)
        _pg_conn.autocommit = False
        log.info("pgvector connecté")
        return True
    except Exception as e:
        log.warning(f"pgvector indisponible ({e}) — mode dégradé sans ANN")
        _pg_conn = None
        return False


def get_pg():
    return _pg_conn


# ─────────────────────────────────────────────────────────────────────────
# MOTEUR DE RECOMMANDATION
# ─────────────────────────────────────────────────────────────────────────

def init_engine(llm_backend: str = "simulation") -> None:
    """Instancie le moteur GraphRAG avec les connexions disponibles."""
    global _recommendation_engine
    import sys
    sys.path.insert(0, str(ROOT / "src" / "05_graphrag"))
    from recommendation_engine import RecommendationEngine

    _recommendation_engine = RecommendationEngine(
        neo4j_driver=_neo4j_driver,
        pg_conn=_pg_conn,
        st_model=_st_model,
        llm_backend=llm_backend,
        top_k=10,
    )
    log.info(f"Moteur de recommandation initialisé (backend={llm_backend})")


def get_engine():
    """Dépendance FastAPI → injecte le moteur dans les routes."""
    if _recommendation_engine is None:
        raise RuntimeError(
            "Moteur non initialisé — vérifier le démarrage de l'API (lifespan)"
        )
    return _recommendation_engine


# ─────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────

def get_services_status() -> dict:
    """Retourne l'état de chaque service pour le health check."""
    return {
        "neo4j":    "connected" if _neo4j_driver else "unavailable",
        "pgvector": "connected" if _pg_conn else "unavailable",
        "st_model": (
            f"loaded ({_st_model.get_sentence_embedding_dimension()}d)"
            if _st_model else "unavailable"
        ),
        "llm": (
            getattr(_recommendation_engine, "llm", None)
            and getattr(_recommendation_engine.llm, "backend", "simulation")
            or "simulation"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────
# FERMETURE DES CONNEXIONS (lifespan shutdown)
# ─────────────────────────────────────────────────────────────────────────

def close_all():
    """Ferme proprement toutes les connexions lors de l'arrêt de l'API."""
    if _neo4j_driver:
        _neo4j_driver.close()
        log.info("Neo4j fermé")
    if _pg_conn and not _pg_conn.closed:
        _pg_conn.close()
        log.info("pgvector fermé")
