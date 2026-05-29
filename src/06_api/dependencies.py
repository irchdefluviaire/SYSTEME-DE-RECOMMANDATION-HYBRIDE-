"""
dependencies.py â€” DÃ©pendances FastAPI (singletons partagÃ©s)
Module 06

GÃ¨re le cycle de vie des connexions :
  - SentenceTransformer fine-tunÃ© (chargÃ© une seule fois au dÃ©marrage)
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

# â”€â”€â”€ Ã‰tat des singletons â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_st_model          = None
_neo4j_driver      = None
_pg_conn           = None
_recommendation_engine = None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SENTENCETRANSFORMER â€” chargÃ© une seule fois (lifespan)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def init_st_model(model_path: Optional[str] = None) -> bool:
    """
    Charge le SentenceTransformer fine-tunÃ© au dÃ©marrage de l'API.
    Pattern singleton : appel unique via lifespan, partagÃ© entre toutes les routes.
    Retourne True si chargÃ© depuis le modÃ¨le fine-tunÃ©, False si fallback baseline.
    """
    global _st_model

    ft_path = model_path or str(ROOT / "models" / "st_finetuned" / "final")
    base_model = "sentence-transformers/all-MiniLM-L6-v2"

    try:
        from sentence_transformers import SentenceTransformer
        if Path(ft_path).exists() and any(Path(ft_path).iterdir()):
            _st_model = SentenceTransformer(ft_path)
            log.info(f"ST fine-tunÃ© chargÃ© : {ft_path}")
            return True
        else:
            log.warning(f"Fine-tuned model absent ({ft_path}), chargement baseline...")
            _st_model = SentenceTransformer(base_model)
            log.info(f"ST baseline chargÃ© : {base_model}")
            return False
    except Exception as e:
        log.error(f"Impossible de charger le ST model : {e}")
        _st_model = None
        return False


def get_st_model():
    """DÃ©pendance FastAPI â†’ injecte le ST model dans les routes."""
    return _st_model


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# NEO4J â€” pool de sessions
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        log.info(f"Neo4j connectÃ© : {uri}")
        return True
    except Exception as e:
        log.warning(f"Neo4j indisponible ({e}) â€” mode dÃ©gradÃ© sans graphe")
        _neo4j_driver = None
        return False


def get_neo4j():
    return _neo4j_driver


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# POSTGRESQL / PGVECTOR
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def init_pgvector(
    dsn: str = "postgresql://postgres:password@localhost:5432/recommandation",
) -> bool:
    global _pg_conn
    try:
        import psycopg
        _pg_conn = psycopg.connect(dsn)
        _pg_conn.autocommit = False
        log.info("pgvector connectÃ©")
        return True
    except Exception as e:
        log.warning(f"pgvector indisponible ({e}) â€” mode dÃ©gradÃ© sans ANN")
        _pg_conn = None
        return False


def get_pg():
    return _pg_conn


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MOTEUR DE RECOMMANDATION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def init_engine(llm_backend: str = "openrouter") -> None:
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
    log.info(f"Moteur de recommandation initialisÃ© (backend={llm_backend})")


def get_engine():
    """DÃ©pendance FastAPI â†’ injecte le moteur dans les routes."""
    if _recommendation_engine is None:
        raise RuntimeError(
            "Moteur non initialisÃ© â€” vÃ©rifier le dÃ©marrage de l'API (lifespan)"
        )
    return _recommendation_engine


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HEALTH CHECK
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_services_status() -> dict:
    """Retourne l'Ã©tat de chaque service pour le health check."""
    return {
        "neo4j":    "connected" if _neo4j_driver else "unavailable",
        "pgvector": "connected" if _pg_conn else "unavailable",
        "st_model": (
            f"loaded ({_st_model.get_sentence_embedding_dimension()}d)"
            if _st_model else "unavailable"
        ),
        "llm": (
            getattr(_recommendation_engine, "llm", None)
            and f"{getattr(_recommendation_engine.llm, 'backend', 'openrouter')}:{getattr(_recommendation_engine.llm, 'model', 'openai/gpt-oss-20b:free')}"
            or "openrouter:openai/gpt-oss-20b:free"
        ),
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FERMETURE DES CONNEXIONS (lifespan shutdown)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def close_all():
    """Ferme proprement toutes les connexions lors de l'arrÃªt de l'API."""
    if _neo4j_driver:
        _neo4j_driver.close()
        log.info("Neo4j fermÃ©")
    if _pg_conn and not _pg_conn.closed:
        _pg_conn.close()
        log.info("pgvector fermÃ©")

