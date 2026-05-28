"""
main.py — Application FastAPI principale
===========================================================================
Module 06 — Exposition du système de recommandation via API REST

Endpoints :
  POST  /recommend             → top-k offres + skill gap + roadmap
  GET   /recommend/candidat/{id} → version GET simplifiée
  POST  /skill-gap             → analyse détaillée d'une paire (candidat, offre)
  POST  /embed                 → encoder des textes en vecteurs 384d
  GET   /offre/{id}            → détails d'une offre
  GET   /offres                → recherche d'offres avec filtres
  GET   /health                → état de tous les services
  GET   /docs                  → Swagger UI (auto-générée)
  GET   /redoc                 → ReDoc (auto-générée)

Lancement :
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
  uvicorn main:app --workers 4   # production

Variables d'environnement :
  LLM local unique: Ollama qwen2:1.5b
  NEO4J_URI=bolt://localhost:7687
  NEO4J_PASSWORD=password
  PG_DSN=postgresql://postgres:password@localhost:5432/recommandation
  MODEL_PATH=./models/st_finetuned/final
===========================================================================
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Assurer que les modules projet sont accessibles
ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src" / "05_graphrag"))
sys.path.insert(0, str(Path(__file__).parent))

from dependencies import (
    init_st_model, init_neo4j, init_pgvector,
    init_engine, close_all, get_services_status,
)
from schemas import HealthResponse, ErrorResponse
from routers import recommend, skill_gap, embed
import offre

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")


def _pg_dsn_from_env() -> str:
    if os.getenv("PG_DSN"):
        return os.getenv("PG_DSN")
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db = os.getenv("PG_DB", "recommandation")
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "password")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


# ─────────────────────────────────────────────────────────────────────────
# LIFESPAN — démarrage et arrêt
# ─────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cycle de vie de l'application.
    Tout ce qui est avant `yield` s'exécute au démarrage.
    Tout ce qui est après `yield` s'exécute à l'arrêt.

    Pattern singleton : les connexions sont initialisées une seule fois
    et partagées entre toutes les routes via les dépendances (Depends).
    """
    log.info("=" * 60)
    log.info("DÉMARRAGE API — Système de Recommandation Emploi-Compétences")
    log.info("=" * 60)

    # 1. Charger le SentenceTransformer fine-tuné
    model_path = os.getenv("MODEL_PATH")
    ft_loaded  = init_st_model(model_path)
    log.info(f"  ST model : {'fine-tuné' if ft_loaded else 'baseline (fallback)'}")

    # 2. Connexion Neo4j (mode dégradé si absent)
    neo4j_ok = init_neo4j(
        uri=os.getenv("NEO4J_URI",      "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER",    "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )
    log.info(f"  Neo4j    : {'OK' if neo4j_ok else 'indisponible (mode dégradé)'}")

    # 3. Connexion pgvector (mode dégradé si absent)
    pg_ok = init_pgvector(dsn=_pg_dsn_from_env())
    log.info(f"  pgvector : {'OK' if pg_ok else 'indisponible (mode dégradé)'}")

    # 4. Instancier le moteur de recommandation
    llm_backend = "ollama"
    init_engine(llm_backend=llm_backend)
    log.info(f"  LLM 2    : {llm_backend}")

    log.info("API prête — en écoute sur http://0.0.0.0:8000")
    log.info("Documentation : http://0.0.0.0:8000/docs")

    yield  # ← L'API tourne ici

    # Arrêt propre
    log.info("Arrêt de l'API — fermeture des connexions...")
    close_all()
    log.info("API arrêtée proprement.")


# ─────────────────────────────────────────────────────────────────────────
# APPLICATION FASTAPI
# ─────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="API Recommandation Emploi-Compétences — Cameroun",
    description="""
## Système de Recommandation Hybride (GraphRAG)

API REST exposant le moteur de recommandation emploi-compétences
conçu pour le marché du travail camerounais.

### Architecture (LLM 1 + LLM 2)

**LLM 1 — SentenceTransformer fine-tuné** (`all-MiniLM-L6-v2`, 384d)
- Encode les profils candidats et les offres en vecteurs denses
- Fine-tuné sur paires (métadonnées → description) d'offres camerounaises
- Stockés dans pgvector avec index HNSW

**LLM 2 — Génératif GraphRAG** (Mistral-7B / GPT-4o)
- Reçoit le contexte Neo4j + pgvector
- Génère recommandations + skill gap + roadmap NCF en français

### Score hybride

`score = 0.40 × sémantique + 0.35 × graphe + 0.25 × collaboratif`

### Référentiels intégrés
- ESCO v1.2 (13 939 compétences, 3 039 métiers)
- MEPC 2013 — INS Cameroun (209 groupes)
- NCF 2017 — INS Cameroun (201 domaines)
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Recommandation", "description": "Recommandation d'offres d'emploi"},
        {"name": "Skill Gap",      "description": "Analyse de l'écart de compétences"},
        {"name": "Embeddings",     "description": "Encodage de textes en vecteurs 384d"},
        {"name": "Offres",         "description": "Consultation des offres d'emploi"},
        {"name": "Système",        "description": "Health check et méta-informations"},
    ],
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────
# MIDDLEWARES
# ─────────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # En production : restreindre aux domaines autorisés
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Middleware de logging des requêtes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0       = time.time()
    response = await call_next(request)
    elapsed  = round((time.time() - t0) * 1000, 1)
    log.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({elapsed}ms)"
    )
    return response


# Handler d'erreurs global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Erreur non gérée : {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            code=500,
            message="Erreur interne du serveur",
            detail=str(exc),
        ).model_dump(mode="json"),
    )


# ─────────────────────────────────────────────────────────────────────────
# ROUTES PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────

app.include_router(
    recommend.router,
    prefix="/recommend",
    tags=["Recommandation"],
)

app.include_router(
    skill_gap.router,
    prefix="/skill-gap",
    tags=["Skill Gap"],
)

app.include_router(
    embed.router,
    prefix="/embed",
    tags=["Embeddings"],
)

app.include_router(
    offre.router,
    prefix="/offre",
    tags=["Offres"],
)


# ─────────────────────────────────────────────────────────────────────────
# ENDPOINT HEALTH
# ─────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Système"],
    summary="État de l'API et de ses services",
    description="""
Vérifie la disponibilité de tous les services :
- SentenceTransformer fine-tuné (LLM 1)
- Neo4j (graphe de connaissances)
- PostgreSQL / pgvector (embeddings HNSW)
- Backend LLM 2 local (Ollama qwen2:1.5b)
""",
)
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services=get_services_status(),
    )


@app.get("/", tags=["Système"], summary="Informations de base sur l'API")
async def root():
    return {
        "name":        "API Recommandation Emploi-Compétences — Cameroun",
        "version":     "1.0.0",
        "docs":        "/docs",
        "redoc":       "/redoc",
        "health":      "/health",
        "endpoints": {
            "POST /recommend":         "Recommandation top-k offres + skill gap + roadmap",
            "GET  /recommend/candidat/{id}": "Recommandation rapide par matricule",
            "POST /skill-gap":         "Analyse skill gap pour une paire (candidat, offre)",
            "POST /embed":             "Encoder textes → vecteurs 384d",
            "GET  /offre/{id}":        "Détails d'une offre",
            "GET  /offre":             "Recherche offres (filtres secteur/ville/NCF)",
            "GET  /health":            "État des services",
        },
    }


# ─────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE DIRECT
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,        # rechargement auto en développement
        log_level="info",
    )
