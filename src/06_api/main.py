"""
main.py â€” Application FastAPI principale
===========================================================================
Module 06 â€” Exposition du systÃ¨me de recommandation via API REST

Endpoints :
  POST  /recommend             â†’ top-k offres + skill gap + roadmap
  GET   /recommend/candidat/{id} â†’ version GET simplifiÃ©e
  POST  /skill-gap             â†’ analyse dÃ©taillÃ©e d'une paire (candidat, offre)
  POST  /embed                 â†’ encoder des textes en vecteurs 384d
  GET   /offre/{id}            â†’ dÃ©tails d'une offre
  GET   /offres                â†’ recherche d'offres avec filtres
  GET   /health                â†’ Ã©tat de tous les services
  GET   /docs                  â†’ Swagger UI (auto-gÃ©nÃ©rÃ©e)
  GET   /redoc                 â†’ ReDoc (auto-gÃ©nÃ©rÃ©e)

Lancement :
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
  uvicorn main:app --workers 4   # production

Variables d'environnement :
  LLM unique: OpenRouter openai/gpt-oss-20b:free
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
from routers import recommend, skill_gap, embed, chat
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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# LIFESPAN â€” dÃ©marrage et arrÃªt
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Cycle de vie de l'application.
    Tout ce qui est avant `yield` s'exÃ©cute au dÃ©marrage.
    Tout ce qui est aprÃ¨s `yield` s'exÃ©cute Ã  l'arrÃªt.

    Pattern singleton : les connexions sont initialisÃ©es une seule fois
    et partagÃ©es entre toutes les routes via les dÃ©pendances (Depends).
    """
    log.info("=" * 60)
    log.info("DÃ‰MARRAGE API â€” SystÃ¨me de Recommandation Emploi-CompÃ©tences")
    log.info("=" * 60)

    # 1. Charger le SentenceTransformer fine-tunÃ©
    model_path = os.getenv("MODEL_PATH")
    ft_loaded  = init_st_model(model_path)
    log.info(f"  ST model : {'fine-tunÃ©' if ft_loaded else 'baseline (fallback)'}")

    # 2. Connexion Neo4j (mode dÃ©gradÃ© si absent)
    neo4j_ok = init_neo4j(
        uri=os.getenv("NEO4J_URI",      "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER",    "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )
    log.info(f"  Neo4j    : {'OK' if neo4j_ok else 'indisponible (mode dÃ©gradÃ©)'}")

    # 3. Connexion pgvector (mode dÃ©gradÃ© si absent)
    pg_ok = init_pgvector(dsn=_pg_dsn_from_env())
    log.info(f"  pgvector : {'OK' if pg_ok else 'indisponible (mode dÃ©gradÃ©)'}")

    # 4. Instancier le moteur de recommandation
    llm_backend = "openrouter"
    init_engine(llm_backend=llm_backend)
    log.info(f"  LLM 2    : {llm_backend}")

    log.info("API prÃªte â€” en Ã©coute sur http://0.0.0.0:8000")
    log.info("Documentation : http://0.0.0.0:8000/docs")

    yield  # â† L'API tourne ici

    # ArrÃªt propre
    log.info("ArrÃªt de l'API â€” fermeture des connexions...")
    close_all()
    log.info("API arrÃªtÃ©e proprement.")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# APPLICATION FASTAPI
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app = FastAPI(
    title="API Recommandation Emploi-CompÃ©tences â€” Cameroun",
    description="""
## SystÃ¨me de Recommandation Hybride (GraphRAG)

API REST exposant le moteur de recommandation emploi-compÃ©tences
conÃ§u pour le marchÃ© du travail camerounais.

### Architecture (LLM 1 + LLM 2)

**LLM 1 â€” SentenceTransformer fine-tunÃ©** (`all-MiniLM-L6-v2`, 384d)
- Encode les profils candidats et les offres en vecteurs denses
- Fine-tunÃ© sur paires (mÃ©tadonnÃ©es â†’ description) d'offres camerounaises
- StockÃ©s dans pgvector avec index HNSW

**LLM 2 â€” GÃ©nÃ©ratif GraphRAG** (Mistral-7B / GPT-4o)
- ReÃ§oit le contexte Neo4j + pgvector
- GÃ©nÃ¨re recommandations + skill gap + roadmap NCF en franÃ§ais

### Score hybride

`score = 0.40 Ã— sÃ©mantique + 0.35 Ã— graphe + 0.25 Ã— collaboratif`

### RÃ©fÃ©rentiels intÃ©grÃ©s
- ESCO v1.2 (13 939 compÃ©tences, 3 039 mÃ©tiers)
- MEPC 2013 â€” INS Cameroun (209 groupes)
- NCF 2017 â€” INS Cameroun (201 domaines)
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Recommandation", "description": "Recommandation d'offres d'emploi"},
        {"name": "Skill Gap",      "description": "Analyse de l'Ã©cart de compÃ©tences"},
        {"name": "Embeddings",     "description": "Encodage de textes en vecteurs 384d"},
        {"name": "Offres",         "description": "Consultation des offres d'emploi"},
        {"name": "SystÃ¨me",        "description": "Health check et mÃ©ta-informations"},
    ],
    lifespan=lifespan,
)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MIDDLEWARES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # En production : restreindre aux domaines autorisÃ©s
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Middleware de logging des requÃªtes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0       = time.time()
    response = await call_next(request)
    elapsed  = round((time.time() - t0) * 1000, 1)
    log.info(
        f"{request.method} {request.url.path} "
        f"â†’ {response.status_code} ({elapsed}ms)"
    )
    return response


# Handler d'erreurs global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Erreur non gÃ©rÃ©e : {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            code=500,
            message="Erreur interne du serveur",
            detail=str(exc),
        ).model_dump(mode="json"),
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ROUTES PRINCIPALES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    chat.router,
    prefix="/chat",
    tags=["Agentic GraphRAG"],
)

app.include_router(
    offre.router,
    prefix="/offre",
    tags=["Offres"],
)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ENDPOINT HEALTH
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["SystÃ¨me"],
    summary="Ã‰tat de l'API et de ses services",
    description="""
VÃ©rifie la disponibilitÃ© de tous les services :
- SentenceTransformer fine-tunÃ© (LLM 1)
- Neo4j (graphe de connaissances)
- PostgreSQL / pgvector (embeddings HNSW)
- Backend LLM 2 OpenRouter (openai/gpt-oss-20b:free)
""",
)
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services=get_services_status(),
    )


@app.get("/", tags=["SystÃ¨me"], summary="Informations de base sur l'API")
async def root():
    return {
        "name":        "API Recommandation Emploi-CompÃ©tences â€” Cameroun",
        "version":     "1.0.0",
        "docs":        "/docs",
        "redoc":       "/redoc",
        "health":      "/health",
        "endpoints": {
            "POST /recommend":         "Recommandation top-k offres + skill gap + roadmap",
            "GET  /recommend/candidat/{id}": "Recommandation rapide par matricule",
            "POST /skill-gap":         "Analyse skill gap pour une paire (candidat, offre)",
            "POST /embed":             "Encoder textes â†’ vecteurs 384d",
            "GET  /offre/{id}":        "DÃ©tails d'une offre",
            "GET  /offre":             "Recherche offres (filtres secteur/ville/NCF)",
            "GET  /health":            "Ã‰tat des services",
        },
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# POINT D'ENTRÃ‰E DIRECT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,        # rechargement auto en dÃ©veloppement
        log_level="info",
    )

