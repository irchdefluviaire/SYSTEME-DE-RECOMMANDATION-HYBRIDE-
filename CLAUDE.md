# CLAUDE.md - Contexte du projet pour Claude Code

## Architecture générale

Le système de recommandation hybride emploi-compétences combine:
- **ETL & Normalisation:** extraction et transformation des données sources
- **Fine-tuning SentenceTransformer:** adaptation d'embeddings au domaine métier
- **Knowledge Graph (Neo4j):** relations institutionnelles et métiers avec poids
- **Recherche vectorielle (PostgreSQL pgvector):** similarité sémantique dense
- **GraphRAG & Agentic:** orchestration LangGraph avec outils et raisonnement
- **APIs & Interfaces:** FastAPI (backend), Streamlit (UI), CLI interactive

## Convention de code
- Python 3.10+
- Type hints requis
- Docstrings en français
- Logs structurés via logging

## Points d'attention
- Gestion des secrets via `.env` (jamais committer)
- Dépendances Poetry (pyproject.toml)
- Modèles stockés dans `models/st_finetuned/`
- Outputs versionnés dans `outputs/`
