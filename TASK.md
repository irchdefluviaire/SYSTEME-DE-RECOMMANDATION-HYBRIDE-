# TASK.md - Backlog des tâches

## Format
Chaque tâche inclut:
- ID unique (kebab-case)
- Titre
- Description
- État: `TODO | IN_PROGRESS | DONE | BLOCKED`

## Tâches actives

### Infrastructure
- [ ] **setup-env** - Configurer `.env` avec variables PostgreSQL/Neo4j
- [ ] **setup-db** - Initialiser schémas PostgreSQL (pgvector) et Neo4j
- [ ] **poetry-install** - Installer les dépendances via Poetry

### Données
- [ ] **etl-sources** - Charger et normaliser les données sources
- [ ] **ingest-esco** - Intégrer le référentiel ESCO
- [ ] **ingest-pdfs** - Chunker et indexer NCF/MEPC

### Modèles
- [ ] **finetune-st** - Fine-tuning SentenceTransformer sur paires locales
- [ ] **eval-embeddings** - Benchmark des embeddings adapté au domaine

### Systèmes
- [ ] **neo4j-graph** - Construire le graphe de connaissances
- [ ] **pgvector-index** - Indexer les embeddings en PostgreSQL
- [ ] **graphrag-engine** - Implémenter le moteur GraphRAG

### APIs & Interfaces
- [ ] **fastapi-setup** - Structure API FastAPI + endpoint /chat/stream
- [ ] **streamlit-ui** - Développer interface chatbot Streamlit
- [ ] **cli-interactive** - Implémenter CLI avec visibilité outils

### Évaluation
- [ ] **eval-system** - Rapports d'évaluation du système
- [ ] **traces-export** - Export LangSmith et visualisations

## Tâches terminées
(à remplir au fur et à mesure)

## Notes
- Chaque tâche peut dépendre d'autres (documenter les dépendances)
- Priorité: Infrastructure → Données → Modèles → Systèmes → APIs → Évaluation
