# PLANNING.md - Feuille de route technique

## Phases du projet

### Phase 1: ETL & Normalisation
- [ ] Chargement des données sources (offres, candidats, référentiels ESCO)
- [ ] Normalisation en Parquet/JSONL
- [ ] Génération des paires pour fine-tuning

### Phase 2: Fine-tuning SentenceTransformer
- [ ] Adaptation du modèle all-MiniLM-L6-v2
- [ ] Évaluation sur paires locales
- [ ] Stockage du modèle fine-tuné

### Phase 3: Knowledge Graph
- [ ] Ingestion Neo4j des relations (candidats, offres, compétences ESCO, métiers)
- [ ] Pondération des relations (confidence, weight)
- [ ] Indexation des PDFs réglementaires (NCF, MEPC)

### Phase 4: Recherche vectorielle
- [ ] Setup PostgreSQL pgvector
- [ ] Indexation des embeddings
- [ ] Implémentation ANN search

### Phase 5: GraphRAG & Agentic
- [ ] Moteur GraphRAG (mode non-agentique)
- [ ] Workflow LangGraph avec outils
- [ ] Orchestration du raisonnement

### Phase 6: APIs & Interfaces
- [ ] API FastAPI avec /chat/stream
- [ ] Interface Streamlit chatbot
- [ ] CLI interactive

### Phase 7: Évaluation
- [ ] Métriques de recommandation
- [ ] Benchmark des embeddings
- [ ] Rapports d'évaluation

## Décisions d'architecture
- Format stockage: Parquet/JSONL (performance + lisibilité)
- Modèle de base: all-MiniLM-L6-v2 (léger, efficace)
- Framework GraphRAG: LangGraph (flexibilité, orchestration)
- DB vecteurs: PostgreSQL pgvector (intégration)
- DB graphes: Neo4j (relations complexes)
