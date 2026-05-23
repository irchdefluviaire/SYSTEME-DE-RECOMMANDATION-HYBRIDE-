# Systeme de recommandation hybride emploi-competences

Ce depot implemente un systeme de recommandation pour le matching
emploi-competences au Cameroun. Le systeme combine:

- des donnees locales d'offres, de demandeurs et de referentiels metiers;
- une normalisation ETL vers des fichiers Parquet/JSONL;
- un encodeur `SentenceTransformer` adapte au domaine;
- un graphe de connaissances Neo4j;
- une base vectorielle PostgreSQL avec `pgvector`;
- un moteur GraphRAG et une API FastAPI;
- une couche agentique LangGraph pour orchestrer le raisonnement.

Ce README decrit ce qui est present dans le code. Il ne remplace pas les
resultats produits par les scripts: les metriques doivent etre lues dans les
artefacts generes localement apres execution.

## Principe du systeme

Le systeme ne pre-entraine pas un modele de langue depuis zero. Le module 02
charge `sentence-transformers/all-MiniLM-L6-v2`, puis l'adapte par fine-tuning
contrastif sur des paires issues des offres locales:

- `sentence1`: metadonnees structurees de l'offre;
- `sentence2`: description textuelle et competences de l'offre;
- `offre_id`: identifiant utilise pour l'evaluation de recherche
  d'information.

Le modele produit des embeddings denses de dimension 384. Ces vecteurs servent
ensuite a la recherche semantique dans `pgvector`. Le graphe Neo4j conserve les
relations institutionnelles et metiers: candidats, offres, competences ESCO,
metiers ESCO, niveaux NCF, groupes MEPC, secteurs, employeurs et localisations.

La recommandation finale est hybride: elle exploite la similarite vectorielle,
les relations du graphe, des regles de niveau/competence et, selon le mode
choisi, une generation LLM pour formuler les explications, le skill gap et la
roadmap. Le score final est interpretable pour un usage recrutement: il combine
la proximite semantique, la couverture des competences, la compatibilite de
niveau NCF et l'alignement metier/secteur, puis produit un verdict operationnel
(`pret_a_postuler`, `postuler_avec_plan_de_montee_en_competence`,
`vivier_a_developper` ou `hors_cible_actuel`).

## Structure utile du depot

```text
.
|-- data/
|   |-- raw/                         donnees sources locales et ESCO
|   |-- processed/                   fichiers normalises Parquet/XLSX/CSV
|   `-- finetune/                    pairs_train/val/test.jsonl
|-- models/
|   `-- st_finetuned/                sorties du fine-tuning SentenceTransformer
|-- notebooks/                       notebooks d'exploration et validation
|-- scripts/
|   `-- run_etl.py                   orchestration ETL
|-- src/
|   |-- 01_etl/                      normalisation et construction des paires
|   |-- 02_finetune_st/              fine-tuning et evaluation ST
|   |-- 03_knowledge_graph/          chargement Neo4j
|   |-- 04_pgvector/                 schema, embeddings et recherche ANN
|   |-- 05_graphrag/                 moteur GraphRAG
|   |-- 06_api/                      API FastAPI
|   |-- 07_evaluation/               evaluation systeme
|   `-- 08_agentic_graphrag/         workflow LangGraph
|-- langgraph.json                   declaration du graphe LangGraph
|-- pyproject.toml                   dependances Poetry
|-- poetry.lock                      versions resolues
`-- README.md
```

## Configuration

Le projet lit les variables d'environnement depuis `.env`. Un modele est fourni
dans `.env.example`. Les variables minimales sont:

```powershell
PG_HOST=localhost
PG_PORT=5432
PG_DB=test_kmer
PG_USER=postgres
PG_PASSWORD=...

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j

AGENT_USE_OLLAMA=0
OLLAMA_MODEL=llama3.1:latest
OLLAMA_BASE_URL=http://localhost:11434
```

Ne versionne pas ton vrai `.env`. Le fichier est ignore par Git.

## Installation avec Poetry

Le projet demande Python `>=3.13` dans `pyproject.toml`.

```powershell
cd "D:\DATA SCIENCES\SYSTEME-DE-RECOMMANDATION-HYBRIDE-"
poetry env use python
poetry lock
poetry install
poetry check
```

Dans l'etat courant du depot, `poetry check` peut signaler que
`pyproject.toml` a change significativement depuis la generation de
`poetry.lock`. Dans ce cas, regenerer le lockfile avec `poetry lock`, puis
relancer `poetry install`.

Toutes les commandes ci-dessous se lancent depuis la racine du depot avec
`poetry run`.

## Etapes d'execution reproductibles

### 1. Executer l'ETL

Le script `scripts/run_etl.py` orchestre quatre etapes:

1. normalisation des offres;
2. normalisation des candidats;
3. alignement des referentiels MEPC, NCF et ESCO;
4. construction des paires de fine-tuning.

Commande:

```powershell
poetry run python scripts/run_etl.py
```

Sorties attendues:

```text
data/processed/offres_normalized.parquet
data/processed/candidats_normalized.parquet
data/processed/mapping_isco_mepc_esco.parquet
data/finetune/pairs_train.jsonl
data/finetune/pairs_val.jsonl
data/finetune/pairs_test.jsonl
```

### 2. Adapter le SentenceTransformer

Configuration:

```text
src/02_finetune_st/config_st.json
```

Parametres actuellement declares:

```text
modele_base   = sentence-transformers/all-MiniLM-L6-v2
dim_embedding = 384
epochs        = 5
batch_size    = 32
learning_rate = 2e-5
loss          = MultipleNegativesRankingLoss
similarite    = cosine
```

Entrainement:

```powershell
poetry run python src/02_finetune_st/train_sentence_transformer.py
```

Test rapide:

```powershell
poetry run python src/02_finetune_st/train_sentence_transformer.py --epochs 1 --batch 8
```

Par defaut, le script charge le modele Hugging Face depuis le cache local
(`local_files_only=True`). Pour autoriser un telechargement reseau:

```powershell
poetry run python src/02_finetune_st/train_sentence_transformer.py --online
```

Sorties:

```text
models/st_finetuned/final/
models/st_finetuned/checkpoints/
models/st_finetuned/evaluation_metrics.json
models/st_finetuned/eval_test/
```

### 3. Evaluer le modele d'embedding

Evaluation comparative baseline vs modele adapte:

```powershell
poetry run python src/02_finetune_st/evaluate_st.py
```

Evaluation sans baseline:

```powershell
poetry run python src/02_finetune_st/evaluate_st.py --no-baseline
```

Evaluation d'un modele donne:

```powershell
poetry run python src/02_finetune_st/evaluate_st.py --model-ft models/st_finetuned/final
```

Le protocole utilise `InformationRetrievalEvaluator`: les metadonnees sont les
requetes, les descriptions sont le corpus, et le document pertinent partage le
meme `offre_id`. Les correlations Pearson/Spearman ne sont pas appropriees ici
si les paires sont uniquement positives.

### 4. Charger le graphe Neo4j

Le module Neo4j cree le schema, charge ESCO, MEPC, NCF, les offres, les
candidats et les relations. Les chargements utilisent `MERGE`, donc ils sont
prevus pour etre relances sans creer de doublons sur les cles gerees par le
schema.

Validation sans ecriture:

```powershell
poetry run python src/03_knowledge_graph/load_neo4j.py --dry-run
```

Chargement complet:

```powershell
poetry run python src/03_knowledge_graph/load_neo4j.py
```

Executer une etape:

```powershell
poetry run python src/03_knowledge_graph/load_neo4j.py --step schema
poetry run python src/03_knowledge_graph/load_neo4j.py --step esco
poetry run python src/03_knowledge_graph/load_neo4j.py --step mepc
poetry run python src/03_knowledge_graph/load_neo4j.py --step ncf
poetry run python src/03_knowledge_graph/load_neo4j.py --step offres
poetry run python src/03_knowledge_graph/load_neo4j.py --step candidats
poetry run python src/03_knowledge_graph/load_neo4j.py --step relations
poetry run python src/03_knowledge_graph/load_neo4j.py --step skills
```

Attention: `--clear` supprime les noeuds et relations de la base avant
rechargement.

```powershell
poetry run python src/03_knowledge_graph/load_neo4j.py --clear
```

### 5. Encoder les entites dans pgvector

Le module 04 cree/verifie le schema PostgreSQL, encode les entites avec le
SentenceTransformer et insere les vecteurs dans la table `embeddings`.

Validation sans insertion:

```powershell
poetry run python src/04_pgvector/embed_all_entities.py --dry-run
```

Encodage complet:

```powershell
poetry run python src/04_pgvector/embed_all_entities.py
```

Encoder une famille d'entites:

```powershell
poetry run python src/04_pgvector/embed_all_entities.py --entity offres
poetry run python src/04_pgvector/embed_all_entities.py --entity candidats
poetry run python src/04_pgvector/embed_all_entities.py --entity skills
poetry run python src/04_pgvector/embed_all_entities.py --entity metiers
poetry run python src/04_pgvector/embed_all_entities.py --entity ncf
poetry run python src/04_pgvector/embed_all_entities.py --entity mepc
```

Modele explicite:

```powershell
poetry run python src/04_pgvector/embed_all_entities.py --model models/st_finetuned/final --batch-size 64
```

### 6. Tester le moteur GraphRAG

Le moteur `src/05_graphrag/recommendation_engine.py` charge un candidat,
construit un contexte via pgvector et Neo4j, puis produit les recommandations,
le skill gap et la roadmap. Le backend par defaut est `simulation`, ce qui
permet un test sans LLM externe.

```powershell
poetry run python src/05_graphrag/recommendation_engine.py --candidat PPKOU2501080016340 --backend simulation --top-k 5
```

Benchmark local sur un echantillon:

```powershell
poetry run python src/05_graphrag/recommendation_engine.py --benchmark --backend simulation
```

Backends declares dans le code:

```text
simulation
mistral
openai
```

Les backends `mistral` et `openai` exigent les dependances, modeles et cles
necessaires dans l'environnement. Le mode `simulation` produit des JSON de test:
il ne doit pas etre interprete comme une validation empirique du systeme.

### 7. Lancer l'API FastAPI

Depuis la racine:

```powershell
poetry run uvicorn src.06_api.main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints exposes par le code:

```text
GET  /
GET  /health
POST /recommend
GET  /recommend/candidat/{id}
POST /skill-gap
POST /embed
GET  /offre/{id}
GET  /offre
GET  /docs
GET  /redoc
```

Documentation locale apres lancement:

```text
http://localhost:8000/docs
http://localhost:8000/redoc
```

### 8. Tester le workflow Agentic GraphRAG

Le graphe LangGraph est declare dans `langgraph.json`:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agentic_graphrag": "./src/08_agentic_graphrag/graph.py:graph"
  },
  "env": ".env"
}
```

Test CLI:

```powershell
poetry run python src/08_agentic_graphrag/run_agent.py --candidat PPKOU2501080016340 --top-k 5
```

Lancement LangGraph Studio/API locale si `langgraph-cli[inmem]` est installe par
Poetry:

```powershell
poetry run langgraph dev
```

Workflow documente dans `src/08_agentic_graphrag/README.md`:

1. `analyse_request`
2. `load_profile`
3. `retrieve_and_check_graph`
4. `compute_skill_gap`
5. `score_and_rank`
6. `critique_recommendations`
7. `create_roadmap`
8. `generate_final_answer`

Le mode declare par la CLI est `real`: le mode simulation est desactive dans ce
workflow agentique.

### 9. Evaluer le systeme

Evaluation complete:

```powershell
poetry run python src/07_evaluation/evaluate_system.py
```

Evaluation par module:

```powershell
poetry run python src/07_evaluation/evaluate_system.py --module st
poetry run python src/07_evaluation/evaluate_system.py --module graphrag --n-candidats 100
poetry run python src/07_evaluation/evaluate_system.py --module scores --n-candidats 100
poetry run python src/07_evaluation/evaluate_system.py --module latence --n-candidats 50
```

Sortie principale:

```text
outputs/evaluation/evaluation_report.json
```

Point de vigilance: `evaluate_system.py` contient des chemins de secours qui
retournent des metriques simulees si certains artefacts ou modeles ne sont pas
disponibles. Pour une interpretation scientifique, utiliser en priorite les
artefacts calcules par `evaluate_st.py`, les logs d'execution et le rapport JSON
genere dans ton environnement.

## Ordre d'execution recommande

```powershell
cd "D:\DATA SCIENCES\SYSTEME-DE-RECOMMANDATION-HYBRIDE-"
poetry lock
poetry install

poetry run python scripts/run_etl.py
poetry run python src/02_finetune_st/train_sentence_transformer.py
poetry run python src/02_finetune_st/evaluate_st.py

poetry run python src/03_knowledge_graph/load_neo4j.py --dry-run
poetry run python src/03_knowledge_graph/load_neo4j.py

poetry run python src/04_pgvector/embed_all_entities.py --dry-run
poetry run python src/04_pgvector/embed_all_entities.py

poetry run python src/05_graphrag/recommendation_engine.py --candidat PPKOU2501080016340 --backend simulation --top-k 5
poetry run uvicorn src.06_api.main:app --host 0.0.0.0 --port 8000 --reload
poetry run python src/08_agentic_graphrag/run_agent.py --candidat PPKOU2501080016340 --top-k 5
```

## Sources techniques

- Poetry CLI: https://python-poetry.org/docs/cli/
- Sentence Transformers, `MultipleNegativesRankingLoss`: https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html
- Sentence Transformers, `InformationRetrievalEvaluator`: https://www.sbert.net/docs/package_reference/sentence_transformer/evaluation.html
- pgvector: https://github.com/pgvector/pgvector
- Neo4j Python driver: https://neo4j.com/docs/python-manual/current/
- FastAPI, lancement avec serveur ASGI: https://fastapi.tiangolo.com/deployment/manually/
- LangGraph: https://langchain-ai.github.io/langgraph/
