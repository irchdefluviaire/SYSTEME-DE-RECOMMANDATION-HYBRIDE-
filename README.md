# Systeme de recommandation hybride emploi-competences

Ce depot implemente un systeme de recommandation pour le matching
emploi-competences au Cameroun. Le systeme combine:

- des donnees locales d'offres, de demandeurs et de referentiels metiers;
- une normalisation ETL vers des fichiers Parquet/JSONL;
- un encodeur `SentenceTransformer` adapte au domaine;
- un graphe de connaissances Neo4j avec relations ponderees;
- une base vectorielle PostgreSQL avec `pgvector`;
- une ingestion des PDFs reglementaires (NCF, MEPC) avec chunking structurel;
- un moteur GraphRAG et une API FastAPI avec streaming SSE;
- une couche agentique LangGraph orchestrant le raisonnement;
- une interface chatbot Streamlit et une CLI interactive.

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
Les relations cles portent une propriete `confidence` ou `weight` permettant
un scoring pondere directement en Cypher.

Les PDFs officiels des nomenclatures (NCF 2017, MEPC 2013, referentiel de
diplomes) sont chunkes selon leur structure hierarchique, embeddes avec le
meme modele fine-tune, indexes dans `pgvector` et relies aux noeuds Neo4j
correspondants par des relations `:DEFINIT` et `:DECRIT`. L'agent peut ainsi
citer les passages reglementaires officiels pour justifier ses recommandations.

La recommandation finale est hybride: elle exploite la similarite vectorielle,
les relations du graphe, des regles de niveau/competence et, selon le mode
choisi, une generation LLM pour formuler les explications, le skill gap et la
roadmap. Le score final est interpretable pour un usage recrutement: il combine
la proximite semantique, la couverture des competences, la compatibilite de
niveau NCF et l'alignement metier/secteur, puis produit un verdict operationnel
(`pret_a_postuler`, `postuler_avec_plan_de_montee_en_competence`,
`vivier_a_developper` ou `hors_cible_actuel`).

## Documents compagnons

- `CLAUDE.md`: contexte du projet pour l'assistant Claude Code (architecture,
  conventions, points d'attention).
- `PLANNING.md`: feuille de route technique et decisions d'architecture.
- `TASK.md`: backlog des taches en cours et terminees.

## Structure utile du depot

```text
.
|-- chatbot_app.py                        interface Streamlit (entry point UI)
|-- cli.py                                CLI interactive avec tool visibility
|-- config.py                             chemins centralises
|-- langgraph.json                        declaration du graphe LangGraph
|-- pyproject.toml / poetry.lock          dependances Poetry
|-- CLAUDE.md / PLANNING.md / TASK.md     documents projet
|
|-- data/
|   |-- raw/                              donnees sources locales et ESCO
|   |-- processed/                        fichiers normalises Parquet/XLSX/CSV
|   |-- finetune/                         pairs_train/val/test.jsonl
|   `-- pdf_chunks/                       cache JSONL des chunks PDF
|
|-- pdf/                                  sources reglementaires officielles
|   |-- Nomenclature-Camerounaise-des-Formations-24.01.2017.pdf
|   |-- Nomenclature-camerounaise-des-metiers-_2013.pdf
|   `-- diplome_certificat.pdf
|
|-- models/
|   `-- st_finetuned/                     sorties du fine-tuning SentenceTransformer
|
|-- notebooks/                            notebooks d'exploration et validation
|
|-- scripts/
|   |-- run_etl.py                        orchestration ETL
|   |-- ingest_pdfs.py                    chunking + indexation des PDFs
|   `-- materialize_similarity.py         calcul des relations :SIMILAIRE_A
|
|-- src/
|   |-- 00_Social_media/                  signaux LinkedIn (collaboratif)
|   |-- 01_etl/                           normalisation, paires, parsing PDF
|   |-- 02_finetune_st/                   fine-tuning et evaluation ST
|   |-- 03_knowledge_graph/               chargement Neo4j (relations ponderees)
|   |-- 04_pgvector/                      schema, embeddings et recherche ANN
|   |-- 05_graphrag/                      moteur GraphRAG (mode non-agentique)
|   |-- 06_api/                           API FastAPI + endpoint /chat/stream
|   |-- 07_evaluation/                    evaluation systeme + benchmark embeddings
|   `-- 08_agentic_graphrag/              workflow LangGraph + tools agent
|
|-- outputs/
|   |-- evaluation/                       rapports d'evaluation (JSON, CSV, PNG)
|   `-- traces/                           exports LangSmith
|
`-- rapport/                              memoire LaTeX et figures
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

# Provider LLM unique
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
API_KEY_OPEN_ROUTEUR=<votre_cle_openrouter>
OPENROUTER_MODEL=openai/gpt-oss-20b:free

# Text2Cypher local Hugging Face
TEXT2CYPHER_MODEL=neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1
TEXT2CYPHER_DEVICE_MAP=auto
TEXT2CYPHER_TORCH_DTYPE=auto
TEXT2CYPHER_MAX_NEW_TOKENS=256

# Tracing optionnel (recommande pour la demonstration H4)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=recommandation-emploi-cameroun
```

Ne versionne pas ton vrai `.env`. Le fichier est ignore par Git.

## Installation avec Poetry

Le projet demande Python `>=3.12,<3.14` (voir `pyproject.toml`).

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

### 3. Evaluer le modele d'embedding (baseline vs fine-tune)

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

### 3-bis. Benchmark multi-modeles d'embedding

Pour valider l'hypothese H3, le systeme compare le SentenceTransformer
fine-tune a un panel de modeles non adaptes couvrant trois familles:
generalistes anglais, multilingues et francais, ainsi qu'a deux baselines
lexicales (TF-IDF, BM25).

Configuration de la liste de modeles:

```text
src/07_evaluation/models_to_benchmark.json
```

Lancement du benchmark complet:

```powershell
poetry run python src/07_evaluation/benchmark_embeddings.py
```

Options utiles:

```powershell
# Inclure les baselines lexicales (TF-IDF, BM25)
poetry run python src/07_evaluation/benchmark_embeddings.py --include-lexical

# Modeles specifiques
poetry run python src/07_evaluation/benchmark_embeddings.py --models st_finetuned,multilingual-e5-base

# Intervalles de confiance par bootstrap
poetry run python src/07_evaluation/benchmark_embeddings.py --bootstrap 1000

# Mode hors-ligne (utilise uniquement le cache local)
poetry run python src/07_evaluation/benchmark_embeddings.py --offline
```

Sorties:

```text
outputs/evaluation/embedding_benchmark.csv      tableau exploitable LaTeX
outputs/evaluation/embedding_benchmark.json     detail par modele
outputs/evaluation/embedding_benchmark_plot.png graphique NDCG@10 x latence
```

Metriques calculees par modele:

- NDCG@1/5/10, MRR@10, Recall@1/5/10, Precision@1/5;
- latence d'encodage (ms/phrase), taille du modele (MB), dimension;
- intervalles de confiance via bootstrap si l'option est activee.

Pool de modeles compares par defaut:

```text
all-MiniLM-L6-v2                                 (EN, 384d)  baseline du fine-tune
paraphrase-multilingual-MiniLM-L12-v2            (ML, 384d)  meme taille, multilingue
distiluse-base-multilingual-cased-v2             (ML, 512d)
intfloat/multilingual-e5-base                    (ML, 768d)  SOTA multilingue
dangvantuan/sentence-camembert-base              (FR, 768d)  francais specifique
models/st_finetuned/final                        (FR-EMPLOI, 384d)  modele du projet
TF-IDF                                           (sparse)    baseline lexicale
BM25                                             (sparse)    reference IR
```

### 4. Charger le graphe Neo4j

Le module Neo4j cree le schema, charge ESCO, MEPC, NCF, les offres, les
candidats et les relations. Les chargements utilisent `MERGE`, donc ils sont
prevus pour etre relances sans creer de doublons sur les cles gerees par le
schema. Les relations cles (`:POSSEDE`, `:REQUIERT`, `:ALIGNE_SUR`,
`:CORRESPOND_A`) portent une propriete `confidence` exploitable en scoring.

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

### 5-bis. Ingestion des PDFs reglementaires

Les PDFs officiels du dossier `pdf/` sont chunkes selon leur structure
hierarchique (chapitres / domaines / groupes), embeddes avec le ST fine-tune,
inseres dans la table `doc_chunks` de pgvector et relies a un noeud
`DocumentReferentiel` dans Neo4j par `:EXTRAIT_DE`. Le script recupere aussi
l'identifiant Neo4j du noeud `DocChunk` cree ou mis a jour (`elementId(d)`) et
le synchronise dans `doc_chunks.neo4j_node_id` cote pgvector. L'agent peut
alors retrouver le chunk vectoriel et citer le passage officiel associe au
noeud du graphe.

Les relations `:DEFINIT` et `:DECRIT` vers les noeuds NCF/MEPC sont des
enrichissements optionnels. Elles dependent de la detection automatique des
codes dans les PDFs et ne sont pas necessaires pour que la recherche
referentielle fonctionne. Le minimum attendu pour cette etape est donc:
chunks JSONL, embeddings dans `doc_chunks`, index HNSW, `neo4j_node_id` rempli
et relations `(:DocChunk)-[:EXTRAIT_DE]->(:DocumentReferentiel)`.

Avant de lancer cette etape, verifier que la bonne base Neo4j locale est
demarree et que `.env` pointe vers cette base (`NEO4J_URI`,
`NEO4J_DATABASE`, `NEO4J_USER`, `NEO4J_PASSWORD`). Si une autre base Neo4j est
active, les `DocChunk` seront crees dans cette base et `neo4j_node_id`
referencera le mauvais graphe.

Ingestion d'un PDF:

```powershell
poetry run python scripts/ingest_pdfs.py --pdf pdf/Nomenclature-Camerounaise-des-Formations-24.01.2017.pdf --source NCF_2017
poetry run python scripts/ingest_pdfs.py --pdf pdf/Nomenclature-camerounaise-des-metiers-_2013.pdf --source MEPC_2013
poetry run python scripts/ingest_pdfs.py --pdf pdf/diplome_certificat.pdf --source diplomes
```

Options utiles:

```powershell
# Forcer l'OCR si le PDF est un scan
poetry run python scripts/ingest_pdfs.py --pdf pdf/diplome_certificat.pdf --source diplomes --ocr

# Strategie de chunking (structurel par defaut, semantique optionnel)
poetry run python scripts/ingest_pdfs.py --pdf ... --chunk-strategy structural
poetry run python scripts/ingest_pdfs.py --pdf ... --chunk-strategy semantic --chunk-size 800

# Validation sans insertion
poetry run python scripts/ingest_pdfs.py --pdf ... --dry-run
```

Sorties:

```text
data/pdf_chunks/{source}.jsonl              chunks bruts + metadonnees
pgvector: table doc_chunks                  embeddings indexes HNSW + neo4j_node_id
Neo4j: (:DocChunk)-[:EXTRAIT_DE]->(:DocumentReferentiel)
Optionnel selon les codes detectes:
       (:DocChunk)-[:DEFINIT]->(:NiveauFormationNCF | :DomaineDÃ©taillÃ©NCF)
       (:DocChunk)-[:DECRIT]->(:MÃ©tier | :GroupeBaseMEPC)
```

Verification apres ingestion:

```powershell
# pgvector : les trois sources doivent avoir neo4j_node_id rempli
poetry run python -c "import sys; sys.path.insert(0,'src/04_pgvector'); import psycopg; from config_pgvector import PG_CONN; conn=psycopg.connect(**PG_CONN); cur=conn.cursor(); cur.execute('select source, count(1), count(neo4j_node_id) from doc_chunks group by source order by source'); print(cur.fetchall()); cur.close(); conn.close()"

# Neo4j : verifier les DocChunk et les relations creees
poetry run python -c "import sys; sys.path.insert(0,'src/03_knowledge_graph'); from neo4j import GraphDatabase; from config_neo4j import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE; driver=GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)); s=driver.session(database=NEO4J_DATABASE); print([(r['source'], r['n']) for r in s.run('MATCH (d:DocChunk) RETURN d.source AS source, count(d) AS n ORDER BY source')]); print([(r['type'], r['n']) for r in s.run('MATCH (:DocChunk)-[rel]->() RETURN type(rel) AS type, count(rel) AS n ORDER BY type')]); s.close(); driver.close()"
```

### 5-ter. Materialiser les relations de similarite dans Neo4j

Calcule les relations `:SIMILAIRE_A` entre noeuds proches (cosinus > seuil)
depuis pgvector, pour activer les requetes de chemin pondere (Dijkstra/A*)
necessaires a la roadmap explicable.

```powershell
poetry run python scripts/materialize_similarity.py --entity skills --threshold 0.75
poetry run python scripts/materialize_similarity.py --entity metiers --threshold 0.80
```

Par defaut, le script prend les `10` voisins les plus proches par entite
avant filtrage par seuil, afin d'eviter une materialisation quadratique de
toutes les paires. Chaque relation creee porte:

```text
weight      similarite cosinus
similarity  meme valeur que weight
metric      cosine
source      pgvector
updated_at  date de materialisation
```

Options utiles:

```powershell
# Tester sans ecrire dans Neo4j
poetry run python scripts/materialize_similarity.py --entity skills --threshold 0.75 --dry-run

# Augmenter ou reduire le voisinage candidat
poetry run python scripts/materialize_similarity.py --entity skills --threshold 0.75 --top-k 20

# Diagnostic rapide sur un sous-ensemble
poetry run python scripts/materialize_similarity.py --entity metiers --threshold 0.80 --limit-entities 100 --dry-run
```

Verification Neo4j:

```cypher
MATCH (a:`CompÃ©tence`)-[r:SIMILAIRE_A]-(b:`CompÃ©tence`)
WHERE elementId(a) < elementId(b)
RETURN count(r) AS relations_competences;

MATCH (a:`MÃ©tier`)-[r:SIMILAIRE_A]-(b:`MÃ©tier`)
WHERE elementId(a) < elementId(b)
RETURN count(r) AS relations_metiers;

MATCH (a)-[r:SIMILAIRE_A]-(b)
WHERE elementId(a) < elementId(b)
RETURN labels(a)[0] AS type_noeud,
       coalesce(a.preferredLabel, a.label) AS source,
       coalesce(b.preferredLabel, b.label) AS cible,
       r.weight AS poids,
       r.metric AS metrique
ORDER BY r.weight DESC
LIMIT 10;
```

### 6. Tester le moteur GraphRAG (mode non-agentique)

Le moteur `src/05_graphrag/recommendation_engine.py` charge un candidat,
construit un contexte via pgvector et Neo4j, puis produit les recommandations,
le skill gap et la roadmap. Le backend generatif unique est OpenRouter avec
`OPENROUTER_MODEL=openai/gpt-oss-20b:free`.

Test avec OpenRouter:

```powershell
poetry run python src/05_graphrag/recommendation_engine.py --candidat PPKOU2501080016340 --top-k 5
```

Benchmark sur un echantillon:

```powershell
poetry run python src/05_graphrag/recommendation_engine.py --benchmark
```

Backends declares dans le code:

```text
openrouter
```

La cle `API_KEY_OPEN_ROUTEUR` doit etre presente dans `.env`. Text2Cypher utilise
separement le modele Hugging Face `neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1`;
prevoir `HF_TOKEN` si l'acces au modele/base Gemma le requiert. Si le modele
Text2Cypher local est indisponible, le systeme bascule sur des templates Cypher
read-only.

### 7. Lancer l'API FastAPI

Depuis la racine:

```powershell
poetry run uvicorn src.06_api.main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints exposes par le code:

```text
GET  /                        racine
GET  /health                  etat des services (Neo4j, pgvector, modele)
POST /recommend               top-k offres + skill gap + roadmap
GET  /recommend/candidat/{id} version GET simplifiee
POST /skill-gap               analyse detaillee d'une paire (candidat, offre)
POST /embed                   encoder des textes en vecteurs 384d
GET  /offre/{id}              details d'une offre
GET  /offres                  recherche d'offres avec filtres
POST /chat                    agent agentic GraphRAG (reponse non-streamee)
POST /chat/stream             agent agentic GraphRAG (Server-Sent Events)
GET  /docs                    Swagger UI
GET  /redoc                   ReDoc
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

Workflow implemente dans `src/08_agentic_graphrag/graph.py`:

1. `analyse_request`              parse de la requete, du candidat et de l'intent
2. `plan_tools`                   selection des tools utiles
3. `execute_tools`                appel des tools pgvector et/ou Neo4j
4. `build_context`                normalisation du contexte recupere
5. `generate_final_answer`        synthese finale + critique de fidelite

Tools exposes a l'agent (`src/08_agentic_graphrag/tools.py`):

```text
service_status                    diagnostic pgvector, Neo4j et modele embeddings
pgvector_semantic_search          recherche dense + lexicale sur les entites metier
pgvector_document_search          recherche vectorielle dans les referentiels indexes
neo4j_graph_query                 Text2Cypher read-only vers Neo4j
hybrid_candidate_recommendation   recommandation candidat via moteur Neo4j + pgvector
```

Test CLI direct:

```powershell
poetry run python src/08_agentic_graphrag/run_agent.py --candidat PPKOU2501080016340 --top-k 5
```

Lancement LangGraph Studio/API locale si `langgraph-cli[inmem]` est installe:

```powershell
poetry run langgraph dev
```

Le mode declare par la CLI est `real`: le mode simulation est desactive dans
ce workflow agentique.

### 9. Interface chatbot Streamlit

L'application `chatbot_app.py` expose une interface conversationnelle avec
visualisation des traces du workflow LangGraph.

```powershell
poetry run streamlit run chatbot_app.py
```

L'interface permet:

- la saisie d'un identifiant candidat (optionnel) et d'une question libre;
- le reglage du `top-k` d'offres a analyser;
- l'affichage optionnel des traces (suite des nodes et tools invoques);
- des questions exemples pre-remplies pour la demonstration.

### 10. CLI interactive de l'agent

Une CLI inspiree du template Cole Medin permet de chatter avec l'agent
en visualisant les outils invoques pour chaque reponse.

```powershell
# Demarrer l'API (terminal 1)
poetry run uvicorn src.06_api.main:app --host 0.0.0.0 --port 8000

# Demarrer la CLI (terminal 2)
poetry run python cli.py

# Connexion a une URL specifique
poetry run python cli.py --url http://localhost:8000
```

Sortie type:

```text
You: Analyse PP001 pour un poste de data analyst a Douala
Tools Used:
  1. load_candidate_profile(id='PP001')
  2. vector_search_offres(query='data analyst Douala', limit=10)
  3. graph_search_compatibles(candidat_id='PP001', secteur='Banque')
  4. compute_skill_gap(candidat_id='PP001', offre_id='OFF_42')
  5. find_formations(skills_manquants=['SQL', 'PowerBI'])
Verdict: postuler_avec_plan_de_montee_en_competence (score: 0.73)
```

Commandes:

```text
help    affiche les commandes
health  verifie la connexion API
clear   reinitialise la session
exit    quitte la CLI
```

### 11. Evaluer le systeme (avec ablation H1-H4)

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

Etude d'ablation pour valider les hypotheses H1 a H4:

```powershell
# H2 : impact du graphe (vector seul vs +graphe)
poetry run python src/07_evaluation/evaluate_system.py --agent-config vector_only
poetry run python src/07_evaluation/evaluate_system.py --agent-config plus_graph

# H4 : impact du score hybride et de la critique agentique
poetry run python src/07_evaluation/evaluate_system.py --agent-config plus_hybrid
poetry run python src/07_evaluation/evaluate_system.py --agent-config plus_critique
```

Sorties principales:

```text
outputs/evaluation/evaluation_report.json
outputs/evaluation/ablation_study.csv       tableau H1-H4 prÃªt pour LaTeX
outputs/evaluation/embedding_benchmark.csv  tableau multi-modeles
```

Point de vigilance: `evaluate_system.py` contient des chemins de secours qui
retournent des metriques simulees si certains artefacts ou modeles ne sont pas
disponibles. Pour une interpretation scientifique, utiliser en priorite les
artefacts calcules par `evaluate_st.py`, `benchmark_embeddings.py`, les logs
d'execution et le rapport JSON genere dans ton environnement.

## Ordre d'execution recommande

```powershell
cd "D:\DATA SCIENCES\SYSTEME-DE-RECOMMANDATION-HYBRIDE-"
poetry lock
poetry install

# 1. ETL et fine-tuning
poetry run python scripts/run_etl.py
poetry run python src/02_finetune_st/train_sentence_transformer.py
poetry run python src/02_finetune_st/evaluate_st.py
poetry run python src/07_evaluation/benchmark_embeddings.py --include-lexical --bootstrap 1000

# 2. Graphe et vecteurs
poetry run python src/03_knowledge_graph/load_neo4j.py --dry-run
poetry run python src/03_knowledge_graph/load_neo4j.py
poetry run python src/04_pgvector/embed_all_entities.py --dry-run
poetry run python src/04_pgvector/embed_all_entities.py

# 3. Enrichissement PDFs et similarites
poetry run python scripts/ingest_pdfs.py --pdf pdf/Nomenclature-Camerounaise-des-Formations-24.01.2017.pdf --source NCF_2017
poetry run python scripts/ingest_pdfs.py --pdf pdf/Nomenclature-camerounaise-des-metiers-_2013.pdf --source MEPC_2013
poetry run python scripts/ingest_pdfs.py --pdf pdf/diplome_certificat.pdf --source diplomes
poetry run python scripts/materialize_similarity.py --entity skills --threshold 0.75
poetry run python scripts/materialize_similarity.py --entity metiers --threshold 0.80

# 4. Test moteur classique
poetry run python src/05_graphrag/recommendation_engine.py --candidat PPKOU2501080016340 --top-k 5

# 5. API + agent + interfaces
poetry run uvicorn src.06_api.main:app --host 0.0.0.0 --port 8000 --reload
poetry run python src/08_agentic_graphrag/run_agent.py --candidat PPKOU2501080016340 --top-k 5
poetry run streamlit run chatbot_app.py     # interface UI
poetry run python cli.py                    # interface CLI

# 6. Evaluation finale
poetry run python src/07_evaluation/evaluate_system.py
```

## Sources techniques

- Poetry CLI: https://python-poetry.org/docs/cli/
- Sentence Transformers, `MultipleNegativesRankingLoss`: https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html
- Sentence Transformers, `InformationRetrievalEvaluator`: https://www.sbert.net/docs/package_reference/sentence_transformer/evaluation.html
- pgvector: https://github.com/pgvector/pgvector
- Neo4j Python driver: https://neo4j.com/docs/python-manual/current/
- Neo4j Vector Index: https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/
- Neo4j Graph Data Science (GDS): https://neo4j.com/docs/graph-data-science/current/
- FastAPI, lancement avec serveur ASGI: https://fastapi.tiangolo.com/deployment/manually/
- FastAPI Server-Sent Events: https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse
- LangGraph: https://langchain-ai.github.io/langgraph/
- LangSmith (tracing): https://docs.smith.langchain.com/
- Streamlit: https://docs.streamlit.io/
- pdfplumber (extraction PDF): https://github.com/jsvine/pdfplumber
- E5 multilingual embeddings: https://huggingface.co/intfloat/multilingual-e5-base
- Sentence-CamemBERT: https://huggingface.co/dangvantuan/sentence-camembert-base

