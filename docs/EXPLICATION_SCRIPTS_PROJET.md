# Explication des scripts du projet

Ce document decrit le role des scripts et notebooks du projet
`SYSTEME-DE-RECOMMANDATION-HYBRIDE-`. Il se concentre sur les fichiers de code,
de configuration et d'execution. Les fichiers de donnees, les sorties generees
et les caches ne sont pas des scripts metier.

## Vue d'ensemble

Le projet est organise comme une chaine complete :

```text
donnees brutes
-> ETL et normalisation
-> paires de fine-tuning
-> modele SentenceTransformer
-> embeddings pgvector
-> graphe Neo4j
-> moteur GraphRAG
-> API / CLI / Streamlit
-> evaluation
-> memoire
```

La logique generale est la suivante :

- `src/01_etl` prepare les donnees.
- `src/02_finetune_st` entraine et evalue le modele d'embeddings.
- `src/03_knowledge_graph` construit le graphe Neo4j.
- `src/04_pgvector` construit et interroge la base vectorielle.
- `src/05_graphrag` implemente le moteur GraphRAG.
- `src/06_api` expose le systeme par FastAPI.
- `src/07_evaluation` calcule les metriques.
- `src/08_agentic_graphrag` orchestre le chatbot agentique LangGraph.
- `notebooks` documente et execute les etapes analytiques.

## Configuration generale

### `config.py`

Fichier central de configuration ETL.

Il definit :

- les chemins vers `data/raw`, `data/processed` et `data/finetune` ;
- les fichiers sources des offres, candidats, MEPC, NCF et ESCO ;
- les fichiers de sortie Parquet et JSONL ;
- les mappings de niveaux d'etudes vers niveaux NCF ;
- les mappings de diplomes, villes, contrats, experience ;
- les constantes de construction des textes d'embeddings.

Il sert de socle commun aux scripts d'ETL. Si un chemin ou une regle de
normalisation change, c'est souvent ici qu'il faut intervenir.

### `pyproject.toml`

Fichier de configuration Poetry.

Il decrit :

- le nom du projet ;
- les dependances Python ;
- les outils de developpement ;
- les versions attendues des bibliotheques.

C'est le fichier qui permet a Poetry de recreer l'environnement d'execution.

### `langgraph.json`

Configuration de LangGraph Studio / LangGraph API.

Il indique quel graphe charger :

```text
agentic_graphrag -> src/08_agentic_graphrag/graph.py:graph
```

Il sert lorsque tu lances LangGraph en mode serveur ou Studio.

## Scripts racine

### `chatbot_app.py`

Interface Streamlit du chatbot.

Role :

- charge le graphe LangGraph ;
- affiche une interface conversationnelle ;
- recupere la question utilisateur ;
- transmet la question a `graph.invoke(...)` ;
- affiche la reponse, les traces du workflow et le resultat du critic.

Chemin reel d'appel :

```text
Streamlit
-> src/08_agentic_graphrag/graph.py
-> src/08_agentic_graphrag/tools.py
-> pgvector / Neo4j / OpenRouter
```

Important : ce chatbot n'a pas besoin que FastAPI soit lancee. Il appelle
directement le graphe LangGraph.

## Dossier `scripts`

### `scripts/run_etl.py`

Orchestrateur principal de l'ETL.

Il execute dans l'ordre :

1. normalisation des offres ;
2. normalisation des candidats ;
3. alignement des referentiels MEPC, NCF et ESCO ;
4. construction des paires de fine-tuning.

Sorties principales :

- `data/processed/offres_normalized.parquet`
- `data/processed/candidats_normalized.parquet`
- `data/processed/mapping_isco_mepc_esco.parquet`
- `data/finetune/pairs_train.jsonl`
- `data/finetune/pairs_val.jsonl`
- `data/finetune/pairs_test.jsonl`

Ce script est a utiliser pour reconstruire toute la base propre a partir des
donnees brutes.

### `scripts/ingest_pdfs.py`

Pipeline d'ingestion des PDF referentiels ou reglementaires.

Il realise :

- extraction texte par `pdfplumber`, avec fallback `PyPDF2` ;
- OCR optionnel ;
- chunking structural ou semantique ;
- generation d'embeddings ;
- sauvegarde JSONL locale ;
- insertion des chunks dans pgvector ;
- creation de noeuds `DocChunk` dans Neo4j ;
- liaison des chunks aux noeuds NCF ou MEPC lorsque des codes sont detectes.

Ce script permet d'alimenter la partie RAG documentaire du systeme.

### `scripts/materialize_similarity.py`

Script de materialisation des similarites dans Neo4j.

Il part des embeddings stockes dans pgvector, cherche les voisins les plus
proches d'une entite, puis cree des relations `SIMILAIRE_A` dans Neo4j.

Entites supportees :

- competences ;
- metiers.

Objectif : transformer une proximite vectorielle en relation explicite dans le
graphe. Cela rend certaines similarites exploitables par Cypher.

### `scripts/setup_project_structure.py`

Script de creation de l'arborescence projet.

Il sert a generer les dossiers de base du projet lorsque l'environnement est
initialise. Il n'est pas au coeur de la recommandation ; c'est un script de
scaffolding.

### `scripts/start_chatbot_poetry.ps1`

Script PowerShell de lancement du chatbot Streamlit avec Poetry.

Il sert a demarrer l'interface utilisateur sans retaper la commande complete.

## Dossier `src/00_Social_media`

### `src/00_Social_media/linkedin.py`

Script lie a la collecte ou au traitement de donnees LinkedIn.

Son role est en amont du systeme principal : recuperer ou preparer des donnees
issues de reseaux sociaux professionnels. Il ne constitue pas le coeur du
moteur GraphRAG, mais peut enrichir les donnees sources.

## Dossier `src/01_etl`

### `src/01_etl/utils.py`

Bibliotheque utilitaire de l'ETL.

Elle contient :

- nettoyage des espaces ;
- suppression de bruit dans les annonces ;
- normalisation des villes ;
- normalisation des secteurs ;
- normalisation des competences ;
- generation d'identifiants ;
- profilage qualite ;
- fonctions de log.

Ce fichier evite de dupliquer les memes fonctions dans les scripts d'ETL.

### `src/01_etl/normalize_offres.py`

Pipeline de normalisation des offres d'emploi.

Il effectue :

- chargement du fichier brut des offres ;
- nettoyage texte ;
- deduplication ;
- nettoyage de la description d'annonce ;
- normalisation des villes, secteurs, competences ;
- mapping du niveau d'etudes vers NCF ;
- mapping de l'experience ;
- normalisation des contrats ;
- generation d'un identifiant stable d'offre ;
- construction de `text_to_embed` ;
- export en Parquet ;
- export d'un rapport qualite.

Ce script produit la table offre propre utilisee par pgvector, Neo4j et le
fine-tuning.

### `src/01_etl/normalize_candidats.py`

Pipeline de normalisation des profils candidats.

Il realise :

- chargement du fichier brut des demandeurs ;
- nettoyage des colonnes ;
- mapping des diplomes et niveaux vers la NCF ;
- normalisation de la mobilite geographique ;
- traitement des valeurs non declarees ;
- generation d'un schema final propre ;
- construction de `text_to_embed` candidat ;
- export en Parquet.

Ce script produit la table candidat exploitee pour le matching.

### `src/01_etl/align_referentiels.py`

Script de chargement et d'alignement des referentiels.

Il charge :

- MEPC ;
- NCF ;
- ESCO.

Puis il construit des correspondances partielles entre les nomenclatures,
notamment entre groupes metiers locaux et entites ESCO/ISCO.

Objectif : eviter que le systeme reste prisonnier des libelles bruts des offres.

### `src/01_etl/build_pairs.py`

Script de construction des paires pour le fine-tuning SentenceTransformer.

Il part des offres normalisees et cree des paires :

```text
sentence1 = metadonnees structurees
sentence2 = competences + description de l'offre
```

Il produit :

- `pairs_train.jsonl`
- `pairs_val.jsonl`
- `pairs_test.jsonl`
- `pairs_metadata.json`

Ces paires sont utilisees pour apprendre un espace vectoriel adapte au domaine
emploi-competences.

## Dossier `src/02_finetune_st`

### `src/02_finetune_st/config_st.json`

Configuration du fine-tuning SentenceTransformer.

Il contient les hyperparametres et chemins utilises par l'entrainement :

- modele de base ;
- batch size ;
- learning rate ;
- nombre d'epochs ;
- chemins de donnees ;
- chemin de sauvegarde.

### `src/02_finetune_st/train_sentence_transformer.py`

Script d'entrainement du SentenceTransformer.

Il :

- charge les paires JSONL ;
- transforme les paires en dataset ;
- construit les evaluateurs IR ;
- charge le modele de base ;
- entraine le modele sur les paires offre-profil ;
- sauvegarde le modele fine-tune ;
- peut fonctionner en mode `eval_only`.

Ce script est responsable de l'adaptation du modele d'embeddings au vocabulaire
emploi-competences.

### `src/02_finetune_st/evaluate_st.py`

Script d'evaluation du SentenceTransformer.

Il compare le modele fine-tune a un modele de base a partir des paires de test.

Metriques :

- Recall@K ;
- NDCG@K ;
- MRR@K ;
- precision.

Il sert a verifier si le fine-tuning a vraiment ameliore le retrieval.

## Dossier `src/03_knowledge_graph`

### `src/03_knowledge_graph/config_neo4j.py`

Configuration Neo4j.

Il centralise les variables de connexion :

- URI ;
- utilisateur ;
- mot de passe ;
- base ;
- chemins de donnees.

### `src/03_knowledge_graph/schema.cypher`

Schema Cypher du graphe.

Il declare les contraintes et index Neo4j pour les noeuds importants :

- offres ;
- candidats ;
- competences ;
- metiers ;
- secteurs ;
- niveaux NCF ;
- entites MEPC.

Ce fichier garantit l'unicite et accelere les requetes.

### `src/03_knowledge_graph/queries_cypher.py`

Bibliotheque de requetes Cypher.

Elle regroupe les requetes recurrentes pour :

- recommander ;
- explorer le graphe ;
- calculer des relations ;
- recuperer des profils ou offres.

Elle evite de disperser les requetes Cypher dans plusieurs scripts.

### `src/03_knowledge_graph/load_neo4j.py`

Script principal de chargement Neo4j.

Il charge dans le graphe :

- competences ESCO ;
- occupations ESCO ;
- hierarchies ISCO ;
- referentiel MEPC ;
- referentiel NCF ;
- offres ;
- candidats ;
- relations complementaires ;
- liens entre entites et competences.

Il contient aussi :

- creation du schema ;
- batch merge ;
- matching lexical de competences ;
- validation du graphe ;
- options `step`, `dry_run`, `clear`.

C'est l'un des scripts centraux du projet : il transforme les donnees tabulaires
en graphe de connaissances.

## Dossier `src/04_pgvector`

### `src/04_pgvector/config_pgvector.py`

Configuration PostgreSQL/pgvector.

Il centralise les informations de connexion et les chemins du module vectoriel.

### `src/04_pgvector/schema_pgvector.sql`

Schema SQL de la base pgvector.

Il cree :

- l'extension `vector` ;
- les types d'entites ;
- la table `embeddings` ;
- les index HNSW ;
- les tables ou index utiles au retrieval.

### `src/04_pgvector/embed_all_entities.py`

Script d'encodage et d'insertion des embeddings.

Il encode :

- offres ;
- candidats ;
- competences ESCO ;
- metiers ESCO ;
- domaines NCF ;
- groupes MEPC.

Il insere ensuite les vecteurs dans PostgreSQL/pgvector.

Il contient aussi des fonctions de validation et de recherche ANN.

### `src/04_pgvector/ann_search.py`

Script de recherche vectorielle ANN.

Il fournit des fonctions pour :

- trouver les offres proches d'un candidat ;
- trouver les competences proches d'une competence ;
- proposer des metiers a partir d'un candidat ;
- chercher depuis un texte libre ;
- evaluer le recall ANN ;
- mesurer la latence.

### `src/04_pgvector/hybrid_search.py`

Script de recherche hybride dense + lexicale.

Il combine :

- recherche dense vectorielle ;
- recherche lexicale PostgreSQL full-text ;
- fusion par Reciprocal Rank Fusion.

Il est utilise par l'agent pour interroger les entites metier, competence,
offre, candidat et par le moteur candidat -> offres.

## Dossier `src/05_graphrag`

### `src/05_graphrag/context_builder.py`

Constructeur de contexte GraphRAG.

Il orchestre :

1. recherche pgvector des offres candidates ;
2. enrichissement Neo4j ;
3. calcul du skill gap ;
4. calcul du score hybride ;
5. construction d'un texte de contexte pour le LLM.

C'est la passerelle entre la recuperation de donnees et la generation.

### `src/05_graphrag/recommendation_engine.py`

Moteur principal de recommandation.

Il :

- charge le profil candidat ;
- appelle `GraphRAGContextBuilder` ;
- appelle OpenRouter pour generer une recommandation JSON ;
- genere une analyse de skill gap ;
- genere une roadmap ;
- sauvegarde les resultats dans PostgreSQL si la connexion est disponible.

Important : le LLM ne cherche pas les donnees lui-meme. Il recoit un contexte
structure construit par pgvector et Neo4j.

### `src/05_graphrag/text2cypher.py`

Module Text2Cypher.

Il transforme une question en langage naturel en requete Cypher controlee.

Il contient :

- le schema Neo4j donne au modele ;
- la validation read-only ;
- l'appel au modele Hugging Face Text2Cypher ;
- des templates fallback si le modele HF est indisponible ;
- l'execution securisee contre Neo4j.

Il intervient surtout pour les requetes de type `graph_query`.

### `src/05_graphrag/document_retriever.py`

Retriever documentaire.

Il cherche dans la table `doc_chunks` de pgvector et remonte les documents
parents ou chunks pertinents.

Il sert aux questions referentielles et aux questions qui demandent des sources
documentaires.

### `src/05_graphrag/prompt_templates.py`

Fichier de prompts.

Il contient :

- prompts systeme ;
- prompts utilisateur ;
- formats de sortie attendus ;
- templates pour recommandation, skill gap et roadmap ;
- fonctions de formatage OpenAI/OpenRouter.

C'est ici que le comportement redactionnel du LLM est cadre.

### `src/05_graphrag/answer_critic.py`

Critic local de reponse.

Il mesure de maniere heuristique :

- fidelite au contexte ;
- couverture du contexte ;
- termes non supportes ;
- decision `accept` ou `revise`.

Il permet au workflow agentique de relancer une recuperation lorsque la reponse
est insuffisamment fondee.

### `src/05_graphrag/roadmap_generator.py`

Generateur de roadmap de montee en competences.

Il associe les competences manquantes a des domaines ou formations NCF lorsque
c'est possible, puis produit une trajectoire de progression.

## Dossier `src/06_api`

### `src/06_api/main.py`

Point d'entree FastAPI.

Il :

- cree l'application ;
- configure CORS ;
- initialise les services au demarrage ;
- declare les routes ;
- expose un endpoint de sante.

Cette couche sert a integrer le systeme dans une application externe ou un site
web.

### `src/06_api/dependencies.py`

Gestionnaire de dependances FastAPI.

Il initialise et partage :

- modele SentenceTransformer ;
- connexion Neo4j ;
- connexion PostgreSQL/pgvector ;
- moteur de recommandation.

Il evite de recharger les modeles et connexions a chaque requete.

### `src/06_api/schemas.py`

Schemas Pydantic de l'API.

Il definit les formats d'entree et de sortie :

- profil candidat ;
- requete de recommandation ;
- requete skill gap ;
- requete embedding ;
- requete chat ;
- reponses structurees.

### `src/06_api/offre.py`

Routes ou helpers lies aux offres.

Il charge les offres normalisees et fournit des fonctions pour serialiser les
listes de valeurs.

### `src/06_api/routers/recommend.py`

Endpoint de recommandation.

Il appelle le moteur de recommandation et renvoie une reponse structuree API.

### `src/06_api/routers/skill_gap.py`

Endpoint d'analyse de skill gap.

Il exploite les sorties du moteur pour exposer les competences acquises,
manquantes et la roadmap.

### `src/06_api/routers/embed.py`

Endpoint d'embedding.

Il encode un texte avec le modele SentenceTransformer charge en memoire.

### `src/06_api/routers/chat.py`

Endpoint chat Agentic GraphRAG.

Il appelle le graphe LangGraph :

```text
requete HTTP
-> graph.invoke(...)
-> outils
-> reponse + traces + critic
```

Il propose aussi une version streaming Server-Sent Events.

## Dossier `src/07_evaluation`

### `src/07_evaluation/eval_retrieval.py`

Fonctions de metriques IR.

Il calcule :

- Precision@K ;
- Recall@K ;
- NDCG@K ;
- MRR@K ;
- intervalles de confiance bootstrap.

### `src/07_evaluation/eval_embedding.py`

Abstractions pour evaluer differents modeles d'embeddings.

Il implemente :

- interface commune `EmbeddingModel` ;
- wrapper SentenceTransformer ;
- modele TF-IDF ;
- modele BM25.

Il permet de comparer des representations denses et lexicales.

### `src/07_evaluation/benchmark_embeddings.py`

Benchmark des modeles d'embeddings.

Il charge les paires de test, evalue plusieurs modeles et exporte :

- CSV ;
- JSON ;
- graphique comparatif.

Sorties typiques :

- `outputs/evaluation/embedding_benchmark.csv`
- `outputs/evaluation/embedding_benchmark.json`
- `outputs/evaluation/embedding_benchmark_plot.png`

### `src/07_evaluation/evaluate_system.py`

Ancien orchestrateur d'evaluation globale.

Il couvre :

- evaluation du SentenceTransformer ;
- evaluation GraphRAG ;
- analyse des scores hybrides ;
- latence ;
- rapport JSON.

Attention : certaines sections utilisent ou mentionnent des simulations. Pour
le memoire, il faut privilegier les resultats effectivement executes et
documentes.

### `src/07_evaluation/evaluate_graphrag.py`

Evaluation legere GraphRAG.

Il calcule :

- recall de contexte ;
- proxy de fidelite ;
- proxy de correction de reponse.

Il sert a tester la coherence entre contexte recupere et reponse.

### `src/07_evaluation/eval_faithfulness.py`

Metriques heuristiques de fidelite.

Il calcule :

- recouvrement de tokens ;
- recouvrement de bigrammes ;
- fidelite par mots-cles ;
- qualite d'une roadmap ;
- score composite.

### `src/07_evaluation/llm_as_judge.py`

Juge LLM ou heuristique.

Il permet de noter une reponse selon des criteres comme :

- pertinence ;
- fidelite ;
- couverture ;
- coherence ;
- citations.

Il peut servir a une evaluation qualitative plus avancee.

### `src/07_evaluation/ablation_pgvector_graph.py`

Module d'ablation recent.

Il compare :

- `pgvector_only` ;
- `pgvector_plus_graph`.

Le protocole :

1. tire un echantillon de candidats ;
2. recupere un pool d'offres avec pgvector ;
3. conserve le classement vectoriel ;
4. enrichit le meme pool avec Neo4j ;
5. rerange par score hybride ;
6. calcule Precision@K, Recall@K, NDCG@K, MRR@K ;
7. exporte CSV, JSON et figures.

Ce fichier sert directement au chapitre evaluation.

### `src/07_evaluation/models_to_benchmark.json`

Configuration des modeles a comparer dans le benchmark d'embeddings.

### `src/07_evaluation/evaluation_report.json`

Rapport JSON genere par une evaluation anterieure.

Ce n'est pas un script, mais un artefact de resultat.

## Dossier `src/08_agentic_graphrag`

### `src/08_agentic_graphrag/graph.py`

Workflow LangGraph du chatbot agentique.

Noeuds principaux :

- `analyse_request` : detecte l'intention ;
- `plan_tools` : choisit les outils ;
- `execute_tools` : execute pgvector, Neo4j, diagnostic, etc. ;
- `build_context` : normalise les resultats ;
- `generate_final_answer` : appelle OpenRouter ou fallback ;
- `answer_critic` : evalue la reponse ;
- `expand_context` : relance une recherche si la reponse doit etre revisee.

C'est le coeur de l'orchestration agentique.

### `src/08_agentic_graphrag/tools.py`

Registre des outils agentiques.

Il expose :

- `service_status` ;
- `pgvector_semantic_search` ;
- `pgvector_document_search` ;
- `neo4j_graph_query` ;
- `hybrid_candidate_recommendation` ;
- `global_graph_summary`.

Il gere aussi les connexions partagees :

- PostgreSQL/pgvector ;
- Neo4j ;
- SentenceTransformer.

### `src/08_agentic_graphrag/llm_client.py`

Client OpenRouter.

Il appelle le modele configure dans `.env`, notamment :

```text
OPENROUTER_MODEL=openai/gpt-oss-20b:free
```

Il gere :

- cle API ;
- base URL ;
- timeout ;
- retries ;
- extraction de la reponse.

### `src/08_agentic_graphrag/run_agent.py`

CLI de test du graphe agentique.

Elle permet d'envoyer une question depuis le terminal et d'obtenir :

- la reponse ;
- le use case detecte ;
- les traces ;
- le critic ;
- les resultats bruts en mode JSON.

### `src/08_agentic_graphrag/__init__.py`

Fichier d'initialisation de package.

Il rend le dossier importable comme module Python.

## Notebooks

### `notebooks/00_Social_Medias_datasets.ipynb`

Notebook d'exploration ou preparation de donnees issues de reseaux sociaux.

Il se situe en amont du pipeline principal.

### `notebooks/01_EDA_Preprocessing.ipynb`

Notebook d'analyse exploratoire et de preprocessing.

Il sert a comprendre les donnees brutes, leurs colonnes, leurs valeurs
manquantes et les transformations necessaires.

### `notebooks/02_Finetuning_SentenceTransformer.ipynb`

Notebook de fine-tuning du SentenceTransformer.

Il documente la preparation des paires, l'entrainement et les premieres
evaluations du modele d'embeddings.

### `notebooks/03_KnowledgeGraph_Neo4j.ipynb`

Notebook de construction ou verification du graphe Neo4j.

Il sert a tester les chargements, requetes Cypher et visualisations du graphe.

### `notebooks/04_pgvector_Embeddings.ipynb`

Notebook lie a l'encodage et a l'interrogation pgvector.

Il documente l'insertion des embeddings et les tests de recherche vectorielle.

### `notebooks/05_GraphRAG_Recommandation.ipynb`

Notebook de demonstration du moteur GraphRAG.

Il montre comment les resultats pgvector et Neo4j sont combines pour produire
une recommandation.

### `notebooks/06_API_FastAPI.ipynb`

Notebook de test ou documentation de l'API FastAPI.

Il sert a verifier les endpoints et les payloads.

### `notebooks/07_Evaluation.ipynb`

Notebook d'evaluation initiale.

Il regroupe les tests de performance, retrieval et qualite de reponse avant le
notebook d'ablation plus recent.

### `notebooks/09_Chapitre3_Implementation_Stats.ipynb`

Notebook genere pour le chapitre implementation.

Il produit :

- statistiques sur les offres ;
- statistiques sur les candidats ;
- figures de repartition ;
- comptages pgvector ;
- comptages Neo4j disponibles ;
- resume JSON pour le memoire.

### `notebooks/10_Evaluation_pgvector_vs_graph.ipynb`

Notebook d'evaluation pour le chapitre performance.

Il execute `ablation_pgvector_graph.py` et produit la comparaison :

```text
pgvector seul vs pgvector + Neo4j
```

Il exporte les resultats dans :

- `outputs/evaluation/pgvector_vs_graph`
- `rapport/figures/generated/evaluation`

## Comment expliquer ces scripts dans le memoire

Dans le chapitre implementation, il ne faut pas lister brutalement tous les
fichiers. Il faut les regrouper par brique :

1. ETL : `src/01_etl` + `scripts/run_etl.py`
2. Embeddings : `src/02_finetune_st` + `src/04_pgvector`
3. Graphe : `src/03_knowledge_graph`
4. GraphRAG : `src/05_graphrag`
5. Agentic workflow : `src/08_agentic_graphrag`
6. Interfaces : `chatbot_app.py` + `src/06_api`
7. Evaluation : `src/07_evaluation` + notebooks 09 et 10

Le point a defendre est simple : chaque script correspond a une responsabilite
precise. Le projet n'est pas seulement un chatbot ; c'est une chaine complete
de donnees, retrieval, graphe, orchestration, generation et evaluation.
