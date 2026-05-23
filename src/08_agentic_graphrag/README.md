# Agentic GraphRAG

Ce module ajoute une couche Agentic GraphRAG testable dans LangGraph Studio.

Workflow:

1. `analyse_request`: transforme l'objectif en plan.
2. `load_profile`: charge le candidat normalise.
3. `retrieve_and_check_graph`: appelle le GraphRAG existant pour pgvector + Neo4j.
4. `compute_skill_gap`: normalise les ecarts de competences.
5. `score_and_rank`: classe par score hybride.
6. `critique_recommendations`: bloque ou relance si les resultats sont faibles.
7. `create_roadmap`: genere une trajectoire de formation.
8. `generate_final_answer`: redige avec Ollama `llama3.1:latest` si active.

Mode par defaut: `real`. Le mode simulation est desactive: le graphe exige Neo4j et PostgreSQL/pgvector.
La generation finale est deterministe par defaut pour garder Studio reactif.

Le scoring graphe s'aligne sur le schema Neo4j actuellement charge:

- candidat -> `POSSEDE` -> `Compétence` ESCO
- offre -> `REQUIERT` -> `Compétence` ESCO
- candidat -> `A_NIVEAU` -> `NiveauFormationNCF`
- candidat -> `A_FORMATION` -> `DomaineDetailléNCF`
- offre -> `REQUIERT_NIVEAU_NCF` / `REQUIERT_NIVEAU` -> `NiveauFormationNCF`
- offre -> `DANS_SECTEUR` -> `Secteur`
- offre -> `LOCALISEE_A` -> `Localisation`

Les relations `POSSEDE` et `REQUIERT` sont creees par:

```powershell
python src/03_knowledge_graph/load_neo4j.py --step skills
```

Ces liens sont des alignements lexicaux vers ESCO. Ils portent `matchMethod`, `sourceText`, `sourceField` et `confidence` pour rester auditables.
Pour utiliser Ollama:

```powershell
$env:AGENT_USE_OLLAMA = "1"
$env:OLLAMA_MODEL = "llama3.1:latest"
```

Pour tracer les executions dans LangSmith, mets ces variables dans `.env` ou dans ta session PowerShell:

```powershell
$env:LANGSMITH_TRACING = "true"
$env:LANGSMITH_API_KEY = "lsv2_..."
$env:LANGSMITH_PROJECT = "agentic-graphrag-recommandation"
$env:LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
```

Ne versionne jamais ta vraie cle API. Le fichier `.env` est ignore par Git; `.env.example` ne contient qu'un modele.

Configuration minimale des bases dans `.env`:

```powershell
$env:PG_HOST = "localhost"
$env:PG_PORT = "5432"
$env:PG_DB = "test_kmer"
$env:PG_USER = "postgres"
$env:PG_PASSWORD = "..."
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "..."
$env:NEO4J_DATABASE = "neo4j"
```

Test CLI:

```powershell
python src/08_agentic_graphrag/run_agent.py --candidat "<ID_CANDIDAT>"
```

Dans LangGraph Studio local, utilise le lien affiche par `langgraph dev`:

```text
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

Le bouton `Deployer` sert au deploiement LangSmith cloud. Pour tester le graphe
localement, utilise `Interagir`, `Chat` ou `Executer l'experience`; ne clique
pas sur `Deployer` tant que tu veux seulement executer le serveur local.

Entree minimale LangGraph Studio:

```json
{
  "message": "Recommande les meilleures offres pour le candidat PPKOU2501080016340, explique les gaps de competences et propose une roadmap."
}
```

Question d'orientation sans identifiant candidat:

```json
{
  "message": "je veux devenir data scientist dans le domaine bancaire au cameroun quelles competences je dois mettre en avant ?"
}
```

Dans ce cas, le graphe bascule vers l'intention `career_advice`: il ne force pas
un faux candidat `AUTO`, ne charge pas le premier profil disponible, et produit
une orientation a partir des offres locales normalisees et des libelles ESCO.

Aliases acceptes pour le texte utilisateur: `message`, `message_humain`,
`question`, `input` ou `user_query`. L'ancien format avec `HumanMessage` reste
compatible:

```json
{
  "messages": [
    {
      "type": "human",
      "content": "Recommande les meilleures offres pour le candidat PPKOU2501080016340, explique les gaps de competences et propose une roadmap."
    }
  ],
  "candidat_id": "PPKOU2501080016340",
  "top_k": 5,
  "mode": "real"
}
```
