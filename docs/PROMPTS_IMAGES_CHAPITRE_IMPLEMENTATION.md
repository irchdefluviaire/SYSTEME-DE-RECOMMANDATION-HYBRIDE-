# Prompts d'images explicatives pour le chapitre Implémentation

Objectif : produire des images explicatives lisibles pour le mémoire, sans TikZ.
Ces images doivent compléter les graphiques statistiques déjà générés par le
notebook `notebooks/11_Analyses_Implementation_Approfondies.ipynb`.

Les figures statistiques déjà disponibles sont dans :

```text
rapport/figures/generated/implementation_deep/
```

## Recommandation de style commun

Pour toutes les images générées par IA, demander :

- format horizontal 16:9 ;
- style schéma académique propre ;
- fond blanc ou très clair ;
- couleurs sobres : bleu, vert, orange, gris ;
- typographie lisible ;
- texte en français ;
- pas de mascotte, pas de décor inutile ;
- pas d'effets 3D excessifs ;
- pas de pseudo-code illisible ;
- pas de logo inventé.

## Image 1 - Vue d'ensemble du système

### Emplacement conseillé

Début du chapitre, juste après l'introduction de la section :

```latex
\section{Vue d'ensemble de l'architecture implémentée}
```

### Rôle

Donner une vue globale du système sans entrer dans les détails techniques.
Cette image doit permettre à un lecteur non informaticien de comprendre le
chemin : données -> stockage -> recommandation -> interface.

### Prompt à donner à l'IA

```text
Créer un schéma académique horizontal 16:9 en français, fond blanc, style propre et professionnel, représentant l'architecture d'un système de recommandation emploi-compétences. Montrer une chaîne de blocs reliés par des flèches : 1) Données sources : offres d'emploi, profils candidats, référentiels métiers et formations ; 2) Préparation et normalisation des données ; 3) Représentation sémantique par embeddings ; 4) Base vectorielle pgvector ; 5) Graphe de connaissances Neo4j ; 6) Moteur GraphRAG et score hybride ; 7) Orchestration agentique LangGraph ; 8) Réponse finale via chatbot, API et CLI. Utiliser des icônes simples de base de données, graphe, loupe, robot et interface utilisateur. Texte lisible, couleurs sobres bleu, vert, orange et gris. Ne pas utiliser de code, ne pas utiliser de personnages, ne pas utiliser de décoration inutile.
```

## Image 2 - Chaîne de préparation des données

### Emplacement conseillé

Dans la section :

```latex
\section{Implémentation de la chaîne ETL}
```

Après le paragraphe qui explique la normalisation des offres et candidats.

### Rôle

Expliquer simplement comment les données brutes deviennent des tables
normalisées utilisables par le système.

### Prompt à donner à l'IA

```text
Créer une infographie académique horizontale en français montrant le pipeline de préparation des données pour un système emploi-compétences. À gauche : fichiers bruts avec offres d'emploi, candidats, référentiels MEPC, NCF et ESCO. Au centre : nettoyage, audit des doublons potentiels sans suppression, normalisation des secteurs, villes, diplômes, niveaux NCF et compétences. À droite : données normalisées avec quatre sorties : offres normalisées, candidats normalisés, référentiels alignés, paires offre-profil pour entraînement. Style sobre, fond clair, flèches simples, blocs rectangulaires, icônes minimalistes de table, balai de nettoyage, dictionnaire, fichier de sortie. Texte en français lisible.
```

## Image 3 - Différence entre pgvector et Neo4j

### Emplacement conseillé

Entre les sections :

```latex
\section{Implémentation de la base vectorielle pgvector}
\section{Implémentation du graphe de connaissances Neo4j}
```

### Rôle

Faire comprendre que pgvector et Neo4j ne jouent pas le même rôle :
pgvector mesure la proximité sémantique, Neo4j structure les relations.

### Prompt à donner à l'IA

```text
Créer un schéma comparatif en deux colonnes, en français, intitulé "Deux mémoires complémentaires du système". Colonne gauche : pgvector, représenter un nuage de points ou vecteurs avec une loupe, texte : recherche sémantique, similarité entre profils et offres, embeddings, top-k. Colonne droite : Neo4j, représenter un graphe de nœuds reliés, texte : relations métier-compétence, niveau NCF, secteur, skill gap, explicabilité. Au centre en bas : moteur hybride qui combine les deux sources. Style mémoire académique, fond blanc, couleurs bleu pour pgvector et orange/vert pour Neo4j, texte très lisible, pas d'éléments décoratifs.
```

## Image 4 - Fonctionnement du score hybride

### Emplacement conseillé

Dans la section :

```latex
\section{Implémentation du moteur GraphRAG}
```

Après le paragraphe qui présente le score hybride.

### Rôle

Montrer que la recommandation ne repose pas seulement sur une similarité
textuelle, mais sur plusieurs signaux.

### Prompt à donner à l'IA

```text
Créer une infographie claire en français expliquant le score hybride d'un système de recommandation emploi-compétences. Montrer quatre jauges ou composantes entrant dans un score final : similarité sémantique, couverture des compétences, compatibilité du niveau NCF, alignement secteur/métier. Les quatre composantes convergent vers un bloc "Score hybride" puis vers trois verdicts possibles : prêt à postuler, montée en compétence, vivier à développer. Style académique, fond clair, couleurs sobres, aucune formule compliquée, texte lisible, composition horizontale 16:9.
```

## Image 5 - Workflow agentique LangGraph

### Emplacement conseillé

Dans la section :

```latex
\section{Implémentation du workflow agentique LangGraph}
```

À insérer avant ou après l'explication des étapes `analyse_request`,
`plan_tools`, `execute_tools`, `build_context`, `generate_final_answer`,
`answer_critic`.

### Rôle

Expliquer visuellement le comportement agentique sans obliger le lecteur à lire
le code.

### Prompt à donner à l'IA

```text
Créer un diagramme de workflow agentique en français pour un chatbot emploi-compétences. Montrer les étapes suivantes reliées par des flèches : Question utilisateur -> Analyse de l'intention -> Choix des outils -> Exécution des outils pgvector, Neo4j, documents -> Construction du contexte -> Génération de la réponse -> Critic de fidélité. Ajouter une boucle de révision : si la réponse est insuffisante, élargir le contexte puis réexécuter les outils. Style diagramme professionnel, fond blanc, blocs arrondis discrets, icônes simples, texte lisible, couleurs sobres violet pour agent, bleu pour données, vert pour validation. Ne pas utiliser de code.
```

## Image 6 - Architecture des interfaces

### Emplacement conseillé

Dans la section :

```latex
\section{API, CLI et interface conversationnelle}
```

### Rôle

Montrer que le même moteur peut être utilisé par plusieurs interfaces.

### Prompt à donner à l'IA

```text
Créer un schéma académique en français montrant trois interfaces connectées au même moteur de recommandation : CLI pour les tests, Streamlit pour le chatbot local, FastAPI pour l'intégration web. Les trois interfaces pointent vers un bloc central "Moteur Agentic GraphRAG", lui-même connecté à pgvector, Neo4j et OpenRouter. Style sobre, fond clair, icônes terminal, fenêtre chat, API web, bases de données. Texte lisible, pas de décor inutile.
```

## Images statistiques déjà prêtes à insérer

Ces images ne doivent pas être générées par IA : elles proviennent des données
réelles du projet.

### À insérer dans la section "Analyse statistique descriptive du corpus"

```text
figures/generated/implementation_deep/03_offres_par_secteur.png
figures/generated/implementation_deep/04_offres_par_ville.png
figures/generated/implementation_deep/05_ncf_et_experience_offres.png
figures/generated/implementation_deep/06_structure_candidats.png
figures/generated/implementation_deep/07_top_competences_offres.png
figures/generated/implementation_deep/09_equilibre_offres_candidats_secteurs.png
```

### À insérer dans la section "Implémentation de pgvector et Neo4j"

```text
figures/generated/implementation_deep/11_pgvector_counts.png
figures/generated/implementation_deep/11_neo4j_node_counts.png
```

### À insérer dans une section "Qualité et disponibilité des données"

```text
figures/generated/implementation_deep/02_qualite_champs_cles.png
figures/generated/implementation_deep/08_longueur_text_to_embed.png
```

## Attention méthodologique

Dans le mémoire, ne pas dire "le script Python a produit". Dire plutôt :

```text
L'analyse descriptive du corpus montre que...
```

ou :

```text
Les données normalisées se composent de...
```

Le lecteur n'a pas besoin de connaître le fichier Python. Il doit comprendre :

- quelles données ont été analysées ;
- ce que montre la figure ;
- pourquoi cela compte pour le système de recommandation.
