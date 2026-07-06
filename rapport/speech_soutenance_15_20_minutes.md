# Script de soutenance - Systeme hybride emploi-competences

**Duree cible : 20 min, dont 4 min de demonstration**

Le texte ci-dessous est concu pour etre prononce naturellement. Les diapositives de transition doivent etre passees rapidement. Les formulations entre crochets sont des consignes et ne doivent pas etre lues.

## Diapositive 1 - Page de garde (25 s)

Monsieur le President du jury, Mesdames et Messieurs les membres du jury, bonjour. Je suis NGOULOU NGOUBILI Irch Defluviaire, eleve ingenieur statisticien economiste, option Data Science et Marketing. Mon travail porte sur la conception d'un systeme de recommandation hybride pour le matching emploi-competences au Cameroun, combinant modeles de langage et graphes de connaissances. L'objectif est de recommander des offres, mais aussi d'expliquer les correspondances et les competences manquantes.

## Diapositive 2 - Plan (20 s)

Ma presentation suit cinq temps. Je partirai du contexte et de la problematique, puis je presenterai les concepts et les travaux qui fondent l'etude. J'exposerai ensuite les donnees et la methodologie, avant de montrer l'implementation et les principaux resultats. Je terminerai par les limites, les perspectives et une demonstration du systeme.

## Diapositive 3 - Transition : introduction (5 s)

Je commence par le contexte, la problematique, les objectifs et les hypotheses de l'etude.

## Diapositive 4 - Contexte (35 s)

Le taux de chomage des jeunes, pris seul, donne une lecture incomplete du marche du travail camerounais. Les estimations OIT diffusees par la Banque mondiale donnent 6,5 % en 2025. Mais 23,2 % des jeunes etaient ni en emploi, ni en etudes, ni en formation en 2021, et l'emploi vulnerable representait 67,8 % de l'emploi total en 2024. Ces indicateurs decrivent des dimensions differentes, mais montrent ensemble une insertion fragile. KmerAI cherche donc a mieux connecter les talents aux opportunites a partir d'outils d'intelligence artificielle adaptes au contexte local.

## Diapositive 5 - Problematique (35 s)

Le probleme ne vient pas uniquement du manque d'offres. Les candidats et les recruteurs decrivent parfois une meme competence avec des termes differents. Les moteurs bases sur des mots-cles detectent mal les synonymes, les competences implicites et les equivalences entre formations. Ils expliquent egalement peu leurs recommandations. La question centrale devient donc : comment combiner la comprehension semantique des textes et les relations entre candidats, competences, metiers, formations et offres pour mesurer le skill gap et proposer une trajectoire de montee en competences ?

## Diapositive 6 - Objectifs (30 s)

L'objectif general est de concevoir et d'evaluer un systeme hybride d'aide au recrutement et a la montee en competences. Quatre objectifs operationnels en decoulent : normaliser les profils et les offres avec les referentiels ; adapter un modele d'embeddings au vocabulaire de l'emploi ; construire un graphe reliant candidats, offres et competences ; enfin, orchestrer et evaluer l'ensemble du pipeline pour produire des recommandations classees et explicables.

## Diapositive 7 - Hypotheses (25 s)

Trois hypotheses guident l'etude. La premiere suppose que des donnees structurees et un modele adapte ameliorent le rapprochement des textes. La deuxieme affirme que les relations entre candidats, competences et offres ameliorent le classement obtenu par la seule recherche semantique. La troisieme suppose que la combinaison du retrieval, du graphe et du controle de generation produit des recommandations plus comprehensibles et capables de signaler les competences manquantes.

## Diapositive 8 - Transition : etat de l'art (5 s)

Ces hypotheses s'appuient sur les concepts et les travaux presentes dans l'etat de l'art.

## Diapositive 9 - Concepts et evolution de l'IA (25 s)

Cette frise montre le passage des regles explicites vers l'apprentissage automatique, le deep learning, les Transformers, puis l'IA generative et agentique. Mon travail se situe a la derniere etape : le modele ne se contente pas de generer du texte. Il utilise des outils, consulte deux memoires de donnees, construit un contexte et controle sa reponse. L'agent reste toutefois encadre par un workflow defini.

## Diapositive 10 - Revue de la litterature (40 s)

La recommandation par contenu compare les caracteristiques du profil et de l'offre, mais reste sensible a la representation choisie. Le filtrage collaboratif exige des historiques d'interactions, qui sont insuffisants ici. Les graphes rendent explicites les relations entre metiers, competences et formations. Le RAG ajoute des preuves a la generation, et le GraphRAG ajoute une navigation relationnelle. Sur le plan economique, Becker souligne le role du capital humain, Autor celui des taches, tandis qu'Akerlof, Spence, Mortensen et Pissarides montrent les effets de l'information imparfaite et des frictions d'appariement. Mon positionnement consiste donc a hybrider recherche semantique, graphe et orchestration agentique.

## Diapositive 11 - Transition : methodologie (5 s)

Je presente maintenant les donnees mobilisees et la chaine methodologique retenue.

## Diapositive 12 - Donnees mobilisees (35 s)

Le systeme exploite 55 255 observations du marche de l'emploi : 13 957 offres et 41 298 profils candidats. Ces donnees sont completees par 17 178 entites de referentiels, dont 13 939 competences ESCO et 3 039 professions, ainsi que les nomenclatures camerounaises des metiers, des formations et des diplomes. L'objectif n'est pas d'empiler les sources, mais de les harmoniser afin de produire des variables comparables et des relations exploitables par la recherche vectorielle et le graphe.

## Diapositive 13 - Base ESCO (30 s)

ESCO est la classification europeenne multilingue des professions et des competences. Dans ce projet, elle fournit un vocabulaire commun et des identifiants stables. Les fichiers des professions, des competences, des hierarchies et des relations metier-competence permettent de reconnaitre qu'un meme besoin peut etre formule differemment dans un profil et dans une offre. ESCO sert donc a normaliser les concepts et a construire les relations utilisees pour calculer la couverture des competences et le skill gap.

## Diapositive 14 - Chaine de construction (30 s)

La demarche comprend quatre briques. L'ETL nettoie, normalise et aligne les donnees. Le modele d'embeddings transforme ensuite les textes en vecteurs indexes dans pgvector. Neo4j represente les relations entre candidats, competences, offres, metiers et formations. Enfin, LangGraph orchestre ces ressources. Les deux memoires sont complementaires : pgvector retrouve rapidement ce qui est proche par le sens, alors que Neo4j verifie si cette proximite repose sur des relations professionnelles observables.

## Diapositive 15 - Proxy de pertinence (40 s)

L'absence d'annotations de recruteurs impose une precaution methodologique. J'ai construit une verite terrain faible, appelee proxy de pertinence, qui combine recouvrement lexical, compatibilite sectorielle et niveau NCF. Une offre est consideree pertinente si son score depasse 0,35 et le soixante-dixieme percentile du vivier du candidat. Ce proxy ne fait pas partie du score hybride et ne prouve pas la pertinence metier. Il fournit seulement une reference commune pour comparer pgvector seul et pgvector enrichi par Neo4j. Cette distinction evite d'evaluer le systeme avec son propre score.

## Diapositive 16 - Transition : ETL (5 s)

La premiere brique operationnelle est l'ETL et la normalisation.

## Diapositive 17 - NCF et chunking (25 s)

La nomenclature camerounaise des formations organise les parcours du grand domaine jusqu'au domaine detaille, ainsi que par niveau de qualification. Cette hierarchie est conservee. Les documents sont decoupes par sections coherentes plutot qu'en fragments arbitraires. Chaque chunk conserve sa source, son titre et sa position. Le systeme peut ainsi retrouver une information de formation interpretable et relier les diplomes des candidats aux niveaux demandes dans les offres.

## Diapositive 18 - Pipeline ETL (25 s)

L'ETL suit trois etapes. L'extraction rassemble les offres provenant de Louma Jobs, FNE Cameroun, ACPE, Jobartis Cameroun, LinkedIn et Emploi.cm, ainsi que les profils et referentiels. La transformation nettoie les textes, harmonise metiers, secteurs et localisations, puis convertit les diplomes en niveaux NCF. Le chargement produit des fichiers Parquet et alimente pgvector et Neo4j. Le processus peut etre relance par lots pour integrer de nouvelles donnees.

## Diapositive 19 - Structure des offres (30 s)

Cette diapositive montre ce que produit concretement l'ETL pour une offre. Les champs bruts, comme le niveau d'etudes ou la description originale, sont conserves pour la tracabilite. En parallele, le systeme cree des variables comparables : titre, secteur, ville, contrat, experience minimale, niveau NCF et liste de competences. La description passe de `details_raw` a `details_clean`, puis `text_to_embed` rassemble les informations utiles a la recherche semantique. Enfin, `pair_query_text` contient la fiche structuree de l'offre et `pair_details_text` sa description ; leur association constitue une paire positive pour le fine-tuning.

## Diapositive 20 - Structure des candidats (20 s)

Pour les candidats, la structure regroupe l'identification, la formation, la qualification, le projet professionnel et la mobilite. Le niveau final est harmonise avec la NCF. Le champ `text_to_embed` synthetise le metier vise, le secteur, le niveau, les etudes, la filiere, la qualification et l'objectif. Il produit ainsi une representation textuelle homogene que le modele peut comparer aux offres.

## Diapositive 21 - Transition : embeddings (5 s)

La deuxieme brique concerne l'adaptation du modele d'embeddings et l'indexation vectorielle.

## Diapositive 22 - Fine-tuning (55 s)

J'ai choisi `all-MiniLM-L6-v2` pour son compromis entre qualite et ressources : il est deja entraine pour produire des embeddings de phrases et ses vecteurs de 384 dimensions sont moins couteux a encoder, stocker et rechercher que des vecteurs de 768 dimensions. Ce n'est pas un Mixture-of-Experts. C'est un encodeur Transformer dense de 6 couches, 12 tetes d'attention, dimension cachee 384, couche intermediaire 1 536, environ 22,7 millions de parametres entrainables et une limite operationnelle de 256 tokens. Le mean pooling produit un vecteur unique par texte. J'ai construit 6 476 paires offre-description, reparties en 4 542 pour l'entrainement, 981 pour la validation et 953 pour le test. Le fine-tuning est complet, sans LoRA ni QLoRA : cinq epoques, batch 32, taux maximal de deux fois dix puissance moins cinq, warmup puis decroissance cosinus, et `MultipleNegativesRankingLoss`, ou les 31 autres descriptions du batch jouent le role de negatifs implicites. L'entrainement a dure environ quatre heures douze sur CPU ; le meilleur comportement apparait autour de l'epoque 4.

## Diapositive 23 - Benchmark semantique (35 s)

Tous les modeles sont compares sur la meme tache : retrouver dans le top 10 la description associee aux caracteristiques d'une offre de test. Le modele adapte obtient un NDCG@10 de 0,6232, un MRR@10 de 0,5874 et un Recall@10 de 0,7398. Concretement, il ordonne mieux les resultats et retrouve environ 74 % des elements attendus dans les dix premieres positions. Il depasse les approches lexicales et les modeles generiques, au prix d'une latence plus elevee.

## Diapositive 24 - Transition : graphe (5 s)

La troisieme brique est le graphe de connaissances et le score hybride.

## Diapositive 25 - Structure du graphe (40 s)

Le chemin central relie un candidat a une competence par `POSSEDE`, puis une offre a cette competence par `REQUIERT`. Il permet de distinguer les competences couvertes des competences manquantes. Les relations vers le metier, le secteur, la localisation et le niveau NCF enrichissent l'explication. Les relations `POSSEDE`, `NECESSITE` et `REQUIERT` dominent logiquement le graphe. Neo4j ne remplace donc pas la similarite semantique : il transforme une proximite textuelle en chemins explicables, sous reserve que les relations extraites soient fiables.

## Diapositive 26 - Score hybride, Cypher et reranking (50 s)

Le matching fonctionne en deux temps. pgvector constitue d'abord un vivier d'offres proches. Pour chaque couple, une requete Cypher parametree suit le chemin `Candidat-POSSEDE-Competence-REQUIERT-Offre` et calcule la couverture : nombre de competences requises possedees divise par le nombre total de competences requises. Le score hybride combine ce taux et la similarite pgvector avec des poids dont la somme vaut un. Text-to-Cypher remplit une autre fonction : pour une question relationnelle libre, il traduit la demande en requete Cypher, qui doit ensuite respecter le schema et les controles du systeme. Il ne calcule pas automatiquement le score de chaque offre. La distribution montre un signal pgvector continu autour de 0,53 et un signal Neo4j plus discret, avec des zeros et de fortes couvertures. Neo4j ne relance donc pas la recherche : il explique et reclasse le vivier. Les 690 couples affiches correspondent aux dix offres enregistrees pour chacun des 69 candidats evalues.

## Diapositive 27 - Transition : Agentic GraphRAG (5 s)

La derniere brique assemble ces composants dans un agent GraphRAG orchestre.

## Diapositive 28 - Architecture de l'agent (35 s)

L'agent commence par analyser l'intention : recommandation, skill gap, recherche documentaire, question relationnelle ou diagnostic. Il planifie ensuite les outils, execute les recherches, construit un contexte et genere une reponse. Par exemple, une recommandation mobilise pgvector et Neo4j, alors qu'une question sur le niveau Master interroge les chunks documentaires. L'agent n'est donc pas un chatbot qui improvise : il suit un graphe d'etats, conserve les resultats et trace les outils appeles.

## Diapositive 29 - Selection et mobilisation des outils (35 s)

Cette diapositive decompose la decision de l'agent. `analyse_request` detecte l'intention, l'identifiant candidat et les entites citees. `plan_tools` associe ensuite l'intention aux outils : une recommandation appelle le moteur hybride ; un skill gap appelle Neo4j ; une question sur un referentiel appelle la recherche documentaire pgvector ; une question relationnelle libre peut mobiliser Text-to-Cypher. `execute_tools` lance les appels et conserve aussi les erreurs. `build_context` normalise ensuite des sorties heterogenes en preuves communes : offres, scores, competences couvertes ou manquantes, chemins du graphe et chunks documentaires. Le LLM ne recoit donc pas directement les bases ; il recoit un contexte selectionne et tracable.

## Diapositive 30 - Generation controlee (30 s)

Les sorties des outils sont rassemblees dans un contexte structure, puis transmises au modele GPT-OSS-20B via OpenRouter. Le critic mesure ensuite si la reponse est suffisamment ancree dans les preuves. Le seuil operationnel est 0,46 : au-dessus, la reponse peut etre acceptee ; en dessous, le workflow demande une revision ou un elargissement du contexte. Ce controle mesure surtout la fidelite lexicale au contexte. Il ne remplace pas une validation humaine de la verite metier.

## Diapositive 31 - Demonstration API et chatbot (4 min)

[Ne pas faire un discours theorique. Annoncer :]

Je vais maintenant montrer comment les briques precedentes fonctionnent ensemble dans l'application. La demonstration suivra quatre etapes : lecture d'un profil, recommandation d'offres, explication du classement et analyse du skill gap.

1. **30 s - Etat du systeme.** Montrer rapidement que l'API, pgvector et Neo4j repondent.
2. **60 s - Recommandation.** Saisir : « Quelles offres correspondent au profil du candidat PPKOU2501080016340 ? Presente les cinq meilleures et explique la premiere. »
3. **60 s - Explication.** Montrer le score semantique, la couverture des competences et les traces des outils appeles.
4. **60 s - Skill gap.** Demander : « Pour la meilleure offre, distingue les competences acquises, manquantes et essentielles manquantes. »
5. **30 s - Conclusion de la demo.** Dire : « Cette sequence montre la complementarite des deux memoires : pgvector retrouve, Neo4j explique et l'agent organise la reponse. »

En cas de panne reseau ou de service, utiliser une capture ou une video locale et commenter exactement les memes etapes.

## Diapositive 32 - Transition : evaluation (5 s)

Apres cette demonstration fonctionnelle, je presente l'evaluation interne du systeme.

## Diapositive 33 - Performance des briques (55 s)

Deux evaluations sont distinguees. D'abord, le modele d'embeddings est teste sur 953 paires non vues : il atteint 0,6232 en NDCG@10, 0,5874 en MRR@10 et 0,7398 en Recall@10. Ensuite, l'ablation demande 80 candidats tires aleatoirement avec la graine 42 ; 69 sont effectivement evalues et 11 echouent pour des erreurs techniques. Pour chaque candidat, pgvector fournit exactement le meme pool initial de 30 offres aux deux variantes. La premiere conserve l'ordre vectoriel ; la seconde enrichit ces offres avec Neo4j puis les reclasse. Au top 10, le NDCG passe de 0,5575 a 0,7392, le MRR de 0,6908 a 0,8093 et le Recall de 0,4491 a 0,6147. La comparaison isole donc l'effet du reranking sur le meme vivier, selon le proxy. HNSW accelere la recherche approximative en evitant une comparaison exhaustive, mais il ne garantit pas une complexite universelle en logarithme.

## Diapositive 34 - Optimisation bayesienne et critic (60 s)

La calibration porte sur 690 couples, soit dix offres pour chacun des 69 candidats. A chaque essai, Optuna propose deux poids positifs, les normalise pour que leur somme vaille un, recalcule le score hybride, reclasse les dix offres de chaque candidat et moyenne la metrique cible. Le sampler TPE separe progressivement les essais performants des essais moins performants, modelise leurs distributions de poids, puis propose davantage de combinaisons dans les zones susceptibles d'ameliorer l'objectif. Apres 500 essais, l'objectif NDCG@10 est maximal sur la frontiere Neo4j seul, avec un NDCG de 0,9697 selon le proxy. Pour l'objectif combinant MRR et precision, les poids retenus sont environ 0,6615 pour pgvector et 0,3385 pour Neo4j. Ces poids ne sont pas universels : ils dependent du proxy et d'un top 10 deja selectionne. Enfin, le critic est calibre separement sur 102 comparaisons, 51 contextes reels et 51 contextes melanges. Le seuil 0,46 rejette les 51 contextes melanges et accepte 50 contextes reels sur 51.

## Diapositive 35 - Bilan, limites et perspectives (45 s)

Le bilan doit rester factuel. Premier resultat : sur le benchmark semantique, le modele adapte atteint 0,6232 en NDCG@10 et 0,7398 en Recall@10. Deuxieme resultat : sur le meme pool de 30 offres, l'ajout du graphe fait progresser le NDCG@10 de 0,5575 a 0,7392 et fournit les competences couvertes et manquantes. Troisieme resultat : le critic calibre a 0,46 distingue presque parfaitement les contextes reels des contextes melanges dans ce protocole lexical. Ces resultats soutiennent surtout l'hypothese de l'apport relationnel, mais ne prouvent pas encore la satisfaction d'un recruteur. Les limites sont l'absence d'annotations humaines, le biais potentiel du proxy, les 11 echecs d'evaluation et la qualite variable des donnees scrapees. Les priorites sont donc un jeu annote par des experts, des tests utilisateurs, l'integration au site KmerAI, puis le deploiement du graphe sur Neo4j Aura et la documentation des artefacts sur Hugging Face.

## Diapositive 36 - Remerciement (10 s)

En conclusion, ce travail valide une architecture operationnelle et l'apport interne du graphe, mais il reste un outil d'aide a la decision qui doit encore etre valide par des experts metier. Je vous remercie pour votre attention et je suis disponible pour vos questions.

## Diapositives 37-38 - Fin ou secours

Ne pas les commenter. Elles peuvent servir de page d'attente pendant les questions ou etre supprimees de la version projetee.

---

## Corrections indispensables avant projection

1. **Titre general :** remplacer « aux Cameroun » par « au Cameroun » et « de graphe de connaissance » par « des graphes de connaissances ».
2. **Diapositive 4 :** remplacer « Source : INS, 2025 » par les sources utilisees dans le memoire : estimations OIT diffusees par la Banque mondiale. Preciser les annees : chomage 2025, NEET 2021, emploi vulnerable 2024.
3. **Diapositive 10 :** remplacer « Burke 2022 » par « Burke, 2002 » et « et all » par « et al. ».
4. **Diapositive 14 :** corriger « 0404 Agentic GraphRAG » en « 04 Agentic GraphRAG ».
5. **Diapositive 18 :** ecrire « Louma Jobs » et non « Lumajobs.com » ; les six sources observees sont Louma Jobs, FNE Cameroun, ACPE, Jobartis Cameroun, LinkedIn et Emploi.cm.
6. **Diapositive 22 :** remplacer « GPU » par « CPU ». L'entrainement a dure environ 4 h 12 min sur CPU.
7. **Diapositive 26 :** retirer « Modele Text-to-Cypher » du calcul systematique du score. Le reranking utilise des requetes Cypher parametrees ; Text-to-Cypher sert aux questions relationnelles libres.
8. **Diapositive 26 :** remplacer « Top 10 offres semantiquement similaires » par « pool initial pgvector, puis top 10 enregistre apres reranking » si le bloc de 690 couples est conserve.
9. **Diapositive 30 :** remplacer la phrase « si le LLM LangGraph juge accepte egalement ». LangGraph n'est pas un LLM ; c'est l'orchestrateur. Dire : « le critic produit accept ou revise selon le score et les regles du workflow ».
10. **Diapositive 33 :** supprimer `O(n^381)` et eviter de presenter `O(log n)` comme une garantie theorique de HNSW. Dire simplement : « HNSW reduit le nombre de comparaisons par rapport a une recherche exhaustive, au prix d'une approximation controlee ».
11. **Diapositive 35 :** remplacer les lettres `F W R P` par `F L R P` ou conserver les mots complets Forces, Limites, Risques, Perspectives.
