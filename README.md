# Projet de TLN français de l'Ecolab : sommaires détaillés, résumé automatique, similarité de documents (19/05/2021)

Bienvenue dans le référentiel consacré au projet de l'Ecolab sur le Traitement du Langage Naturel (français).

Vous pourrez trouver ici les codes finis et les codes exploratoires produits par l'équipe en charge du projet. Tout le projet est écrit en Python (ou Jupyter).

## Objectif : produire des outils, via du Machine Learning et du Natural Language Processing, pour aider les auditeurs qui élaborent les avis sur les études d'impact environnemental.

En juin, le projet est divisé en plusieurs parties :
1. une première partie sur la détection et l'extraction des sommaire de documents PDF, puis leur découpage;
2. une partie sur le traitement et l'analyse des enjeux (via du **topic modelling**);
3. une partie sur le résumé automatique de sections (via du **Deep Learning**, plus moins *supervisé*).
4. une partie sur l'extraction de mots-clés (via du **Deep Learning**,**Graphs** et approches **Statistiques** *non supervisées*)
4. une dernière partie sur un système de recommandation d'avis (fondé sur du **Deep Learning** et du **Collaborative Filtering** notamment )

Pour le moment, il existe quatres dossiers :
* **Pipeline** : qui correspond aux chemins d'exécutions des travaux ayant été exécutés sur Dataiku. Notamment, l'analyse des enjeux, la détection et l'extraction des sommaires, le découpage des documents; la détection des avis vides.
* **Ruben** : tous les codes _exploratoires_ de Ruben Partouche (@rbpart).
* **Theo** : tous les codes _exploratoires_ de Théo Roudil-Valentin (@TheoRoudilValentin).
* **Zakaria** : tous les codes ainsi que la documentation liés à l'extraction de mots-clés et à la recommandation d'avis (@IIZCODEII).

Ces codes exploratoires sont parfois flous pour des personnes extérieures au projet, mais cela est normal. Les codes finaux seront mis dans **Pipeline** et expliqués par des documents et par des précisions dans le code lui-même.

Pour setup le projet, il faut exécuter le script setup.py depuis la racine du projet (2021-NLP_AE) qui va aussi chercher les dernières données (environ 10Go) sur le disque partagé du SRI (le renommer en K si ce n'est pas le cas pour votre ordinateur). Attention : pour le moment le setup va juste identifier les noms de fichier sans comparer le contenu : il faut supprimer les fichier puis relancer setup si on veux actualiser un fichier (amélioration a faire).

## 1 - Détection et extraction du sommaire
Avant toute chose, nous avons transformés en HTML les études d'impact dont nous dispositions, à savoir les dossiers clos disponibles sur https://www.projets-environnement.gouv.fr/pages/home/ (soit environ 650 études). Pour cela, nous avons utilisé le logiciel propriétaire ABBYY Fine Reader. Sur ces 650 études, seules 150 ont été utilisables après le traitement ABBYY, cette réduction est dûe à plusieurs erreurs : mot de passe, PDF trop lourd etc...
### 1.1 - Construction de la base HTML
Une fois cette OCRisation (ROC, Reconnaissance Optique de Caractères, ici transformation de PDF en texte) faîte, nous avons utilisé le premier code disponible dans **Pipeline** pour découper chaque fichier disponible, ligne par ligne. Ainsi, nous avons une base de données où pour chaque numéro d'études, nous avions un ensemble de lignes correspondants à celles disponibles dans le fichier HTML : numéro x lignes. Ces lignes comportent donc toute l'information du fichier HTML. Au final, la dimension de la base était d'environ 2M de lignes x une quarantaine de colonnes.

Notre hypothèse était que, si l'OCR était correcte, les titres devraient avoir une distribution de balises HTML différente des lignes normales. Ainsi, il devrait être possible de les détecter.
### 1.2 - Création des variables (feature engineering)
Dans un second temps, nous avons donc dû créer des variables permettant d'avoir une représentation de ces distributions. Pour cela, nous avons créé un ensemble de variables dont beaucoup de binaires pour capter l'information disponible dans les lignes.
Pour chaque balise présente dans l'ensemble des fichiers HTML fut donc codé une variable binaire pour indiquer la présence de la balise dans la ligne considérée. D'autres variables renseignent sur la longueur de la ligne, le nombre de mots, le nombre de caractères, la présence de caractères spéciaux ou encore la taille de la police d'écriture. De même, une variable importante a été le repérage des lignes proches du mot **sommaire** et avant la première répétition de la première ligne correspondante. Concrètement, lorsqu'apparait le mot "sommaire" (ou "Table des matières" etc...) il est très probable que les lignes suivantes soient des titres, et que ce sommaire s'arrête à la première apparition de la première ligne après le mot "sommaire" (autrement dit, l'apparition du premier titre). Cependant cette variable a posé beaucoup de problèmes car nous avons remarqué que notre processus n'arrivait pas à coder correctement toutes les études. En effet, ces études sont très différentes et font preuve d'un spectre de structure très large, rendant donc la création de variables particulièrement difficile.
### 1.3 - Les étapes de la détection
Avant de passer directement à la détection, nous avons tenté de faciliter l'apprentissage des futurs modèles. Notamment, nous avons remarqué que beaucoup de variables binaires semblaient apporter la même information. Nous avons donc mis en place une ACP pour réduire la dimension de notre base. Pour les 33 variables binaires dont nous suspections une rendondance d'information, nous avons encodé 12 variables via l'ACP avec un score d'explication de variance très élevé (proche de 98%).

Notre but final étant d'avoir un classifieur permettant de détecter parfaitement les titres, nous avons procédé par étapes.
D'abord, comme nous travaillions en aveugle, nous n'avions pas d'informations sur les lignes et donc ne savions pas lesquelles étaient des titres ou non, nous avons appliqué un algorithme de K-means (K=2) pour séparer une première fois les lignes, toujours en se basant sur l'hypothèse initiale. Ainsi nous obtenions deux groupes, dont un comportant une grande majorité de ce que nous considérions, à l'oeil humain, comme des titres. L'autre étant constitué de beaucoup de lignes de textes. Cette première classification nous a donc permis d'obtenir une premire __idée__ de la __vraie valeur__ des lignes.
Puis, nous avons sélectionné un sous-ensemble de 300000 lignes dont nous avons relabellisé les déchets du __groupe des titres__. Ce choix de ne considérer qu'un groupe repose sur le fait que le groupe des titres du K-means était beaucoup plus réduit et semblait contenir beaucoup plus d'erreurs relativement à sa taille. Une piste d'amélioration serait de relabelliser l'ensemble, mais cela demande beaucoup de temps et constitue un travail peu gratifiant. Une fois ce sous-ensemble relabellisé, nous étions donc en possession d'un nouveau label, un raffinement de la prédiction du K-means. C'est ce label qui nous a servi pour entraîner un modèle de __machine learning__ supervisé.
A partir de là, nous avons entraîné un modèle de Forêt Aléatoire pour retrouver le label raffiné, en prenant l'ensemble des variables créées via le HTML ainsi que le label fourni par le K-means. Ce modèle réussissait quasiment parfaitement à retrouver le label que nous lui fournissions (score proche voir égal à 1). Cependant, après vérification, bien que le score soit de 1, les titres fournis furent décevants : sommaires incomplets, déchets dans le sommaire voir absence de sommaire.
Nous avons donc rajouté une étape de relabellisation de ce dernier label fourni par le modèle de Forêt Aléatoire et de même retravaillé la partie de création de variables.
Au final, après plusieurs itérations, les résultats furent nettement améliorés bien qu'imparfaits. Pour une majorité d'études, nous étions capables de sortir le sommaire complet. Cependant, nous ne pouvons fournir de score rigoureux car cela nécessiterait une labellisation humaine.
### 1.4 - Extraction et découpage
Une fois les titres labellisés, il a suffi de les sortir et d'essayer de les retrouver dans le texte HTML que nous avions à l'origine. Cette étape s'est révélé anormalement complexe et peu concluante. Un processus standard de découpage du texte entre deux titres pour chercher les paragraphes correspondant a très peu fonctionné et a semblé incapable de nous fournir, pour chaque titre, les paragraphes associés.

Face à cette difficulté,

## 2 - Traitement et analyse des enjeux
**Objectif** : identifier les enjeux présent dans un texte de longueur variable (idéalement le plus court possible)

En non supervisé, on a testé plusieurs approches de topic modeling (LDA, LSA, word2vec + Kmeans, etc...) qui n'ont pas été concluantes car les résultats étaient trop peu utiles et pertinents du point de vue métier.
On a donc ramené le problème a celui d'une classification multiclasse et multilabel, en utilisant un algorithme semi-supervisé, dans le sens ou il ne prend pas en entrée la "cible" pour s'orienter, mais un thésaurus (dictionnaire de mots associés aux enjeux).

Pour une prise en main et utilisation rapide, voir les codes "Avis_semisupervised.py" et "Section_semisupervised.py" qui servent d'exemple d'utilisation du pipe.

### 2.1 - Algorithme, paramètres, métriques
L'algorithme utilisé est CorEx (version adaptée pour le topic modeling, voir https://github.com/gregversteeg/corex_topic/tree/master/corextopic).
C'est un algorithme dont le principal défaut est d'avoir une grande variance : l'initialisation est semi-aléatoire et le résultat final change grandement entre deux tests.

Les scripts appliquant les développements décrits plus bas sur les sections et les avis sont section_semisupervised.py et avis_semisupervised.py.
La classe principale regroupant les techniques et développements appliqués pour améliorer le score est contenue dans topicmodeling_pipe.py.

Chaque amélioration a ses propres paramètres qui sont décrits dans chaque section. CorEx n'a, initialement, que 3 paramètres auxquels on s'est intéressés :
Lors de l'initialisation de la classe CorEx :
n_hidden : nombre de topics qui vont être recherchés
Lors du fit de l'algo :
anchor_strength : coefficient appliqué pour augmenter l'importance des ancres (dictionnaire du thésaurus) dans le modèle
anchors : liste de listes d'ancres. Chaque liste d'ancres correspond a un dictionnaire pour un enjeu.

En terme de métriques :
Puisqu'on ne voulait pas se contenter de mesures "non-supervisées" pour obtenir un résultat vraiment intéressant du point de vue métier (auditeurs de la DREAL),
nous avons labellisé avec une auditrice un petit nombre d'exemple (un peu moins d'une centaine) d'avis (pas de sections pour le moment !). On a donc les scores
classiques : precision, accuracy, recall, F1.

Certaines fonctions utiles ont été créées pour mieux visualiser ces scores et les effets d'altérations sur ces scores dans Pipeline.Enjeux.utils (voir notamment evaluate)

Sur certaines approches (bagging et boosting notamment), ont été expérimentés d'autres métriques intéressantes pour notre problème multiclass/multilabel :
la Hamming Loss (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html) et la Label Ranking Loss (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.label_ranking_loss.html). Plus de détails dans la section concernée.

### 2.2 - Enrichissement
**Objectif** : améliorer les performances par l'enrichissement du thésaurus.
**Approches** : entrainement d'un word2vec puis calcul de similarité
**Pistes d'améliorations** : tester d'autres métriques de similarités, trouver de nouvelles approches donnant des recommandations plus pertinentes ?

#### 2.2.1 - Calcul de similarité
**Attention** : utilisation d'un GPU très hautement recommandée pour le calcul de similarité.
**Utilisation** :
On appelle la classe maksimilarity de Pipeline.Enjeux.enrichissement sur le thésaurus.
Ensuite, le .fit permet d'entrainer le modèle W2V gensim, le .transform crée des listes de vecteurs correspondant aux mots du thésaurus retrouvés dans le vocabulaire du W2V (attention aux problèmes de compatibilité selon les versions de gensim, ici tourne sur la version XX), .cos_moyen_all fait le calcul de cosimilarité (étape nécessitant du GPU), .cos_moyen_batch exécute la même tache mais en batch pour réaliser des tests.

A chaque étape, le code sauvegarde automatiquement un fichier correspondant a la sortie générée (sauf le transform qui est très rapide), qu'il est possible d'utiliser lorsqu'on initialise la classe (cf aide de la classe) pour ne pas refaire certaines étapes longues (entrainement du W2V et calcul de cosimilarité).

Le code est optimisé pour tourner sur gpu (via torch.cuda) et cpu mais il reste malgré tout extrêmement lent sur cpu (25 jours de traitement sur un serveur avec 64 coeurs contre 10 min sur gpu pour traiter toutes les données des avis, soit 10Mo de données).

La sortie finale est une matrice (n_words,n_topic) contenant, pour chaque mot, la cosimilarité (CosineSimilarity) moyenne avec chaque vecteur de chaque topic.
Pour avoir les mots les plus intéressants d'un topic, on regarde donc les mots avec la cosimilarité moyenne la plus forte pour ce topic.

Pour le moment, les tests d'enrichissement automatisés sur les enjeux mal identifiés ne sont pas très concluants (pas d'amélioration en moyenne du score sur un modèle CorEx simple pour la Gestion des déchets sur les Avis).
Il pourrait être intéressant de tester l'enrichissement sur des modèles en bagging ou boosting, pour éliminer la variance et voir si on constate des améliorations plus significatives en moyenne.
Le pipeline d'enrichissement n'est pas implémenté car les tests ne sont pas concluants. Il est toutefois possible de retrouver les test dans "semisupervised_avis.py".

### 2.3 - Stratification et augmentation des données
**Objectif** : améliorer les performances sur certains labels spécifiquement.
**Approches** : Stratification des données, augmentation des données
Un des problèmes constatés est que les enjeux les moins représentés sont moins bien détectés par l'algorithme (performances a 0,3 en F1 contre 0,7 en moyenne).
Pour résoudre ce souci, deux approches complémentaires ont été testées : la stratification puis l'augmentation des données.

#### 2.3.1 - Stratification des données
**Utilisation** :
En utilisation directe on appelle la fonction get_minority_instance(X,y) avec X dataframe des documents labellisés avec leurs features, y les labels correspondants (même index pour X et y)
On récupère X_sub et y_sub, sous ensemble des documents aux labels minoritaires.
En utilisation indirecte, dans la classe principale CorExBoosted, on appelle la méthode stratify() pour générer les attributs .X_sub et .y_sub, sinon lors du .fit l'argument optionnel stratify lance ou non la stratification (elle est systématiquement faite si on ne précise pas l'inverse)

La stratification consiste ici a sélectionner les exemples (avis, paragraphe, autre...) avec les les labels les moins représentés pour augmenter leur proportion par rapport a la distribution initiale, et avoir des proportions similaires pour tous les enjeux.

L'implémentation est assez simple : on calcule pour chaque label sa fréquence d'apparition et on fait la moyenne des fréquences : tous les labels sous représentés par rapport a la moyenne sont considérés comme minoritaires. On récupère ensuite un sous ensemble des données correspondant aux documents contenant au moins un enjeu minoritaire.

En revanche si des enjeux cooccurent souvent, comme c'est le cas pour la gestion des déchets avec d'autres enjeux par exemple, il sera difficile d'augmenter la proportion des enjeux sous representés pour la faire arriver au même niveau que les autres : il s'agit en fait d'un problème de machine learning en soit.

**Pistes d'améliorations** : essayer une autre méthode consistant a prendre uniquement les documents qui n'ont pas un trop grand nombre de label (2 par exemple) puis équilibrer à partir de ce sous ensemble. Le vocabulaire sera potentiellement plus spécifique.

#### 2.3.2 - Augmentation des données
**Attention** : Il semble que malgré les améliorations, si on rajoute trop de samples(plus de 2 fois le nombre de samples d'origine environ), cela déséquilibre plus la distribution qu'autre chose ! Problème a confirmer/diagnostiquer mais voir explications plus bas sur son orginine supposée.
**Utilisation** :
En utilisation directe on appelle la fonction MLSMOTE2(X_sub,y_sub,n_samples), avec X_sub et y_sub dataframes sous ensemble des documents aux labels minoritaires, n_samples le nombre de samples a rajouter.
En utilisation indirecte, dans la classe principale CorExBoosted, on appelle la méthode augment(n_samples = 50), sinon lors du .fit l'argument optionnel augment lance ou non l'augmentation (elle est systématiquement faite si on ne précise pas l'inverse)

Pour pallier au problème précédent, on peux procéder à de l'augmentation de données qui consiste à créer artificiellement des exemples de documents avec MLSMOTE2 (MultiLabel SMOTE, voir : https://medium.com/thecyphy/handling-data-imbalance-in-multi-label-classification-mlsmote-531155416b87). L'algo sélectionne une référence qui est un élément au hasard parmis les données d'entrée, prend un de ses 5 plus proches voisins au hasard, et crée une nouvelle ligne en ajoutant une perturbation basée sur la différence entre la référence et son voisin. Si un enjeu est représenté dans au moins 3 des voisins, alors il est indiqué qu'il est présent dans l'entrée synthétique.

Initialement, l'algorithme n'enrichissait pas suffisament les labels sous représentés. Plusieurs raisons peuvent expliquer cela :
Les données sont si pauvres en label sous représentés que trop peu d'exemples ont au moins 3 voisins avec ces labels minoritaires, donc peu de nouveaux exemples synthétiques contenant ces enjeux sont créés.

**Pistes d'améliorations** :  on peut régler ce problème de deux manière :
  - Améliorer la stratification un peu plus pour continuer à uniformiser la distribution des enjeux, cela passe par de la labellisation.
  - Diminuer la sélectivité du processus d'augmentation, par exemple en autorisant, si un label est sous représenté, a considérer que la ligne synthétique contient l'enjeu si seulement 1 voisin contient le label sous représenté. Cela pourrait par contre fausser l'apprentissage. Cette solution est implémentée avec MLSMOTE2 (MLSMOTE est l'algorithme initial décrit dans l'article précédent).

### 2.4 - Bagging et boosting
**Objectif** : améliorer les performances globales
**Approches** : Bagging simple, boosting
Un des problèmes constatés est la grande variance de l'algo : pour deux tests a paramètres identiques (sans random seed) on a des résultats qui peuvent être radicalement différents en terme de performance.
Pour améliorer cela, on fait du bagging : on entraine un grand nombre de modèle pour arriver a la prédiction moyenne.


#### 2.4.1 - Bagging
**Utilisation** :
Via la classe principale CorExBoosted, utiliser l'argument n_classif du .fit() ! n_classif correspond au nombre d'instances de CorEx qui vont êtres entrainées.
Si on ne stratifie pas et qu'on augmente pas les données, tout est entrainé sur le corpus de base
Si on stratifie sans augmenter, tout est entrainé sur le sous ensemble des documents aux labels minoritaires
Si on stratifie et qu'on augmente, chaque classifieur est entrainé sur un dataset augmenté différent.

Les performances sont légèrement meilleures (on augmente le F1 score moyen de 0,05), et cela diminue grandement la variance.
Un paramètre qui est ajouté par cette approche est la sélectivité du modèle : normalement, CorEx applique un seuil, lorsqu'il calcule la probabilité d'apparition d'un topic dans un document, si celle ci est supérieure à 0.5, il considère que l'enjeu est présent. Il est possible avec la méthode .predict() de changer la sélectivité (arg selectivity) et même de l'optimiser avec .optimize_selectivity(). Les bornes théoriques sont bien 0 et 1, mais si on n'encadre pas un peu plus finement celles ci, on peux obtenir des résultats absurdes (sélectivité de 0.005 par exemple...).

**Pistes d'améliorations** : tester ce qui se passe si on stratifie&augmente et qu'on entraine tous les classifs sur le même dataset augmenté, sinon boosting

#### 2.4.2 - Boosting
Pour aller plus loin, des approches de boosting (bagging avec coefficients en fonction des performances de modèles) doivent être testées.
**Utilisation** : depuis la classe principale CorExBoosted, on appelle la méthode .optimize_weights(). On peux changer la méthode utilisée par scipy (voir la doc de minimize : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)

Déjà testé : scipy.optimize.minimize pour minimiser la Hamming Loss, la Ranking Label loss, une loss "artisanale" basée sur le score F1 moyen. Pas concluant du tout, les poids ne bougent pas du tout !

**Pistes d'améliorations** : Nouvelles approches ou algos a réimplémenter de zéro (AdaBoost, GradientBoost, Deep learning ?)

## 3 - Résumé automatique de sections
**Objectif** : effectuer du résumé extractif (sélection des phrases pertinentes) sur les sections des études d'impacts.

Vous trouverez l'ensemble des travaux opérationnels et exploratoires dans le dossier Théo.
 
Nous développons quatre approches différentes qui sont :
* Une famille de modèles basées sur du __Deep Learning__
* Un modèle utilisant l'algorithme TextRank
* Un modèle basé sur la similarité de l'embedding des phrases
* Enfin une famille de modèle __benchmark__ pour la comparaison

Veuillez vous référer au dossier Théo pour y lire la note technique, ainsi qu'une explication plus précise et formelle dans le README disponible.

## 4 - Extraction de mots-clés

**Objectifs** :
* Extraire des mots-clés pertinents sur les sections des études d'impacts (et tout autres documents) en l'absence de données labellisées et de recul sur le meilleur modèle pour un cas d'usage donné.
* Proposer un package directement actionnable pour toutes personnes/organisations voulant extraire des mots-clés sur un corpus de texte sans expertise prélable ni a priori sur les performances des nombreux modèles disponibles.

Pour une présentation détaillée du modèle *keyBoost* qui en découle , se référer au dossier *Zakaria*. Une documentation complète est disponible dans le sous-répertoire *Docs* de *Zakaria*. Cette document retrace le contexte, l'architecture technique, la validation scientifique de la pertience de *keyBoost*, un demonstrateur web interactif ainsi qu'un tutoriel/documentation sur le package python *keyBoost*.

## 4 - Système de recommandation d'avis
**Objectif** : formuler des recommandations d'avis correspondant à des études d'impacts similaires.




# **Contacts et citations :**
Ruben Partouche : ruben.partouche@student-cs.fr  
Théo Roudil-Valentin : theo.roudil-valentin@ensae.fr  
Zakaria Bekkar : zakaria.bekkar@ens-paris-saclay.fr  
