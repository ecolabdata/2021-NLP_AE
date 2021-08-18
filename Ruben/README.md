
# 2 Dossier Ruben - Analyse des enjeux

Date de la dernière modification : **05/08/2021**

Bienvenue dans le dossier de Ruben Partouche contenant les exemples d'utilisation des codes présents dans Pipeline/Enjeux

[Pipelin\\Enjeux\\topicmodeling_pipe.py](../2021-NLP_AE/Pipeline/Enjeux/topicmodeling_pipe.py)
Code principal contenant la classe CorExBoosted qui encapsule un certain nombre d'améliorations utiles sur notre problème.

# Table des Matières
1. [Contenu du dossier](#contenu)
2. [L'approche de l'analyse des enjeux](#approches)  
    1. [Algorithme, paramètres, métriques](#a1)  
    2. [Enrichissement](#a2)  
        a.[Calcul de similarité](#a21)  
    3. [Stratification et augmentation des données](#a3)    
        a.[Stratification des données](#a31)   
        b.[Augmentation des données](#a32)  
    4. [Bagging et boosting](#a4)  
        a.[Bagging](#a41)  
        b.[Boosting](#a42)  
    5. [Suite du projet](#a5) 


## 1. Listes des codes et applications :
<a name="contenu"></a>

* __Avis_Semisupervised.py__ : code d'exécution des pipelines sur les Avis, avec notamment l'utilisation de données corrigées pour stratifier et augmenter les données artificiellement.
* __Section_Semisupervised.py__ : code d'exécution des pipelines sur les Sections découpées (impropres)
* __Enrichissement_explo.py__ : code exploratoire définissant des métriques pour enrichir le thésaurus
* __compute_mots_finale_Theo_Ruben.py__ : obsolète ! ancien script Dataiku pour réunir des résultats non supervisés de topic modeling

## 2. L'approche de l'analyse des enjeux
<a name="approches"></a>

En non supervisé, nous avons testé plusieurs approches de topic modeling (LDA, LSA, word2vec + Kmeans, etc...) qui n'ont pas été concluantes car les résultats étaient trop peu utiles et pertinents du point de vue métier.

On a donc ramené le problème a celui d'une classification multiclasse et multilabel, en utilisant un algorithme semi-supervisé, dans le sens ou il ne prend pas en entrée la "cible" pour s'orienter, mais un thésaurus (dictionnaire de mots associés aux enjeux).

### 2.1 - Algorithme, paramètres, métriques
<a name="a1"></a>

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
<a name="a2"></a>

**Objectif** : améliorer les performances par l'enrichissement du thésaurus.
**Approches** : entrainement d'un word2vec puis calcul de similarité
**Pistes d'améliorations** : tester d'autres métriques de similarités, trouver de nouvelles approches donnant des recommandations plus pertinentes ?

#### 2.2.1 - Calcul de similarité
<a name="a21"></a>

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
<a name="a3"></a>

**Objectif** : améliorer les performances sur certains labels spécifiquement.
**Approches** : Stratification des données, augmentation des données
Un des problèmes constatés est que les enjeux les moins représentés sont moins bien détectés par l'algorithme (performances a 0,3 en F1 contre 0,7 en moyenne).
Pour résoudre ce souci, deux approches complémentaires ont été testées : la stratification puis l'augmentation des données.

#### 2.3.1 - Stratification des données
<a name="a31"></a>

**Utilisation** :
En utilisation directe on appelle la fonction get_minority_instance(X,y) avec X dataframe des documents labellisés avec leurs features, y les labels correspondants (même index pour X et y)
On récupère X_sub et y_sub, sous ensemble des documents aux labels minoritaires.
En utilisation indirecte, dans la classe principale CorExBoosted, on appelle la méthode stratify() pour générer les attributs .X_sub et .y_sub, sinon lors du .fit l'argument optionnel stratify lance ou non la stratification (elle est systématiquement faite si on ne précise pas l'inverse)

La stratification consiste ici a sélectionner les exemples (avis, paragraphe, autre...) avec les les labels les moins représentés pour augmenter leur proportion par rapport a la distribution initiale, et avoir des proportions similaires pour tous les enjeux.

L'implémentation est assez simple : on calcule pour chaque label sa fréquence d'apparition et on fait la moyenne des fréquences : tous les labels sous représentés par rapport a la moyenne sont considérés comme minoritaires. On récupère ensuite un sous ensemble des données correspondant aux documents contenant au moins un enjeu minoritaire.

En revanche si des enjeux cooccurent souvent, comme c'est le cas pour la gestion des déchets avec d'autres enjeux par exemple, il sera difficile d'augmenter la proportion des enjeux sous representés pour la faire arriver au même niveau que les autres : il s'agit en fait d'un problème de machine learning en soit.

**Pistes d'améliorations** : essayer une autre méthode consistant a prendre uniquement les documents qui n'ont pas un trop grand nombre de label (2 par exemple) puis équilibrer à partir de ce sous ensemble. Le vocabulaire sera potentiellement plus spécifique.

#### 2.3.2 - Augmentation des données
<a name="a32"></a>

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
<a name="a4"></a>

**Objectif** : améliorer les performances globales
**Approches** : Bagging simple, boosting
Un des problèmes constatés est la grande variance de l'algo : pour deux tests a paramètres identiques (sans random seed) on a des résultats qui peuvent être radicalement différents en terme de performance.
Pour améliorer cela, on fait du bagging : on entraine un grand nombre de modèle pour arriver a la prédiction moyenne.


#### 2.4.1 - Bagging
<a name="a41"></a>

**Utilisation** :
Via la classe principale CorExBoosted, utiliser l'argument n_classif du .fit() ! n_classif correspond au nombre d'instances de CorEx qui vont êtres entrainées.
Si on ne stratifie pas et qu'on augmente pas les données, tout est entrainé sur le corpus de base
Si on stratifie sans augmenter, tout est entrainé sur le sous ensemble des documents aux labels minoritaires
Si on stratifie et qu'on augmente, chaque classifieur est entrainé sur un dataset augmenté différent.

Les performances sont légèrement meilleures (on augmente le F1 score moyen de 0,05), et cela diminue grandement la variance.
Un paramètre qui est ajouté par cette approche est la sélectivité du modèle : normalement, CorEx applique un seuil, lorsqu'il calcule la probabilité d'apparition d'un topic dans un document, si celle ci est supérieure à 0.5, il considère que l'enjeu est présent. Il est possible avec la méthode .predict() de changer la sélectivité (arg selectivity) et même de l'optimiser avec .optimize_selectivity(). Les bornes théoriques sont bien 0 et 1, mais si on n'encadre pas un peu plus finement celles ci, on peux obtenir des résultats absurdes (sélectivité de 0.005 par exemple...).

**Pistes d'améliorations** : tester ce qui se passe si on stratifie&augmente et qu'on entraine tous les classifs sur le même dataset augmenté, sinon boosting

#### 2.4.2 - Boosting
<a name="a42"></a>

Pour aller plus loin, des approches de boosting (bagging avec coefficients en fonction des performances de modèles) doivent être testées.
**Utilisation** : depuis la classe principale CorExBoosted, on appelle la méthode .optimize_weights(). On peux changer la méthode utilisée par scipy (voir la doc de minimize : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)

Déjà testé : scipy.optimize.minimize pour minimiser la Hamming Loss, la Ranking Label loss, une loss "artisanale" basée sur le score F1 moyen. Pas concluant du tout, les poids ne bougent pas du tout !

**Pistes d'améliorations** : Nouvelles approches ou algos a réimplémenter de zéro (AdaBoost, GradientBoost, Deep learning ?)


## 2.5 Suite du projet
<a name="a5"></a>

A la suite du projet, reste a faire :
- Implémenter certaines des pistes données pour améliorer les performances
- Démarrer un travail de correction des sorties sur les enjeux dans les sections pour constituer une base de donnée permettant la stratification des données sur les sections. Pour ça, exporter les sorties en excel en précisant chaque colonne, avec le texte associé, puis faire corriger le tableau aux experts métiers.
