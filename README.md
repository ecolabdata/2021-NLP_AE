# Projet de TLN français de l'Ecolab : sommaires détaillés, résumé automatique, similarité de documents (19/05/2021)

Bienvenue dans le référentiel consacré au projet de l'Ecolab sur le Traitement du Langage Naturel (français).

Vous pourrez trouver ici les codes finis et les codes exploratoires produits par l'équipe en charge du projet. Tout le projet est écrit en Python (ou Jupyter).

## Objectif : produire des outils, via du Machine Learning et du Natural Language Processing, pour aider les auditeurs qui élaborent les avis sur les études d'impact environnemental.

Les auditeurs de la DREAL, Bretagne en l'occurrence, doivent analyser et rendre un avis environnemental sur des études d'impacts. Ces études sont parfois très longues (en moyenne 300 pages, avec un écart-type de 400, certaines études font plus de 1000 pages) et contiennent de nombreuses annexes (plans, photos etc...).  
L'objectif de ce projet est de produire un outil, la __synthèse augmentée__, permettant d'avoir accès à un document court présentant les informations importantes de chaque paragraphe d'un document long et dense en information. Cette synthèse permet donc de jeter un coup d'oeil transversal rapide sur l'ensemble de l'étude, mais aussi d'y revenir lors du temps de l'analyse, de pouvoir comparer plus aisément au sein du même document mais aussi entre documents.  

Ce document synthétique se présente sous la forme d'un sommaire, où pour chaque titre est associé trois éléments : les enjeux importants, un résumé ainsi que des mots clés. Ce sommaire sera à terme navigable.

L'ensemble des travaux développés sont ré-utilisables et applicables à tout type de document, modulo un ré-entraînement potentiel pour une meilleure adéquation avec le corpus considéré. C'est d'ailleurs dans cet esprit que nous avons travaillé sur ce projet : permettre une bonne généralisation.
# Table des matières

Notre point de départ est donc l'ensemble des études d'impacts de la DREAL Bretagne, qui suivent un processus en plusieurs étapes :

1. [la détection et l'extraction des sommaire de documents PDF, puis le découpage des sections](#sommaire) ;
2. [le traitement et l'analyse des enjeux (via du **topic modelling**)](#enjeux);
3. [le résumé automatique de sections (via du **Deep Learning**, plus moins *supervisé*)](#resume).
4. [l'extraction de mots-clés (via du **Deep Learning**,**Graphs** et approches **Statistiques** *non supervisées*)](#motscles)

Enfin, une autre fonctionnalité développée est :

5. [un système de recommandation d'avis (fondé sur du **Deep Learning** et du **Collaborative Filtering** notamment )](#recommandation)

De manière visuelle, le pipeline du projet de __synthèse augmentée__ se présente comme suit :

<p align = 'center'> <img src="chaine.png"/> </p>

Pour le moment, il existe quatres dossiers :
* **Pipeline** : qui correspond aux chemins d'exécutions des travaux ayant été exécutés sur Dataiku. Notamment, l'analyse des enjeux, la détection et l'extraction des sommaires, le découpage des documents; la détection des avis vides.
* **Ruben** : tous les codes _exploratoires_ de Ruben Partouche (@rbpart).
* **Theo** : tous les codes _exploratoires_ de Théo Roudil-Valentin (@TheoRoudilValentin).
* **Zakaria** : tous les codes ainsi que la documentation liés à l'extraction de mots-clés et à la recommandation d'avis (@IIZCODEII).

Ces codes exploratoires sont parfois flous pour des personnes extérieures au projet, mais cela est normal. Les codes finaux seront mis dans **Pipeline** et expliqués par des documents et par des précisions dans le code lui-même. Les codes exploratoires sont laissés à but informatif.

Pour setup le projet, il faut exécuter le script setup.py depuis la racine du projet (2021-NLP_AE) qui va aussi chercher les dernières données (environ 10Go) sur le disque partagé du SRI (le renommer en K si ce n'est pas le cas pour votre ordinateur). Attention : pour le moment le setup va juste identifier les noms de fichier sans comparer le contenu : il faut supprimer les fichier puis relancer setup si on veux actualiser un fichier (amélioration a faire).

<a name="sommaire"/></a>
## 1 - Détection et extraction du sommaire
Afin de réaliser la __synthèse augmentée__, nous avions besoin d'avoir accès au sommaire. C'est grâce à l'extraction du sommaire que nous allions pouvoir découper les documents, puis pouvoir travailler sur les différentes sections en les analysant. Pour cela, il fallait d'abord les détecter.
Ce [dossier](./Pipeline/Extraction_sommaire) présente donc la démarche ainsi que les codes que nous avons utilisé pour détecter et extraire le sommaire des documents.
Dans la mesure du possible, nous avons tenté de reprendre ces codes pour qu'ils soient adaptables à d'autres documents. Il convient cependant de comprendre que ces codes ne sont pas encore totalement généralisables, et que nous ne sommes pas responsables des particularités d'autres documents qui viendraient empêcher le code de fonctionner.


<a name="enjeux"/></a>
## 2 - Traitement et analyse des enjeux
**Objectif** : identifier les enjeux présent dans un texte de longueur variable (idéalement le plus court possible)

Vous trouverez l'ensemble des travaux opérationnels dans Pipeline/Enjeux
Quelques travaux exploratoires et exemples d'utilisation sur les avis et les sections peuvent être trouvés dans le répertoire Ruben, voir les codes "Avis_semisupervised.py" et "Section_semisupervised.py" qui servent d'exemple d'utilisation du pipe.
Un README y est également présent pour expliquer les développements et explorations faites sur cette partie du projet.

<a name="resume"/></a>
## 3 - Résumé automatique de sections
**Objectif** : effectuer du résumé extractif (sélection des phrases pertinentes) sur les sections des études d'impacts.

Vous trouverez l'ensemble des travaux opérationnels et exploratoires dans le dossier [Théo](./Theo).
 
Nous développons quatre approches différentes qui sont :
* Une famille de modèles basées sur du __Deep Learning__
* Un modèle utilisant l'algorithme TextRank
* Un modèle basé sur la similarité de l'embedding des phrases
* Enfin une famille de modèle __benchmark__ pour la comparaison

Veuillez vous référer au dossier Théo pour y lire la note technique, ainsi qu'une explication plus précise et formelle dans le README disponible.

<a name="motscles"/></a>
## 4 - Extraction de mots-clés

**Objectifs** :
* Extraire des mots-clés pertinents sur les sections des études d'impacts (et tout autres documents) en l'absence de données labellisées et de recul sur le meilleur modèle pour un cas d'usage donné.
* Proposer un package directement actionnable pour toutes personnes/organisations voulant extraire des mots-clés sur un corpus de texte sans expertise prélable ni a priori sur les performances des nombreux modèles disponibles.

Pour une présentation détaillée du modèle *keyBoost* qui en découle , se référer au dossier *Zakaria*. Une documentation complète est disponible dans le sous-répertoire *Docs* de *Zakaria*. Cette document retrace le contexte, l'architecture technique, la validation scientifique de la pertience de *keyBoost*, un demonstrateur web interactif ainsi qu'un tutoriel/documentation sur le package python *keyBoost*.

<a name="recommandation"/></a>
## 4 - Système de recommandation d'avis
**Objectif** : formuler des recommandations d'avis correspondant à des études d'impacts similaires.




# **Contacts et citations :**
Ruben Partouche : ruben.partouche@student-cs.fr  
Théo Roudil-Valentin : theo.roudil-valentin@ensae.fr  
Zakaria Bekkar : zakaria.bekkar@ens-paris-saclay.fr  

