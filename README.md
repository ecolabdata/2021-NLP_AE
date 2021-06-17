# Projet de TLN français de l'Ecolab : sommaires détaillés, résumé automatique, similarité de documents (19/05/2021)

Bienvenue dans le référentiel consacré au projet de l'Ecolab sur le Traitement du Langage Naturel (français).

Vous pourrez trouver ici les codes finis et les codes exploratoires produits par l'équipe en charge du projet. Tout le projet est écrit en Python (ou Jupyter).

## Objectif : produire des outils, via du Machine Learning et du Natural Language Processing, pour aider les auditeurs qui élaborent les avis sur les études d'impact environnemental.

En juin, le projet est divisé en plusieurs parties :
1. une première partie sur la détection et l'extraction des sommaire de documents PDF, puis leur découpage;
2. une partie sur le traitement et l'analyse des enjeux (via du **topic modelling**);
3. une partie sur le résumé automatique de sections (via du **Deep Learning**, plus moins *supervisé*).
4. une dernière partie sur un système de recommandation d'avis (fondé sur du **Deep Learning** et du **Collaborative Filtering** notamment )

Pour le moment, il existe quatres dossiers :
* **Pipeline** : qui correspond aux chemins d'exécutions des travaux ayant été exécutés sur Dataiku. Notamment, l'analyse des enjeux, la détection et l'extraction des sommaires, le découpage des documents; la détection des avis vides.
* **Ruben** : tous les codes _exploratoires_ de Ruben Partouche (@rbpart).
* **Theo** : tous les codes _exploratoires_ de Théo Roudil-Valentin (@TheoRoudilValentin).
* **Zakaria** : tous les codes _exploratoires_ de Zakaria Bekkar (@IIZCODEII).

Ces codes exploratoires sont parfois flous pour des personnes extérieures au projet, mais cela est normal. Les codes finaux seront mis dans **Pipeline** et expliqués par des documents et par des précisions dans le code lui-même.

Pour setup le projet, il faut exécuter le script setup.py depuis la racine du projet (2021-NLP_AE) qui va aussi chercher les dernières données (environ 10Go) sur le disque partagé du SRI (le renommer en K si ce n'est pas le cas pour votre ordinateur). Attention : pour le moment le setup va juste identifier les noms de fichier sans comparer le contenu : il faut supprimer les fichier puis relancer setup si on veux actualiser un fichier (amélioration a faire).

## 1 - Détection et extraction du sommaire
### 1.1 - Construction de la base HTML
### 1.2 - Création des variables (feature engineering)
### 1.3 - Les étapes de la détection
### 1.4 - Extraction et découpage

## 2 - Traitement et analyse des enjeux
**Objectif** : identifier les enjeux présent dans un texte de longueur variable (idéalement le plus court possible)
### 2.1 -

## 3 - Résumé automatique de sections
**Objectif** : effectuer du résumé extractif (sélection des phrases pertinentes) sur les sections des études d'impacts.
### 3.1 - Approches
Nous développons quatre approches différentes qui sont :

### 3.2 - Un peu de méthode
Pour des explications plus précises, voir _Note technique_ dans Theo.
### 3.3 - Résultats


## 4 - Système de recommandation d'avis
**Objectif** : formuler des recommandations d'avis correspondant à des études d'impacts similaires.




**Contacts :**
Ruben Partouche : ruben.partouche@student-cs.fr
Théo Roudil-Valentin : theo.roudil-valentin@ensae.fr
Zakaria Bekkar : zakaria.bekkar@ens-paris-saclay.fr
