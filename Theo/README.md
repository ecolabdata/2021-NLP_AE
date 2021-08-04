# Dossier Théo - Résumé automatique

Date de la dernière modification : **03/08/2021**

Bienvenue dans le dossier de Théo Roudil-Valentin contenant tous les travaux concernant le résumé automatique.

Vous trouverez tous les éléments de codes permettant de produire des résumés.

Le code __fats.py__ est le module regroupant un ensemble de classes et fonctions lié au projet notamment pour le nettoyage, la préparation du texte et le développement et l'application des modèles.

Ce dossier contient deux dossiers :
* **Exploration** : il contient l'ensemble des codes préliminaires qui ont amené au travail abouti que vous avez ci plus haut. Je les laisse à but informatif et de compréhension.
* **Model** : contient les modèles pour faire tourner correctement les pipelines et plus généralement les fonctions de résumé.

## Listes des codes et applications :

* __Note_technique.pdf__ : note concernant la stratégie envisagée pour le traitement Deep Learning du résumé, expliquant l'esprit et la méthode du travail.
* __pipeline_final.ipynb__ :
* __pipeline_DL.ipynb__ :
* __fats.py__ : fichier **module**, c'est-à-dire comportant l'aboutissement de tout le travail fonctionnel sur le résumé. Il rassemble toutes les fonctions utiles pour cela. Il est appelé très souvent au sein des codes aboutis, donc pensez à bien le mettre dans votre dossier.

## Prise en main 

Pour prendre en main ce dossier vous devez :
* **1.** d'abord cloner le repository en local (dans votre invite de commandes windows, mettez vous dans votre dossier choisi et entrez : **git clone https://github.com/ecolabdata/2021-NLP_AE.git** , attention au proxy si vous êtes au bureau 😉 ! ) ;
* **2.** avoir installé une version de python (conseil : la 3.6.7 64-bit) ;
* (**2.1** conseil également : créez-vous un environnement virtuel, c'est plus sain, vous pouvez chercher sur le net c'est assez direct.) ;
* **3.** avoir installé une interface python : par exemple Jupyter (classique), je conseille VS Code, qui est vraiment très utile, agréable et polyvalent ;
* **4.** avoir installé les packages du document __requirements.txt__ disponible ;

Une fois que vous avez fait cela, vous avez (normalement) tout bon pour lancer le projet et ses exemples.  
Vous pouvez alors ouvrir votre interface python (Jupyter, VS Code ou autre), puis lancer les cellules (Ctrl + Enter ou shift + enter).  
Vous devrez probablement à un moment donné renseigner votre chemin vers votre dossier ou vous avez téléchargé les données, l'emplacement sera indiqué directement dans le code.  
N'oubliez cependant pas que tous les '\' doivent être soit remplacés par des '\\' ou des '/' ! Sinon python ne pourra pas trouver le chemin.  

Une fois que cela est fait, vous pouvez lancer les exemples.  
Vous pourrez de même introduire vos propres données (attention au format et à la dimension, mais cela est indiqué dans les pipelines).

## L'approche du résumé automatique
### 1. Les approches et les méthodes associées
#### 1.1 - Deep Learning Oriented Extractive Summarizer (DLOES)
#### 1.2 - TextRank for Extractive Summarizer (TRES)
#### 1.3 - BertScore
#### 1.4 - Lead-3 et RandomSummary
### 3. Résultats 

## Citation

## Sources :
