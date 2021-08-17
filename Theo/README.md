# Dossier Théo - Résumé automatique

Date de la dernière modification : **03/08/2021**

Bienvenue dans le dossier de Théo Roudil-Valentin contenant tous les travaux concernant le résumé automatique.

Vous trouverez tous les éléments de codes permettant de produire des résumés.

**French Automatic Text Summarizer (fats)**  
Le code __fats.py__ est le module regroupant un ensemble de classes et fonctions lié au projet notamment pour le nettoyage, la préparation du texte et le développement et l'application des modèles. Il est indispensable pour tous les fichiers qui se trouvent dans ce dossier. 

Ce dossier contient des codes et un dossier :
* **Exploration** : qui contient l'ensemble des codes préliminaires qui ont amené au travail abouti que vous avez ci plus haut. Je les laisse à but informatif et de compréhension.

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
N'oubliez cependant pas que tous les '\\' doivent être soit remplacés par des '\\\\' ou des '/' ! Sinon python ne pourra pas trouver le chemin.  

Une fois que cela est fait, vous pouvez lancer les exemples.  
Vous pourrez de même introduire vos propres données (attention au format et à la dimension, mais cela est indiqué dans les pipelines).

## L'approche du résumé automatique

petite intro

### 1. Les approches et les méthodes associées

* **1.1** Une famille de modèles basées sur du __Deep Learning__
* **1.2** Un modèle utilisant l'algorithme TextRank
* **1.3** Un modèle basé sur la similarité de l'embedding des phrases
* **1.4** Enfin une famille de modèle __benchmark__ pour la comparaison

Avant de développer des modèles d'extraction, il convient de nettoyer les phrases des paragraphes que l'on va tenter de résumé automatiquement. En effet, ces phrases sont pleines de marques de ponctuations, d'accents etc... qui peuvent venir rendre plus difficile l'apprentissage ou l'inférence des modèles.  
C'est pourquoi nous avons enlevé les éléments suivants :
* les accents
* la ponctuation
* les chiffres
* les articles et autres mots __vides__ (c'est-à-dire présent trop souvent pour apporter de l'information)
* 

#### 1.1 - Deep Learning Oriented Extractive Summarizer (DLOES)
Ces dernières années, les techniques de Deep Learning appliquées au traitement du langage naturel se sont largement développés et proposent des outils désormais très puissants. Dans cette même veine, l'Ecolab a décidé de tenter la création d'un modèle de Deep Learning pour extraire les phrases importantes d'un paragraphe.  

Le travail débute par plusieurs étapes de pre-processing :

1. Tout d'abord, il convient __tokenizer__ les mots, c'est-à-dire de les couper en bouts (des __tokens__) encore plus petits. Pour cela on utilise un [_tokenizer_](https://github.com/google/sentencepiece), c'est-à-dire un modèle capable de repérer et découper les mots au bonne endroit. 
2. Ensuite, nous avons transformé ces listes de tokens en vecteur via un _embedding_, celui du modèle [CamemBERT](https://huggingface.co/transformers/model_doc/camembert.html).
3. Une fois ces représentations des phrases sous forme de vecteur récupérées, nous pouvons les introduire dans les différents modèles de DL que nous avons construit.

Une des limites des modèles [BERT](https://github.com/google-research/bert) est la dimension fixe des objets en entrée : 512. Autrement dit, ces modèles qui reçoivent des vecteurs de 512 tokens, nous obligent par la même occasion à contraindre nos phrases à faire 512 tokens. Cependant, il n'y a, a priori, aucune raison pour que ce soit le cas, le nombre des tokens des phrases présentes dans nos paragraphes n'a aucune raison d'être égal ou inférieur à 512. Par conséquent, il faut trouver un moyen pour outrepasser cette limitation.  
Il est par exemple possible de couper le paragraphe et ne prendre que ses 512 premiers tokens, mais ce serait une grande limitation, et occulterait une trop grande partie de l'information disponible. Comme notre recherche a pour but le développement d'un produit utilisable par les auditeurs de la DREAL Bretagne, nous ne pouvons nous permettre une telle perte.  
Une seconde approche serait de couper en parties égales les paragraphes, pour avoir des vecteurs possédant la même taille par paragraphes, ou une taille proche. Mais cela éloignerait et découperait trop l'information qui serait disparate.  
Enfin, une autre approche, celle que nous avons choisi, est celle de concaténer les tokens des phrases jusqu'à atteindre la phrase dont les tokens feraient basculer le vecteur à une dimension supérieure à 512. Par exemple, soit un paragraphe composé de 10 phrases, nous tokenisons au fur et à mesure les phrases lorsque à la phrase 6, la somme des tokens est de 509. Nous sommes donc très proche de 512. Disons que la phrase 7 possède 15 tokens, si on l'ajoute au vecteur présent nous aurons une dimension de 509+15>512. On clot donc le vecteur présent, en le pavant de 1 jusqu'à 512 (soit trois tokens 1) puis on crée un nouveau vecteur, auquel on ajoutera la phrase 7 ainsi que les suivantes, ainsi de suie.  
Le pavement de 1 peut paraître étonnant, mais c'est quelque chose de très classique dans les modèles BERT, où lorsque il n'y a pas assez de tokens pour atteindre 512, on rajoute des 1 pour indiquer au modèle qu'ils ne sont pas des vrais tokens informatifs. D'ailleurs, l'entrée de ces modèles n'est pas constituée que d'un vecteur de tokens, mais également d'un masque composé de 0 et de 1 pour de nouveau indiquer au modèle les tokens (indices ici) qui sont informatifs et les autres qui sont ici pour paver, c'est-à-dire remplir les trous. Par exemple dans notre cas, le vecteur de dimension 509 aura un masque associé de 1 en position de 0 à 509, puis de 0 de 510 à 512.  
Par conséquent, certains paragraphes sont donc séparés sur plusieurs vecteurs. Bien entendu, cette approche à des limites, qui sont proches de la seconde approche proposée. Nous en sommes conscients, mais c'est pour nous l'approche qui est le meilleur compromis.

Les modèles que nous avons développé sont au nombre de 4. Ils sont relativement rudimentaires, par manque de temps, mais proposent et utilisent, pour certains, des méthodes assez modernes. Ils sont tous écrits en [torch](https://pytorch.org/) et sont tous disponibles dans le module __fats.py__ présenté plus haut.

* **Simple Linear Model** : une structure d'une seule couche linéaire suivie d'une LeakyRelu.
* **Multi Linear Model** : trois couches linéaires successives, dont les deux premières sont suivies d'une Softmax.
* **Convolutional Network** : une première couche de convolution (1D) suivie d'une LeakyRelu et d'un pooling, une deuxième convolution (1D) toujours associée à un LeakyRelu, deux couches linéaires et LeakyRelu pour finir sur une linéaire.
* **SelfMultiHeadAttention Model** : une couche de Self Multi Head Attention suivie d'une LayerNormalization, puis de deux couches linéaires toutes deux suivies d'une LeakyRelu

La fonction de perte est une perte L1 re-pondérée pour sur-pondérer les tokens de début de phrases importantes, qui sont donc généralement 3 sur une dimension de 512. Leur faible nombre m'a poussé à les surpondérer pour faire en sorte que le modèle se concentre sur ces tokens. 

Les modèles sont entraînés sur les données [MLSUM](https://github.com/huggingface/datasets/tree/master/datasets/mlsum), sur des batch de taille 64, sur un GPU NVIDIA disponible sur le datalab [Onyxia](https://datalab.sspcloud.fr/home) du [SSP Cloud](https://www.sspcloud.fr/).

#### 1.2 - TextRank for Extractive Summarizer (TRES)
#### 1.3 - BertScore
Ce modèle propose d'extraire les phrases les plus importantes de manière rudimentaire. Comme expliqué plus haut, les phrases doivent passer par un processus pour être transformés en vecteurs. D'abord la tokenization, puis l'embedding via CamemBERT. L'idée est de choisir les phrases qui représentent le mieux l'idée générale du paragraphe. En termes vectorielles, supposons qu'il existe un vecteur qui représente parfaitement _l'idée générale_ du paragraphe, on cherche à extraire les vecteurs qui sont les plus proches de ce _vecteur idée générale_ et ainsi faire émerger les phrases les plus importantes.  
La question désormais est : **comment obtenir ce vecteur idée générale** ? Il convient d'en chercher une approximation _suffisante_, c'est-à-dire suffisamment bonne pour que les résumés aient du sens et soient utilisables. Notre proposition est d'utiliser la moyenne des représentations vectorielles des phrases comme approximation de l'idée générale.   
Ce modèle repose donc sur l'hypothèse suivante : **la moyenne des représentations vectorielles des phrases constitue une approximation suffisamment bonne du vecteur de l'idée générale.**


#### 1.4 - Lead-3 et RandomSummary

### 2. Résultats 

## Citation

## Sources :
[Camembert: a tasty french language model](https://arxiv.org/abs/1911.03894)  
[SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)  
[MLSUM: The Multilingual Summarization Corpus](https://arxiv.org/abs/2004.14900)  
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
