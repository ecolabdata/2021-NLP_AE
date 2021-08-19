# Dossier Théo - Résumé automatique

Date de la dernière modification : **18/08/2021**

Bienvenue dans le dossier de Théo Roudil-Valentin contenant tous les travaux concernant le résumé automatique.

Vous trouverez tous les éléments de codes permettant de produire des résumés.

# Table des matières
1. [Contenu du dossier](#contenu)
2. [Prise en main](#prise)
3. [Le résumé automatique](#resume)
    1. [Approches](#approches)  
        a.[Deep Learning Oriented Extractive Summarizer](#appDL)  
        b.[TextRank Extractive Summarizer](#appTR)  
        c.[BertScore](#appBS)  
        d.[Lead-3 & RandomSummary](#appL3)  
    2. [Résultats](#resultats)  
4. [Sources et citations](#source) 
5. [Contacts](#contact)

**French Automatic Text Summarizer (fats)**  
Le code __fats.py__ est le module regroupant un ensemble de classes et fonctions lié au projet notamment pour le nettoyage, la préparation du texte et le développement et l'application des modèles. Il est indispensable pour tous les fichiers qui se trouvent dans ce dossier. 


## Listes des codes et applications :
<a name="contenu"></a>

Ce dossier contient des codes et un dossier :
* **Exploration** : qui contient l'ensemble des codes préliminaires qui ont amené au travail abouti que vous avez ci plus haut. Je les laisse à but informatif et de compréhension.
* **Model** : qui contient certains des modèles nécessaires pour faire tourner les fonctions de résumés. Malheureusement, étant donné la limite d'espace de git, je n'ai pas pu mettre tous les modèles disponibles. Si vous faîtes partie du CGDD, vous pouvez y accéder via le fichier setup.py. Si vous faîtes partie du MTE, vous devez pouvoir en demander l'accès, ou demander à l'équipe de l'Ecolab de vous fournir les données. Si vous êtes extérieur, vous pouvez demander à l'équipe de vous les envoyer par un lien de téléchargement.

* __Note_technique.pdf__ : note concernant la stratégie envisagée pour le traitement Deep Learning du résumé, expliquant l'esprit et la méthode du travail. Elle n'est plus vraiment à jour, et à ce titre, les explications ci-dessous sont plus récentes, mais elle éclaire mieux de manière conceptuelle les problèmes que nous rencontrons et les éventuelles solutions. De même, une grande partie concerne les briques fondamentales des modèles que l'on va utiliser : les modèles BERT, l'attention, les transformers etc...
* __Paragraphes_exemple.pickle__ : exemple de paragraphes pour les deux pipelines résumés. 
* __pipeline_final.ipynb__ : code permettant de lancer la fonction finale de résumé. Vous avez un large choix de modèles, qui sont disponibles dans le dossier [Model](./Model). De plu vous avez un exemple présent dans ce dossier.
* __pipeline_DL.ipynb__ : code permettant de lancer la fonction de résumé Deep Learning.
* __fats.py__ : fichier **module**, c'est-à-dire comportant l'aboutissement de tout le travail fonctionnel sur le résumé. Il rassemble toutes les fonctions utiles pour cela. Il est appelé très souvent au sein des codes aboutis, donc pensez à bien le mettre dans votre dossier.

## Prise en main 
<a name="prise"></a>
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
<a name="resume"></a>

Le résumé pratiqué ici est extractif, en accord avec les auditeurs de la DREAL Bretagne. C'est-à-dire que l'on sélectionne les phrases les plus pertinentes de chaque paragraphe.

Nous avons développés ici 4 types de modèles. Certains sont assez simples, d'autres basés sur des techniques modernes de Deep Learning.  

### 1. Les approches et les méthodes associées
<a name="approches"></a>
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

#### 1.1 - Deep Learning Oriented Extractive Summarizer (DLOES)
<a name="appDL"></a>
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

Une fois ces vecteurs de tokens prêts, nous les insérons dans CamemBERT, puis prenons la dernière couche cachée, de dimension 512x768, que nous insérons ensuite dans les modèles développés ci-dessous.

Ceux-ci sont au nombre de 4. Ils sont relativement rudimentaires, par manque de temps, mais proposent et utilisent, pour certains, des méthodes assez modernes. Ils sont tous écrits en [torch](https://pytorch.org/) et sont tous disponibles dans le module __fats.py__ présenté plus haut.

* **Simple Linear Model** : une structure d'une seule couche linéaire suivie d'une LeakyRelu.
* **Multi Linear Model** : trois couches linéaires successives, dont les deux premières sont suivies d'une Softmax.
* **Convolutional Network** : une première couche de convolution (1D) suivie d'une LeakyRelu et d'un pooling, une deuxième convolution (1D) toujours associée à un LeakyRelu, deux couches linéaires et LeakyRelu pour finir sur une linéaire.
* **SelfMultiHeadAttention Model** : une couche de Self Multi Head Attention suivie d'une LayerNormalization, puis de deux couches linéaires toutes deux suivies d'une LeakyRelu

La fonction de perte est une perte [L1](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html) re-pondérée pour sur-pondérer les tokens de début de phrases importantes, qui sont donc généralement 3 sur une dimension de 512. Leur faible nombre m'a poussé à les surpondérer pour faire en sorte que le modèle se concentre sur ces tokens. 
L'optimiseur est un [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

Les modèles sont entraînés sur les données [MLSUM](https://github.com/huggingface/datasets/tree/master/datasets/mlsum), sur des batch de taille 64, sur un GPU NVIDIA disponible sur le datalab [Onyxia](https://datalab.sspcloud.fr/home) du [SSP Cloud](https://www.sspcloud.fr/).

#### 1.2 - TextRank for Extractive Summarizer (TRES)
<a name="appTR"></a>
Nous avons appliqué le modèle TextRank, dérivation du modèle PageRank de Google sur des données textuelles pour de l'extraction de mots-clés ou phrases, à nos données MLSUM. Ce dernier modèle est basé sur la théorie des graphes et propose d'étudier les relations entre différents objets. TextRank prend comme objets des phrases ou des mots par exemples. Dans notre cas, nous sommes intéressés par l'extraction de phrases. Donc nos objets, formellement nos sommets (__vertices__), sont donc nos phrases de paragraphe. L'algorithme va donc chercher des liens d'importance au sein d'un réseau. Avant toute chose il convient donc d'obtenir ce réseau.

Dans le cas de l'extraction de phrases, cela peut être une matrice de similarité entre les phrases. Pour cela nous avons besoin d'un embedding. Nous avons choisi deux embeddings différents, qui aboutissent donc à deux modèles différents : TRW (pour TextRankWord2Vec) basé sur l'embedding de Word2Vec, et TRB (pour TextRankBert), quant à lui basé sur CamemBERT. Une fois les embeddings obtenus, il suffit de calculer la similarité, cosinus dans notre cas, entre les phrases pour obtenir la matrice, soit le réseau.

Une fois le réseau obtenu, nous avons appliqué l'algorithme PageRank disponible via le module [networkx](https://networkx.org/). 

Dans notre cas, le graphe est pondéré, car la relation entre deux phrases n'est pas binaire, mais représenté la force de la similarité. Le score pondéré est introduit par les auteurs de TextRank (voir [Sources](#sources)).
#### 1.3 - BertScore
<a name="appBS"></a>
Ce modèle propose d'extraire les phrases les plus importantes de manière rudimentaire. Comme expliqué plus haut, les phrases doivent passer par un processus pour être transformés en vecteurs. D'abord la tokenization, puis l'embedding via CamemBERT. L'idée est de choisir les phrases qui représentent le mieux l'idée générale du paragraphe. En termes vectorielles, supposons qu'il existe un vecteur qui représente parfaitement __l'idée principale__ du paragraphe, on cherche à extraire les vecteurs qui sont les plus proches de ce __vecteur idée principale__ et ainsi faire émerger les phrases les plus importantes.  
La question désormais est : **comment obtenir ce vecteur idée générale** ? Il convient d'en chercher une approximation _suffisante_, c'est-à-dire suffisamment bonne pour que les résumés aient du sens et soient utilisables. Notre proposition est d'utiliser la moyenne des représentations vectorielles des phrases comme approximation de l'idée générale.   
Ce modèle repose donc sur l'hypothèse suivante : **la moyenne des représentations vectorielles des phrases constitue une approximation suffisamment bonne du vecteur de l'idée principale.**

Une fois ce proxy obtenu, nous pouvons calculer la similarité cosinus de chaque phrase à cette _idée principale_. Le résumé consiste alors en les phrases les plus proches de cette dernière.

#### 1.4 - Lead-3 et RandomSummary
<a name="appL3"></a>
Le premier consiste simplement à prendre les trois premières phrases d’un paragraphe. L’hypothèse sous jacente est que l’information importante est souvent énoncée dès le début d’un paragraphe et ce pour annoncer au lecteur la teneur principale du paragraphe considéré.
Le second, RandomSummary (RS), consiste à sélectionner les phrases aléatoirement dans le paragraphe. Il constitue ainsi un le premier plancher que les différents modèles précédents devront surpasser. En effet, ces derniers doivent, pour être intéressants, être plus performant que le hasard. Le modèle Lead-3 constitue donc un second plancher qualitatif à dépasser.

### 2. Résultats 
<a name="resultats"></a>
|Métriques| L3  | RS  | TRW  | TRB  | BS  | Multi | Simple | SMHA | Net |
|---|---|---|---|---|---|---|---|---|---|
True Positive Mean|0.2291|0.1726|0.2166|0.1686|0.1517|0.2029|0.2015|0.1696|0.1812|
False Positive Mean| 0.1273  | 0.1275 | 0.1254  |  0.1364 | 0.1335  |0.1333|0.1325|0.1389|0.1367|
False Negative Mean|  0.7709 |  0.8274 | 0.7834  | 0.8314  |  0.8483 |0.7971|0.7985|0.8304|0.8188|
Precision| 0.2291  | 0.1858  | 0.2166  |  0.1686 | 0.1517  |0.2029|0.2015|0.1696|0.1812|
Recall   |0.2291|0.1726|0.2166|0.1686|0.1517|0.2029|0.2015|0.1696|0.1812|
F1       |0.2291|0.1770|0.2166|0.1686|0.1517|0.2029|0.2015|0.1696|0.1812|
Rang |1|6|2|8|9|3|4|7|5|

## Sources et citations
<a name="sources"/></a>

[Camembert: a tasty french language model](https://arxiv.org/abs/1911.03894)  
[SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)  
[MLSUM: The Multilingual Summarization Corpus](https://arxiv.org/abs/2004.14900)  
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)  
[Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)  
[TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252.pdf)  
[The anatomy of a large-scale hypertextual Web search engine (PageRank)](https://www.sciencedirect.com/science/article/pii/S016975529800110X)  
[Neural Machine Translation by Jointly Learning to Align and Translate (Attention)](https://arxiv.org/abs/1409.0473)  
[Attention Is All You Need (Transformers)](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)  
## Contacts
<a name="contact"></a>
[Théo Roudil-Valentin](mailto:theo.roudil-valentin@ensae.fr)
