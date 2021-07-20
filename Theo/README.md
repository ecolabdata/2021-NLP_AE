# Dossier Théo - Résumé automatique

Bienvenue dans le dossier de Théo Roudil-Valentin contenant tous les travaux concernant le résumé automatique.

Vous trouverez tous les éléments de codes permettant de produire des résumés.

**Attention** : beaucoup de ces codes sont **EXPLORATOIRES**. C'est-à-dire qu'ils ont eu vocation à créer le projet, pas à être généraliser. Ces codes seront rangés dans un dossier spécifique à but **informatif** uniquement. 

Le code __fats.py__ est le module regroupant un ensemble de classes et fonctions lié au projet notamment pour le nettoyage, la préparation du texte et le développement et l'application des modèles.

Tous les modèles ou données cités sont sourcés en bas de la page, n'hésitez pas à vous y référer au besoin.

## Listes des codes et applications :

* __BARThez_BERTSUM.py__ : code important pour la compréhension des dimensions de l'entrée des modèles Deep Learning. Ce code produit le tokenizer à partir des données OrangeSum puis introduit une fonction permettant de sortir les tenseurs liés aux articles des données. En d'autres termes, il permet de comprendre l'esprit de l'input des modèles DL. L'output associé est déjà créé (puisqu'il est téléchargé, voir le code __creation_output_orangesum_ext.py__). Il met l'ensemble dans un dictionnaire. Les données sont ensuite testées (la structure/dimension) sur différents modèles de DL. Enfin on jette un coup d'oeil aux données BERTSum et au modèle BARThez.
* __BERT_Score.ipynb__ : Jupyter introduisant le modèle **BertScore** et notamment son esprit (l'idée générale du paragraphe via le proxy de la moyenne des représentations vectorielles des phrases).
* __BERT_essai_1.py__ : prise en main des modèles Bert/CamemBERT, des modules torch et fairseq. Petite analyse du nombre de tokens par phrase : c'est-à-dire en combien de bout les tokenizers classiques coupent nos phrases (ceux des études d'impacts). Essai de concaténation à la main de tokens : pas concluants, quelque chose de plus simple est proposé dans __BARThez_BERTSUM.py__ (et finalement dans __fats.py__). Prise en main des tokenizers vierges (à entraîner) puis essais d'application CamemBERT.
* __BERT_fine_tuning_1.py__ : différents essais de __réglage de précision (fine-tuning)__ de Camembert dans une optique non-supervisé (**non-concluant**).
* __BERT_from_scratch.py__ : ébauche de création d'un modèle BERT pour le MTE (**non-concluant**).
* __BERT_nonsupervise.py__ : Jupyter ébauche de modèles K-Means et Gaussian Mixture sur l'embedding fourni par CamemBERT (non poursuivi).
* __BERT_score.py__ : première ébauche du code __BERT_Score.ipynb__, veuillez vous y référer directement.
* __BertSum.py__ : briques de modèles provenant directement de https://github.com/nlpyang/BertSum/blob/master/src/models/encoder.py.
* __FUES.model__ : tokenizer entraîné sur les données OrangeSum (voir __BARThez_BERTSUM.py__).
* __FUES.vocab__ : vocabulaire du tokenizer entraîné sur les données OrangeSum (voir __BARThez_BERTSUM.py__).
* __MLSUM_prep.ipynb__ :
* __MLSUM_prep.py__ :
* __Note_technique.pdf__ :
* __Stat_sommaire.py__ :
* __TextRank_essai_1.py__ :
* __compter_pages.py__ :
* __crea_base.py__ :
* __creation_output_orangesum_ext.py__ :
* __decoupage_essai.py__ :
* __diving_into_html.py__ :
* __essai_DL.py__ :
* __essai_en_tout_genres.py__ :
* __fats.py__ :
* __fonction_mots.py__ :
* __gpu_essai_1.ipynb__ :
* __make_metrics.py__ :
* __make_train_dataset.py__ :
* __resume.py__ :
* __tentative_scraping_1.py__ :
* __test_dl.ipynb__ :
* __topic_modelling.py__ :
* __torch_tutorial.py__ :
* __training_dl.ipynb__ :


## Sources :
* **OrangeSum** : https://github.com/Tixierae/OrangeSum
* **BertSum** : https://github.com/nlpyang/BertSum
* **BARThez** : https://github.com/moussaKam/BARThez
* **BERT** : https://github.com/google-research/bert
* **CamemBERT** : https://huggingface.co/transformers/model_doc/camembert.html
