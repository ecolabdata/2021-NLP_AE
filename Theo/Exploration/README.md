# Dossier Exploration - Résumé automatique

Date de la dernière modification : **26/07/2021**

Bienvenue dans le dossier d'Exploration de Théo Roudil-Valentin contenant tous les travaux préliminaires concernant le résumé automatique.

Vous trouverez tous les éléments de codes permettant de produire des résumés.

**Attention** : la totalité de ces codes sont **EXPLORATOIRES**. C'est-à-dire qu'ils ont eu vocation à créer le projet, pas à être généraliser. Ces codes seront rangés dans un dossier spécifique à but **informatif** uniquement. 

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
* __MLSUM_prep.ipynb__ : Jupyter pour la préparation et le nettoyage des données MLSUM et notamment leur adaptation à notre cas de résumé extractif pour __Deep Learning__. 
* __MLSUM_prep.py__ : Téléchargement des données MLSUM, découpage pour préparation plus rapide des données (car très volumineuses) puis différentes boucles de création d'output pour le résumé extractif.
* __Pipe_resume_vf.py__ :
* __Stat_sommaire.py__ : code permettant de produire quelques statistiques rapides sur les données disponibles et retravaillées des études d'impact.
* __TextRank_essai_1.py__ : code de création et de développement du modèle de résumé TextRank avec deux embedding différents.
* __compter_pages.py__ : quelques statistiques sur le nombre de pages des documents présents.
* __crea_base.py__ : code de création de la base de données des html de toutes les études OCRisées, une ligne par html et gestion des études mal OCRisées.
* __creation_output_orangesum_ext.py__ : préparation des données OrangeSum et notamment création de l'output pour utilisation en résumé extractif.
* __decoupage_essai.py__ : essai de découpages des études une fois le sommaire associé récupéré.
* __diving_into_html.py__ : premiers travaux sur la matière brut des études d'impact en html, notamment recherche de balises etc...
* __essai_DL.py__ : quelques essais de deep learning en Keras et TensorFlow.
* __essai_en_tout_genres.py__ :
* __fonction_mots.py__ : premiers travaux sur les thèmes des documents et élaboration de fonctions associées.
* __gpu_essai_1.ipynb__ : premier jupyter concernant l'entraînement sur GPU des modèles de deep learning. 
* __make_metrics.py__ : code produisant les sorties de tous les modèles pour pouvoir ensuite les comparer (contient aussi des corrections de certaines fonctions).
* __make_train_dataset.py__ : code permettant de faire rapidement les __data loader__ (des objets torch pour l'entraînement des modèles DL).
** __pipeline_resume.py__ : code présentant la pipeline résumé avec un exemple fonctionnel.
* __resume.py__ : code produisant les exemples de résumé sur les véritables données DREAL (notamment ceux pour l'atelier ayant eu lieu à Rennes le 25 juin 2021).
* __tentative_scraping_1.py__ : scraping du site projet-environnement-diffusion (des études d'impact) ainsi que du Thésaurus sur LegiFrance.
* __test_dl.ipynb__ : deuxième jupyter pour l'entraînement GPU des modèles DL, un peu plus light et direct.
* __topic_modelling.py__ : travaux sur les thèmes des études, suite de __fonction_mots.py__.
* __torch_tutorial.py__ : quelques essais pour comprendre un petit peu l'esprit de torch.
* __training_dl.ipynb__ : troisième (et dernier) jupyter pour l'entraînement GPU des modèles DL, plus light et direct, entraînement en boucle de plusieurs modèles.


## Sources :
* **OrangeSum** : https://github.com/Tixierae/OrangeSum
* **BertSum** : https://github.com/nlpyang/BertSum
* **BARThez** : https://github.com/moussaKam/BARThez
* **BERT** : https://github.com/google-research/bert
* **CamemBERT** : https://huggingface.co/transformers/model_doc/camembert.html
* **projet environnement diffusion** : https://www.projets-environnement.gouv.fr/explore/dataset/projets-environnement-diffusion/table/?disjunctive.dc_subject_category&disjunctive.dc_subject_theme&disjunctive.vp_status&disjunctive.dc_type
