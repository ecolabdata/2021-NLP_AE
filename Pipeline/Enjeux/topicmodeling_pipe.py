#%%
class CorExBoosted():
    def __init__(self,docs,thesaurus,model = None):
        super(CorExBoosted)
        """
        docs = DataFrame contenant une colonne "id" correspondant
        a l'id donné par l'AE, et une colonne "text_processed" avec le texte
        du document préprocessed (tokenisé, nettoyé, etc...) sous forme d'une string unique
        """
        self.docs = docs
        self.Thesaurus = thesaurus
        from nltk.corpus import stopwords
        self.stop_words = stopwords.words('french')
        import numpy as np
        if model is not None:
            self.models = model.models
            self.vectorizer = model.vectorizer

    def preprocess(self,colname):
        """
        colname = nom de la colonne a preprocess sur les documents ayant permis l'initialisation de l'instance.
        """
        from Pipeline.Enjeux.processing_encoding import processing
        from tqdm import tqdm
        tqdm.pandas(desc= 'preprocessing...')
        self.docs['text_processed'] =self.docs[colname].progress_apply(processing)
        return self

    def encode(self, min_df = 13,max_df = 0.95, ngram_range=(1,3), stop_words = None):
        """
        CountVectorizer pré-réglé. Voir la doc de countVectorizer
        Génère les attributs associés au countVecto et les attache directement a l'instance de CorExBoosted (pas optimisé en mémoire).
        """
        import numpy as np
        if stop_words == None:
            stop_words = self.stop_words
        else:
            self.stop_words = stop_words
        from sklearn.feature_extraction.text import CountVectorizer
        from Pipeline.Enjeux.processing_encoding import get_info

        countVecto = CountVectorizer(min_df = min_df, max_df = max_df, ngram_range=ngram_range, stop_words = stop_words)
        process = countVecto.fit_transform(self.docs['text_processed'].values)  
        X = process.toarray().astype(int)
        self.X = np.matrix(X)
        self.word2id,self.vocab,self.words_freq,self.vocab_sort,self.notinvoc = get_info(countVecto,self.X,self.Thesaurus)
        
        self.enjeux_list = self.Thesaurus.Enjeux.values.tolist()
        self.vectorizer = countVecto
        thesau_list = []
        for enj in self.Thesaurus.Dictionnaire.values.tolist():
            ll = [w for w in enj if w in self.vocab]
            thesau_list.append(ll)
        self.thesau_list = thesau_list

        self.dico_thesau = {e:d for e,d in zip(self.enjeux_list,self.thesau_list)}
        return self
    
    def diagnostic(self,max_df=True,min_df=True,N=15):
        """
        Sors des graphiques et informations aidant le choix des paramètres max_df et min_df du vectoriseur (couverture du Thésaurus).
        """
        from Pipeline.Enjeux.processing_encoding import analysis_min_df,analysis_max_df
        resultats = []
        if max_df:
            couvertures_max_df = analysis_max_df(self.docs.text_processed.values,self.Thesaurus,N=N) #Retourne la couverture pour chaque enjeu en faisant varier le paramètre max_df de 1 a 0,85
            resultats.append(couvertures_max_df)
        if min_df:
            couvertures_min_df = analysis_min_df(self.docs.text_processed.values,self.Thesaurus,N=N) #Retourne la couverture pour chaque enjeu en faisant varier le paramètre min_df de 0 a 15
            resultats.append(couvertures_min_df)
        return(resultats)

    def stratify(self,df_corrige):
        """
        df_corrige = dataframe avec, au moins, une colonne 'id_AAE', et dont les ***dernières*** colonnes correspondent au True/False pour chaque enjeu.
        """

        import numpy as np
        import pandas as pd
        from Pipeline.Enjeux.utils import separate
        #On récupère les y labellisés (vrais), et X sur les mêmes indices
        y_true,X_sub = separate(self.docs,self.X,df_corrige=df_corrige)
        y_true_df = pd.DataFrame(y_true,columns = self.enjeux_list)
        X_df = pd.DataFrame(X_sub,columns=self.vocab)
        from Pipeline.Enjeux.multilabel_balancing import get_minority_instance
        #Stratification a partir des éléments labellisés
        #On récupère un sous ensemble de ces X 
        # avec des proportions équilibrées pour chaque enjeu
        self.X_sub, self.y_sub = get_minority_instance(X_df, y_true_df)
    
    def augment(self,n_sets,n_samples = 50):
        """
        Entrée :
        n_sets = nombre de datasets a générer. A tester : possibilité d'utiliser le même dataset pour tous les CorEx.
        Optionnel : 
        n_samples = nombre de samples a rajouter au dataset stratifié (X_sub,y_sub sous ensemble des données plus équilibrés en topics) utilisé
        """
        from Pipeline.Enjeux.multilabel_balancing import MLSMOTE2
        self.datasets = []
        for j in range(n_sets):
            self.datasets.append(MLSMOTE2(self.X_sub,self.y_sub,n_samples))
        return(self)

    def fit_one(self,model,X,strength):
        """
        Utilitaire pour fit
        """
        model.fit(X,words=self.vocab, anchors=self.thesau_list, anchor_strength=strength)
        return(model)
    
    def fit(self,n_classif = 100,stratify = True,augment = True, n_samples = 50,strength = 1,df_corrige = None):
        """
        Fonction la plus intéressante a paramétrer. Tous les arguments sont optionnels.
        n_classif = nombre de classifieurs pour le bagging. Si n = 1, correspond au cas sans bagging
        stratify = déclenche la stratification. Voir la doc de la méthode stratify. Nécessite un argument pour df_corrige.
        augment = déclenche l'augmentation de données. Voir la doc de la méthode augment. Nécessite un df_corrige
        n_samples = argument de augment. Voir la doc de la methode augment.
        strength = poids appliqué au thésaurus dans CorEx. Voir la doc de CorEx.
        df_corrige = dataframe des données corrigées. Voir la doc de la méthode stratify.
        """
        from joblib import Parallel,delayed
        from corextopic import corextopic as ct
        import os
        import numpy as np
        storage = []
        for j in range(n_classif):
            storage.append(ct.Corex(n_hidden=len(self.enjeux_list)))
        if stratify and augment and df_corrige is not None:
            self.stratify(df_corrige = df_corrige)
            self.augment(n_sets = n_classif,n_samples = n_samples)
            storage = [(model,data) for model,data in zip(storage,self.datasets)]
            storage_fitted = Parallel(n_jobs=os.cpu_count(),verbose=0)(delayed(self.fit_one)(item[0],np.matrix(item[1][0]),strength) for item in storage)
            self.models = storage_fitted
            return self
        if stratify and not augment and df_corrige is not None:
            self.stratify(df_corrige = df_corrige)
            data = (self.X_sub,self.y_sub)
            storage_fitted = Parallel(n_jobs=os.cpu_count(),verbose=0)(delayed(self.fit_one)(item,np.matrix(data[0]),strength) for item in storage)
            self.models = storage_fitted
            return self
        else:
            storage_fitted = Parallel(n_jobs=os.cpu_count(),verbose=0)(delayed(self.fit_one)(model,self.X,strength) for model in storage)
            self.models = storage_fitted
            return self

    def predict(self,X,weights = None, selectivity = 0.5, n=None):
        """
        Entrée :
        X = Matrice numpy (n_docs,n_features)
        Optionnel:
        weights = liste de scalaires de taille nombre de classifieurs (défini dans fit). Doit sommer a 1 (combinaison de proba de tous les classifieurs)
        selectivity = scalaire entre 0 et 1
        n = nombre de classifieurs utilisés si on ne souhaite pas tous les prendre
        Sortie : 
        Matrice numpy True/False (n_docs,n_topics)
        """
        import numpy as np
        if n is None:
            n = len(self.models)
        bagofclassif = self.models[:n]
        labels_moy = np.zeros(bagofclassif[0].predict_proba(X)[0].shape)
        if weights is None:
            try:
                if self.weights is not None:
                    weights = self.weights
            except:
                weights = np.array([1/len(bagofclassif)]*len(bagofclassif))
        for classif,w in zip(bagofclassif,weights):
            labels_moy += classif.predict_proba(X)[0]*w
        return(labels_moy>selectivity)

    def experimental_loss_weights(self,weights):
        """
        Fonction de loss a optimiser pour les poids sur les modèles (sorte de boosting)
        Possible de changer la loss : a tester.
        expérimental/obsolète 
        """
        from sklearn.metrics import hamming_loss
        from sklearn.metrics import label_ranking_loss
        from Pipeline.Enjeux.utils import separate
        y_true,X_sub,y_pred = separate(self.docs,self.X,prediction=self.predict(self.X,weights))
        return label_ranking_loss(y_true,y_pred)

    def loss_select(self,select):
        """
        Fonction de loss a optimiser pour la sélectivité
        Possible de changer la loss : a tester.
        """
        from sklearn.metrics import hamming_loss
        from sklearn.metrics import label_ranking_loss
        from Pipeline.Enjeux.utils import separate
        y_true,X_sub,y_pred = separate(self.docs,self.X,prediction=self.predict(self.X,selectivity=select))
        return label_ranking_loss(y_true,y_pred)
    
    def optimize_selectivity(self,bnds =(0.2,0.8)):
        """
        A lancer directement si on veux modifier la sélectivité pour améliorer les résultats
        Initialement, s = 0.5
        Entrée :
        bnds = tuple (bound1,bound2). Les bornes théoriques sont 0 et 1 mais cela peut engendrer des résultats incertains
        Sortie :
        Scalaire correspond a la sélectivité
        """
        from scipy.optimize import minimize_scalar
        return minimize_scalar(self.loss_select, method = 'bounded',bounds = bnds)

    def format_results(self,results):
        """
        results = matrice numpy de True/False des résultats
        renvoie une liste de listes des enjeux pour chaque doc
        """
        enjeux = self.Thesaurus.Enjeux.values
        resultats_liste = []
        for doc in results:
            doc_res = []
            for k in range(len(doc)):
                if doc[k] ==True:
                    doc_res.append(enjeux[k])
            resultats_liste.append(doc_res)
        return(resultats_liste)

    def extract_topics(self,sections):
        """
        Fonction permettant d'extraire directement les topics a partir d'une array ou
        d'une liste de paragraphes en string
        Entrée :
        sections = ['lorem ipsum ...','...']
        Sortie :
        Matrice numpy True/False (n_docs,n_topics)
        """
        from tqdm import tqdm
        import pandas as pd
        from Pipeline.Enjeux.processing_encoding import processing
        sections = [processing(section) for section in tqdm(sections)]
        process = self.vectorizer.transform(sections)
        X = process.toarray().astype(int)
        return(self.format_results(self.predict(X)))


# %%
