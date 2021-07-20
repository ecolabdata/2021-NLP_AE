#%%
class CorExBoosted():
    def __init__(self,docs,thesaurus):
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

    def preprocess(self,colname):
        from Pipeline.Enjeux.processing_encoding import processing
        from tqdm import tqdm
        tqdm.pandas(desc= 'preprocessing...')
        self.docs['text_processed'] =self.docs[colname].progress_apply(processing)
        return self

    def encode(self, min_df = 13,max_df = 0.95, ngram_range=(1,3), stop_words = None):
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

        thesau_list = []
        for enj in self.Thesaurus.Dictionnaire.values.tolist():
            ll = [w for w in enj if w in self.vocab]
            thesau_list.append(ll)
        self.thesau_list = thesau_list

        self.dico_thesau = {e:d for e,d in zip(self.enjeux_list,self.thesau_list)}
        return self
    
    def diagnostic(self,max_df=True,min_df=True,N=15):
        from Pipeline.Enjeux.processing_encoding import analysis_min_df,analysis_max_df
        resultats = []
        if max_df:
            couvertures_max_df = analysis_max_df(self.docs.text_processed.values,Thesaurus,N=N) #Retourne la couverture pour chaque enjeu en faisant varier le paramètre max_df de 1 a 0,85
            resultats.append(couvertures_max_df)
        if min_df:
            couvertures_min_df = analysis_min_df(self.docs.text_processed.values,Thesaurus,N=N) #Retourne la couverture pour chaque enjeu en faisant varier le paramètre min_df de 0 a 15
            resultats.append(couvertures_min_df)
        return(resultats)

    def stratify(self,df_corrige):
        import numpy as np
        import pandas as pd
        from Pipeline.Enjeux.utils import separate
        #On récupère les y labellisés (vrais), et X sur les mêmes indices
        y_true,X_sub = separate(self.docs,self.X)
        y_true_df = pd.DataFrame(y_true,columns = self.enjeux_list)
        X_df = pd.DataFrame(X_sub,columns=self.vocab)
        from Pipeline.Enjeux.multilabel_balancing import get_minority_instance
        #Stratification a partir des éléments labellisés
        #On récupère un sous ensemble de ces X 
        # avec des proportions équilibrées pour chaque enjeu
        self.X_sub, self.y_sub = get_minority_instance(X_df, y_true_df)
    
    def augment(self,n_sets,n_samples = 50):
        from Pipeline.Enjeux.multilabel_balancing import MLSMOTE2
        self.datasets = []
        for j in range(n_sets):
            self.datasets.append(MLSMOTE2(self.X_sub,self.y_sub,n_samples))
        return(self)

    def fit_one(self,model,X,strength):
        model.fit(X,words=self.vocab, anchors=self.thesau_list, anchor_strength=strength)
        return(model)
    
    def fit(self,n_classif = 100,stratify = True,augment = True, n_samples = 50,strength = 1):
        from joblib import Parallel,delayed
        from corextopic import corextopic as ct
        import os
        import numpy as np
        storage = []
        for j in range(n_classif):
            storage.append(ct.Corex(n_hidden=len(self.enjeux_list)))
        if stratify and augment:
            self.stratify()
            self.augment(n_sets = n_classif,n_samples = n_samples)
            storage = [(model,data) for model,data in zip(storage,self.datasets)]
            storage_fitted = Parallel(n_jobs=os.cpu_count(),verbose=0)(delayed(self.fit_one)(item[0],np.matrix(item[1][0]),strength) for item in storage)
            self.models = storage_fitted
            return self
        if stratify and not augment:
            self.stratify()
            data = (self.X_sub,self.y_sub)
            storage_fitted = Parallel(n_jobs=os.cpu_count(),verbose=0)(delayed(self.fit_one)(item,np.matrix(data[0]),strength) for item in storage)
            self.models = storage_fitted
            return self
        else:
            storage_fitted = Parallel(n_jobs=os.cpu_count(),verbose=0)(delayed(self.fit_one)(model,self.X,strength) for model in storage)
            self.models = storage_fitted
            return self

    def predict(self,X,weights = None, selectivity = 0.5):
        import numpy as np
        bagofclassif = self.models
        labels_moy = np.zeros(bagofclassif[0].predict_proba(X)[0].shape)
        if weights is None:
            weights = np.array([1/len(bagofclassif)]*len(bagofclassif))
        for classif,w in zip(bagofclassif,weights):
            labels_moy += classif.predict_proba(X)[0]*w
        return(labels_moy>selectivity)

    def loss_weights(self,weights):
        from sklearn.metrics import hamming_loss
        from sklearn.metrics import label_ranking_loss
        from Pipeline.Enjeux.utils import separate
        y_true,X_sub,y_pred = separate(self.docs,self.X,prediction=self.predict(self.X,weights))
        return label_ranking_loss(y_true,y_pred)

    def loss_select(self,select):
        from sklearn.metrics import hamming_loss
        from sklearn.metrics import label_ranking_loss
        from Pipeline.Enjeux.utils import separate
        y_true,X_sub,y_pred = separate(self.docs,self.X,prediction=self.predict(self.X,selectivity=select))
        return label_ranking_loss(y_true,y_pred)
    
    def optimize_selectivity(self,bnds =(0.2,0.8)):
        from scipy.optimize import minimize_scalar
        return minimize_scalar(self.loss_select, method = 'bounded',bounds = bnds)
    
    def optimize_weights(self, method = None,tol = None):
        import numpy as np
        from scipy.optimize import minimize
        x0 = np.random.rand(len(self.models))
        x0 = x0/x0.sum()
        bnds =[]
        for k in range(len(x0)):
            bnds.append((0,None))
        bnds = tuple(bnds)
        self.x0 = x0
        if method is None:
            return minimize(self.loss_weights,x0,tol = tol,bounds = bnds,constraints=({'type':'eq','fun':lambda x:x.sum()-1}))
        else:
            return minimize(self.loss_weights,x0,tol = tol, method = method,bounds = bnds,constraints=({'type':'eq','fun':lambda x:x.sum()-1}))