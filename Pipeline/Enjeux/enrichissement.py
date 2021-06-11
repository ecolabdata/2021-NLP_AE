# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd, numpy as np
import os
import pickle
import torch


base = pd.read_csv(open("Data\Etude_html_csv\\base_html_06_04.csv",'rb'))

thesaurus="Data\Thesaurus_csv\Thesaurus1.pickle"
thesaurus=pickle.load(open(thesaurus,'rb')) #dictionnaire de W2V

# %%
base=base[base.texte.isna()==False]
import re
from unidecode import unidecode
from bs4 import BeautifulSoup
import time
start=time.time()
base['clean']=[unidecode(re.sub(r'[^A-Za-z.]',' ',
                BeautifulSoup(
                    base.texte.values[i],"html.parser").get_text()).lower())
                     for i in range(len(base))]
end=time.time()
print("Durée :",round((end-start)/60,2),"minutes")
base['clean']=[' '.join([i for i in base.clean.values[k].split() if len(i)>3]) for k in range(len(base))]


# %% [markdown]
# ##### 28 Avril 2021
# On modifie ce code pour prendre en compte la structure du dictionnaire d'enjeu

# %%
def represent_word(word):
    import re
    from unidecode import unidecode
    # True-case, i.e. try to normalize sentence-initial capitals.
    # Only do this if the lower-cased form is more probable.
    text = word.replace("lire page","") #replce "lire page" par "" (en gros delete "lire page")
    text = unidecode(text) #take a unicode object and returns a string
    text=text.lower()
    text = re.sub(r'[^A-Za-z]',' ',str(text))
    #means any character that IS NOT a-z OR A-Z
    text = ' '.join([i for i in text.split() if len(i)>2])
    return text


# %%
### A ne pas faire tourner quand on aura le vrai thesau (puisque les données ne proviendront pas d'un W2V)

thesau={}
for e,i in zip(thesaurus['Enjeu environnemental'],thesaurus.Dictionnaire):
    thesau[e]=[represent_word(k) for k in i ]+[represent_word(e)]
thesau

#%%
import functools
import operator
###### On crée notre base de phrases pour le Word2Vec
import gensim

sentences = []
for c in base.clean.values:
    doc = str(c).split('.')
    for sent in doc:
        if sent != '':
            if sent[0] == ' ':
                sentences.append(sent[1:].split(' '))
            else:
                sentences.append(sent.split(' ')) 


#%%
import pickle


compute = False
#Ok je force un peu sur le nombre d'epoch
if compute:
    inst = makesimilarity(sentences,thesau)
    inst.fit(e=100)
    pickle.dump(inst.W2V,open('Data\Workinprogress\\w2vmodel.pickle','wb'))
    inst.transform()
else:
    inst = makesimilarity(sentences,thesau,embedding=pickle.load(open('Data\Workinprogress\\w2vmodel.pickle','rb')))
    inst.transform()
#%%


#%%
#Attention c'est MAXI LONG, a recharger impérativement : 3800 heures sinon
#inst.cos_moyen_all()
#%%
#top = inst.top_words_topic(10)
# %%

class makesimilarity():
    def __init__(self, sentences, thesaurus, embedding = None, cosimilarite = None, verbose = 0, jobs = True):
        super(makesimilarity)
        """
        Entrées:
        -sentences : phrases pour le modèle W2V sous la forme d'une liste/vecteur contenant
         les phrases, qui sont elles même des listes/vecteur contenant les mots de la phrase
        -thesaurus : dictionnaire { enjeu : liste de mots }
        Sorties :
        -vecteur de similarité de chaque mot avec chaque vecteur moyen des enjeux,
         shape (n_words,n_topics)
        """
        self.verb = verbose
        self.sentences = sentences
        self.thesaurus = thesaurus
        if jobs:
            import os
            self.jobs = os.cpu_count()
        else:
            self.jobs = jobs
        from tqdm import tqdm
        import torch
        self.cosim = torch.nn.CosineSimilarity(dim=0)
        if embedding != None :
            self.W2V = embedding
        if cosimilarite != None :
            self.cosimilarite = cosimilarite

    def fit(self,fenetre=15,minimum=1,d=300,e=10):
        """
        Entraine le modèle W2V
        """
        import gensim
        corpus_sentences = self.sentences
        start=time.time()
        model=gensim.models.Word2Vec(size=d,window=fenetre,min_count=minimum)
        model.build_vocab(corpus_sentences)
        model.train(corpus_sentences,total_examples=model.corpus_count,epochs=e)
        end=time.time()
        print("Durée entraînement du Word2Vec :",round((end-start)/60,2),"minutes.")
        self.W2V = model

    def transform(self):
        """
        Construit les listes qui vont permettre de feed le calcul
        des similarités entre vecteurs
        self.embedding : liste de tenseurs correspondant aux embeddings des mots du W2V
        self.enjeux : dict { enjeu : liste de tenseurs des embeddings W2V des mots des enjeux}
        """

        #Ensemble de mots du modèle
        self.M =list(self.W2V.wv.vocab.keys())
        ###### On définit E_T_tilde par l'intersection entre E_T_ et M
        #Ensemble de mots du Thesaurus contenu dans le vocab du modèle
        #Optimisation : tenir compte du fait que M est trié
        E_thesau={}
        for i in self.thesaurus.keys():
            E_thesau[i]=[m for m in self.thesaurus[i] if m in self.M and m != '']
        self.E_thesau = E_thesau
        #On va prendre les vecteurs de l'intersection Thesaurus et Mots:
        Vect_E_T_tilde={}
        for i in thesau.keys():
            Vect_E_T_tilde[i]=[torch.tensor(self.W2V[v]) for v in self.E_thesau[i]]
        #On récupère l'ensemble des vecteurs de chaque mot
        self.embedding = [torch.tensor(self.W2V[v])for v in self.M]
        self.enjeux = Vect_E_T_tilde

    def fit_transform(self,fenetre=15,minimum=1,d=300,e=10):
        """
        tu fit puis tu transform
        tu fit puis tu transform
        tu fit puis tu transform
        """
        self.fit(fenetre,minimum,d,e)
        self.transform()

    def cos_moyen(self,vect,vect_enjeu):
        """
        Calcule la cosimilarité d'un vecteur
        avec tous les mots d'un enjeu, puis moyennise
        Renvoie un tensor scalaire
        """
        from joblib import Parallel,delayed
        cos=np.mean(
            Parallel(n_jobs=self.jobs,verbose=self.verb)(delayed(self.cosim)(vect_e,vect) for vect_e in vect_enjeu))
        return cos

    def cos_moyen_enjeux(self,vect):
        """
        Calcule la cosimilarité moyenne d'un vecteur
        avec tous les mots de chaque enjeu
        Renvoie une liste contenant la cosim avec chaque enjeu (taille n_enjeux)
        """
        from joblib import Parallel,delayed

        coll = self.enjeux
        cos=Parallel(n_jobs=self.jobs,verbose=self.verb)(delayed(self.cos_moyen)(vect,coll[e]) for e in coll)
        return cos

    def cos_moyen_all(self,save_data = True):
        """
        Calcule la cosimilarité moyenne de tous les vecteurs du vocab
        avec tous les mots de chaque enjeu
        Renvoie une liste contenant la cosim avec chaque enjeu (taille n_enjeux)
        """
        from tqdm import tqdm
        import numpy as np
        import pickle
        cossim = []
        with tqdm_joblib(tqdm(desc="My calculation", total=len(self.embedding))) as progress_bar:
            cossim = Parallel(n_jobs=self.jobs)(delayed(self.cos_moyen_enjeux)(v) for v in self.embedding)
        cossim = pd.DataFrame(np.matrix(cossim),columns = [e for e in self.thesaurus])
        if save_data:
            pickle.dump(cossim,open("Data\Workinprogress\\cosimilarite.pickle",'wb'))
        self.cosimilarite = cossim

    def top_words_topic(self,n):
        topwords = {}
        cossim = self.cosimilarite
        for enjeu in self.thesaurus:
            nlarge = cossim[enjeu].nlargest(n,keep = 'all')
            idxs = [idx for idx in nlarge.index]
            words = [self.M[idx] for idx in idxs]
            topwords[enjeu] = words
        return(topwords)



# %%
import contextlib
import joblib
from tqdm import tqdm    
from joblib import Parallel, delayed

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close() 

# %%
