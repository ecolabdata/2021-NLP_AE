# %%
import pandas as pd, numpy as np
import os
import pickle
import torch
import re
from unidecode import unidecode
from bs4 import BeautifulSoup
import time

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

class makesimilarity():
    def __init__(self, thesaurus,sentences = None, embedding = None, cosimilarite = None, verbose = 0, jobs = True):
        super(makesimilarity)
        """
        Entrées:
        -sentences : phrases pour le modèle W2V sous la forme d'une liste/vecteur contenant
         les phrases, qui sont elles même des listes/vecteur contenant les mots de la phrase
        -thesaurus : dictionnaire { enjeu : liste de mots }
        - embedding (optionnel) : fichier contenant le modèle W2V ssauvegardé
        - cosimilarité (optionnel) : vecteur de similarité de chaque mot avec chaque vecteur moyen des enjeux,
         shape (n_words,n_topics)

        Sorties :
        -vecteur de similarité de chaque mot avec chaque vecteur moyen des enjeux,
         shape (n_words,n_topics)
        """
        self.verb = verbose
        self.thesaurus = thesaurus
        if sentences:
            self.sentences = sentences
        if jobs:
            import os
            self.jobs = os.cpu_count()
        else:
            self.jobs = jobs
        from tqdm import tqdm
        import torch
        self.cosim = torch.nn.CosineSimilarity(dim=0)
        if torch.cuda.is_available():
            self.cosim = torch.nn.CosineSimilarity(dim=0)
        if embedding != None :
            self.W2V = embedding
            self.transform()
        if cosimilarite is not None :
            self.cosimilarite = cosimilarite
            self.M = cosimilarite.words.values.tolist()

    def fit(self,fenetre=15,minimum=1,d=300,e=10,save_data = True):
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
        if save_data:
            import pickle
            pickle.dump(model,open('Data\Workinprogress\\w2vmodel.pickle','wb'))

    def transform(self):
        """
        Construit les listes qui vont permettre de feed le calcul
        des similarités entre vecteurs
        self.embedding : liste de tenseurs correspondant aux embeddings des mots du W2V
        self.enjeux : dict { enjeu : liste de tenseurs des embeddings W2V des mots des enjeux}
        """

        #Ensemble de mots du modèle
        #Pour gensim 3
        self.M =list(self.W2V.wv.vocab.keys())
        #Pour gensim 4
        #self.M = list(self.W2V.wcv)
        ###### On définit E_T_tilde par l'intersection entre E_T_ et M
        #Ensemble de mots du Thesaurus contenu dans le vocab du modèle
        #Optimisation : tenir compte du fait que M est trié
        E_thesau={}
        for i in self.thesaurus.keys():
            E_thesau[i]=[m for m in self.thesaurus[i] if m in self.M and m != '']
        self.E_thesau = E_thesau
        #On va prendre les vecteurs de l'intersection Thesaurus et Mots:
        Vect_E_T_tilde={}
        if torch.cuda.is_available():
            for i in thesau.keys():
                Vect_E_T_tilde[i]=[torch.tensor(self.W2V[v]).cuda() for v in self.E_thesau[i]]
                #On récupère l'ensemble des vecteurs de chaque mot
                self.embedding = [torch.tensor(self.W2V[v]).cuda() for v in self.M]
                self.enjeux = Vect_E_T_tilde
        else:
            for i in thesau.keys():
                Vect_E_T_tilde[i]=[torch.tensor(self.W2V[v]) for v in self.E_thesau[i]]
            #On récupère l'ensemble des vecteurs de chaque mot
            self.embedding = [torch.tensor(self.W2V[v])for v in self.M]
            self.enjeux = Vect_E_T_tilde

    def fit_transform(self,fenetre=15,minimum=1,d=300,e=10):
        """
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
        
        if torch.cuda.is_available():
            cos = []
            for vect_e in vect_enjeu:
                cos.append(self.cosim(vect_e,vect))
            cos = torch.tensor(cos).cuda()
            cos = torch.mean(cos).cuda().cpu()
        else:
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
        if torch.cuda.is_available():
            cos = []
            for e in coll:
                cos.append(self.cos_moyen(vect,coll[e]))
        else:
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
        if torch.cuda.is_available():
            for v in tqdm(self.embedding):
                cossim.append(self.cos_moyen_enjeux(v))
        else:
            from Pipeline.Enjeux.utils import tqdm_joblib
            from joblib import Parallel, delayed
            with tqdm_joblib(tqdm(desc="Calcul de cossimilarité", total=len(self.embedding))) as progress_bar:
                cossim = Parallel(n_jobs=self.jobs)(delayed(self.cos_moyen_enjeux)(v) for v in self.embedding)
        cossim = pd.DataFrame(np.matrix(cossim),columns = [e for e in self.thesaurus])
        cossim['words'] = inst.M
        if save_data:
            pickle.dump(cossim,open("cosimilarite.pickle",'wb'))
        self.cosimilarite = cossim
    
    
    def cos_moyen_batch(self,save_data = True, batchsize = 1000,start = 0):
        """
        Calcule la cosimilarité moyenne de tous les vecteurs du vocab
        avec tous les mots de chaque enjeu
        Renvoie une liste contenant la cosim avec chaque enjeu (taille n_enjeux)
        """
        from tqdm import tqdm
        import numpy as np
        import pickle
        batch = self.embedding[start:start+batchsize]
        cossim = []
        if torch.cuda.is_available():
            for v in tqdm(batch):
                cossim.append(self.cos_moyen_enjeux(v))
        else:
            from Pipeline.Enjeux.utils import tqdm_joblib
            from joblib import Parallel, delayed
            with tqdm_joblib(tqdm(desc="My calculation", total=len(batch))) as progress_bar:
                cossim = Parallel(n_jobs=self.jobs)(delayed(self.cos_moyen_enjeux)(v) for v in batch)
        cossim = pd.DataFrame(np.matrix(cossim),columns = [e for e in self.thesaurus])
        if save_data:
            pickle.dump(cossim,open("Data\Workinprogress\\cosimilarite"+str(start)+"-"+str(start+batchsize)+".pickle",'wb'))
        self.cosimilarite = cossim

    def top_words_topic(self,n):
        topwords = {}
        cossim = self.cosimilarite
        for enjeu in self.thesaurus:
            subset = cossim[~cossim['words'].isin(self.thesaurus[enjeu])]
            nlarge = subset[enjeu].nlargest(n,keep = 'all')
            idxs = [idx for idx in nlarge.index]
            words = [self.M[idx] for idx in idxs]
            topwords[enjeu] = words
        return(topwords)

#Executé seulement si on travaille directement dans le fichier. 
#Ce code sert a générer le fichier de cosimilarité (300Mo) qui peut être utilisé ailleurs pour enrichir le thésaurus
if __name__ == "main":
    #On vérifie s'il y'a besoin de réentrainer le W2V
    if not os._exists('Data\Workinprogress\\w2vmodel.pickle'):
        #Préparation texte et thésaurus
        base = pd.read_csv(open("Data\Etude_html_csv\\base_html_06_04.csv",'rb'))
        thesaurus="Data\Thesaurus_csv\Thesaurus1.pickle"
        thesaurus=pickle.load(open(thesaurus,'rb')) #dictionnaire de W2V

        base=base[base.texte.isna()==False]
        start=time.time()
        base['clean']=[unidecode(re.sub(r'[^A-Za-z.]',' ',
                        BeautifulSoup(
                            base.texte.values[i],"html.parser").get_text()).lower())
                            for i in range(len(base))]
        end=time.time()
        print("Durée :",round((end-start)/60,2),"minutes")
        base['clean']=[' '.join([i for i in base.clean.values[k].split() if len(i)>3]) for k in range(len(base))]

        sentences = []
        for c in base.clean.values:
            doc = str(c).split('.')
            for sent in doc:
                if sent != '':
                    if sent[0] == ' ':
                        sentences.append(sent[1:].split(' '))
                    else:
                        sentences.append(sent.split(' ')) 

        thesau={}
        for e,i in zip(thesaurus['Enjeu environnemental'],thesaurus.Dictionnaire):
            thesau[e]=[represent_word(k) for k in i ]+[represent_word(e)]

        #Initialisation
        inst = makesimilarity(sentences,thesau)
        #Entrainement et stockage du W2V
        inst.fit(e=100)
        #Calcul des cosimilarités
        inst.cos_moyen_all()
    else:
        #Initialisation a partir des données sauvegardées
        inst = makesimilarity(sentences,thesau, 
        cosimilarite=pickle.load(open('Data\Workinprogress\\cosimilarite.pickle','rb')),
        embedding=pickle.load(open('Data\Workinprogress\\w2vmodel.pickle','rb')))
