# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd, numpy as np
import os
import pickle


# %%
base = pd.read_csv(open("Data\Etude_html_csv\\base_html_06_04.csv",'rb'))
base


# %%
thesaurus="Data\Thesaurus_csv\Thesaurus1.pickle"

thesaurus=pickle.load(open(thesaurus,'rb')) #dictionnaire de W2V

thesaurus


# %%
base=base[base.texte.isna()==False]
import re
from unidecode import unidecode
from bs4 import BeautifulSoup
import time
start=time.time()
base['clean']=[unidecode(re.sub(r'[^A-Za-z]',' ',
                BeautifulSoup(
                    base.texte.values[i],"html.parser").get_text()).lower())
                     for i in range(len(base))]
end=time.time()
print("Durée :",round((end-start)/60,2),"minutes")
base['clean']=[' '.join([i for i in base.clean.values[k].split() if len(i)>3]) for k in range(len(base))]


# %%
base

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
thesaurus


# %%
### A ne pas faire tourner quand on aura le vrai thesau (puisque les données ne proviendront pas d'un W2V)

thesau={}
for i in thesaurus["Enjeu environnemental"]:
    for z in thesaurus[thesaurus["Enjeu environnemental"]==i].Dictionnaire:
        thesau[i]=[represent_word(k)  for k in z ]
thesau


# %%
thesaurus #dictionnaire


# %%
###### On définit E par l'ensemble des mots des enjeux, dont le proxy est le Thésaurus final fourni par la DREAL
###### Pour le moments nous n'avons ni thesaurus ni enjeux, en revanche nous avons LégiFrance
###### On définit E_T par le Thesaurus final de la Dreal et donc
###### On définit un ensemble E_T_ (prononcé E T prime) qui contient les mots du Thesaurus LégiFrance
import functools
import operator
#E_T_=functools.reduce(operator.iconcat, list(thesau.values()), [])
#E_T_=[unidecode(i.lower()) for i in E_T_]

###### On crée notre base de phrases pour le Word2Vec
import gensim
sentences =np.array([str(c).split() for c in base.clean.values])

###### On crée et entraine le modèle d'embedding
fenetre=15
minimum=1
d=300
e=10
satrt=time.time()
W2V=gensim.models.Word2Vec(size=d,window=fenetre,min_count=minimum)
W2V.build_vocab(sentences)
W2V.train(sentences,total_examples=W2V.corpus_count,epochs=e)
end=time.time()
print("Durée entraînement du Word2Vec :",round((end-start)/60,2),"minutes.")


# %%
from sklearn.feature_extraction.text import TfidfVectorizer
#maxf=10000
vect = TfidfVectorizer(analyzer = "word", ngram_range=(1, 3), tokenizer = None, preprocessor = None,max_df=0.9,min_df=0.05)
train_features = vect.fit_transform(base.clean)


# %%
base_tfidf=pd.DataFrame(train_features.toarray(),columns=vect.get_feature_names())


# %%
vect.get_feature_names()

# %% [markdown]
# On crée nos ensembles de mots

# %%
#Ensemble de mots du modèle
M=list(W2V.wv.vocab.keys())
len(M)


# %%
M_=[m for m in vect.get_feature_names() if m in M]
print(len(M_))


# %%
###### On définit E_T_tilde par l'intersection entre E_T_ et M
#Ensemble de mots du Thesaurus contenu dans le vocab du modèle
E_thesau={}
for i in thesau.keys():
    E_thesau[i]=[m for m in thesau[i] if m in M]

#E_T_tilde=[i for i in E_T_ if i in M]
#print(len(E_T_tilde))

##### On définit T comme l'ensemble des thèmes, donc T_tilde les :
# thèmes qui sont dans l'ensemble de mots
# On ne s'intéresse finalement plus au thème
#T_tilde=[i for i in [unidecode(z.lower()) for z in theme] if i in M]
#len(T_tilde)

#On va prendre les vecteurs de l'interseciton Thesaurus et Mots:
Vect_E_T_tilde={}
for i in thesau.keys():
    Vect_E_T_tilde[i]=[W2V[v] for v in E_thesau[i]]
#print(len(Vect_E_T_tilde))

#On récupère l'ensemble des vecteurs de chaque mot
Vect_M=[W2V[v] for v in M]
Vect_M_=[W2V[v] for v in M_]
print(len(Vect_M),len(Vect_M_))

#On récup les vecteurs des thèmes
#Vect_T_tilde=[W2V[v] for v in T_tilde]
#print(Vect_T_tilde)


# %%
#Vect_E_T_tilde
#Vect_M_


# %%
def euclid(x):
    import numpy as np
    d=np.sqrt(sum([i**2 for i in x]))
    return d

def cos_sim(x,y):
    a=x@y
    l=euclid(x)*euclid(y)
    sim=a/l
    return sim

# %% [markdown]
# ### A voir pour changer la fonction de cosim, là tu utilises la mienne, mais elle est vraiment pas optimiser
# Tu peux utiliser ça plutôt, qui vient directement de torch :

# %%
import torch
cosim=torch.nn.CosineSimilarity(dim=0)
x=torch.rand(10)
y=torch.rand(10)
print(x,y)
cosim(x,y)


# %%
from joblib import Parallel,delayed
import time
duree=[]
absi=[]
import psutil
cpu=psutil.cpu_count()
for i in range(1,cpu):
    absi.append(i)
    start=time.time()
    Parallel(n_jobs=i,verbose=0)(delayed(cos_sim)(Vect_M[0],v) for v in Vect_E_T_tilde['construction'])
    end=time.time()
    duree.append(end-start)

import matplotlib.pyplot as plt
f,a=plt.subplots(1,figsize=(12,6))
a.plot(absi,duree)
a.set(xlabel="Nombre de job",ylabel='Durée',
      title='Durée en fonction du degré de parallélisation')


# %%
jobs=[duree.index(i) for i in sorted(duree)[0:1]][0]+1


# %%



# %%
from joblib import Parallel,delayed
import time
from tqdm import tqdm
import psutil
#start=time.time()
#cos_moyen=[np.mean(
#    Parallel(n_jobs=jobs,verbose=0)(delayed(cos_sim)(m,v) for v in Vect_E_T_tilde))
#  for m in tqdm(Vect_M)]
#end=time.time()
#print("La parallélisation a durée :",round((end-start)/60,3),"minutes")


###########Par thème

def cos_moyen(Vect_E_T_tilde,Vect_M,jobs,verb=0):
    from joblib import Parallel,delayed
    import numpy as np
    from tqdm import tqdm
    cos=[np.mean(
        Parallel(n_jobs=jobs,verbose=verb)(delayed(cos_sim)(m,v) for v in Vect_E_T_tilde))
      for m in tqdm(Vect_M)]
    return cos

cpu=psutil.cpu_count()
print(cpu,"CPU vont être utilisés")

from functools import partial

cos_moyen_=partial(cos_moyen,Vect_M=Vect_M,jobs=cpu,verb=0)
 #cos_thesau_={}

start=time.time()
cos_thesau_=Parallel(n_jobs=cpu)(delayed(cos_moyen_)(Vect_E_T_tilde[i]) for i in tqdm(Vect_E_T_tilde.keys()))
#[np.mean(Parallel(n_jobs=jobs,verbose=0)(delayed(cos_sim)(m,v) for v in Vect_E_T_tilde[i])) for m in tqdm(Vect_M)]
end=time.time()
print("La parallélisation a durée :",round((end-start)/60,3),"minutes")

# pd.DataFrame(Vect_M,columns=M)


# %%
import pickle
path = "Data\Bagging_model"

pickle.dump(cos_thesau_,open(path+"\cos_thesau.pickle",'wb'))


# %%
cos_thesau_


# %%
import pickle
path = "Data\Bagging_model"
cos_thesau=pickle.load(open(path+"\cos_thesau.pickle",'rb'))


# %%
len(cos_thesau),len(cos_thesau[0])


# %%
M_


# %%
N=10
mots_thesau={}
for cos_m in cos_thesau_:
    i=pd.Series(cos_m).nlargest(N)
    mots_thesau[list(thesau.keys())[cos_thesau_.index(cos_m)]]=[M[k] for k in i.index.values.tolist()]

mots_thesau


# %%
da=dataiku.Folder('XXZ13n5V')
pickle.dump(mots_thesau,open(da.get_path()+"/mots_thesau_W2V.pickle",'wb'))


# %%
mots_thesau_2={}
for i in thesau.keys():
    ouais=[]
    for k in mots_thesau[i]:
        if k not in thesau[i]:
            ouais.append(k)
    mots_thesau_2[i]=ouais
mots_thesau_2


# %%
for i in mots_thesau_2.keys():
    print("################### ",i)
    print(mots_thesau_2[i],"\n")


# %%
cosmax=max(cos_moyen)
index_max=cos_moyen.index(cosmax)


# %%
N=10
i = pd.Series(cos_thesau[0]).nlargest(N)
i.index.values.tolist()


# %%
pd.DataFrame(cos_moyen,M)


# %%
len(vocab_theme)


# %%
thesaurus_enjeu_df = pd.DataFrame(cos_moyen) # Compute a Pandas dataframe to write into Thesaurus_enjeu


# Write recipe outputs
thesaurus_enjeu = dataiku.Dataset("Thesaurus_enjeu")
thesaurus_enjeu.write_with_schema(thesaurus_enjeu_df)


