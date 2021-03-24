#%%
import pandas as pd
import numpy as np
import sklearn
import pickle

chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"

df=pd.read_csv(chemin+'etudes_dataset_cleaned_prepared.csv',sep=";")
X_train=df.docs[:25]
X_test=df.docs[25:]
# %%
########################################################################################################
#############            LSA            ###########################################################################################
##################################################################################################################


from sklearn.decomposition import TruncatedSVD
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#%%
import re
theme=list(np.unique(re.sub("[\(\[].*?[\)\]]", "",
    re.sub(","," ",
        re.sub(";"," ",' '.join(np.unique(df.theme.values))))).split(' ')))
theme.remove('ET'),theme.remove('')

database={}
for i in range(len(theme)):
    database[theme[i]]=df[[True if theme[i] in df.theme[z] else False for z in range(len(df))]]

# %%
maxf=100000
vect = TfidfVectorizer(analyzer = "word", ngram_range=(1, 2), 
tokenizer = None, preprocessor = None, max_features = maxf)

ndim=100
niter=1000
svd = TruncatedSVD(n_components=ndim, n_iter=niter) 
svdm={}
train={}
train_lsa={}
name={}
vect_theme={}
for t in theme:
    train[t]=vect.fit_transform(database[t].docs)
    vect_theme[t]=vect
    name[t]=vect.get_feature_names()

#n_components est le nombre de dimensions (thèmes) auquel on veut arriver
# n_iter = Number of iterations for randomized SVD solver.
    svdm[t]=svd.fit(train[t])
    train_lsa[t]=svdm[t].transform(train[t])
    print(train_lsa[t].shape)
# X_test_lsa=svdm.transform(test_features)
    # print(svd.explained_variance_ratio_.sum()*100)

try:
    os.mkdir(chemin+"LSA")
except:
    print('Le dossier existe déjà.')

pickle.dump(svdm,open(chemin+'LSA/LSA_theme_model.pickle',"wb"))
pickle.dump(name,open(chemin+'LSA/LSA_theme_name_features.pickle',"wb"))
pickle.dump(vect_theme,open(chemin+'LSA/LSA_theme_vect.pickle',"wb"))
pickle.dump(train,open(chemin+'LSA/LSA_theme_docs.pickle',"wb"))
pickle.dump(train_lsa,open(chemin+'LSA/LSA_theme_docs_svd.pickle',"wb"))

# print(len(svdm.components_))
# svdm.explained_variance_ratio_
# svdm.singular_values_
# %%
svdm=pickle.load(open(chemin+'LSA/LSA_theme_model.pickle',"rb"))
name=pickle.load(open(chemin+'LSA/LSA_theme_name_features.pickle',"rb"))
vect_theme=pickle.load(open(chemin+'LSA/LSA_theme_vect.pickle',"rb"))
train=pickle.load(open(chemin+'LSA/LSA_theme_docs.pickle',"rb"))
train_lsa=pickle.load(open(chemin+'LSA/LSA_theme_docs_svd.pickle',"rb"))
for t in theme:
    for i, comp in enumerate(svdm[t].components_):
        terms_comp = zip(name[t], comp)
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
        print("Topic "+str(i)+": ")
        print([sorted_terms[i][0] for i in range(len(sorted_terms))])




# %%
########################################################################################################
#############     éléments les plus informatifs du tf-idf          ###########################################################################################
##################################################################################################################


matrice=(train_features).toarray()
len(matrice),len(matrice[0])
# %%
def Nmaxelements(list2,N): 
    final_list = []
    list1=list(list2.copy())
    for i in range(0, N):  
        max1 = 0
          
        for j in range(len(list1)):      
            if list1[j] > max1: 
                max1 = list1[j]; 
                  
        list1.remove(max1); 
        final_list.append(max1) 
          
    return final_list
def get_max_names(train_features,vect,N): 
    matrice=(train_features).toarray()
    maxnamesN=[]
    number_of_docs=len(matrice)
    for i in range(number_of_docs):
        maxn=Nmaxelements(matrice[i],N)
        maxnames=[]
        for k in range(N):
            maxnames.append(vect.get_feature_names()[list(matrice[i]).index(maxn[k])])
        maxnamesN.append(maxnames)
    print(number_of_docs,' listes des', N,' plus grands éléments.')
    return maxnamesN
# %%
mots_theme={}
for t in theme:
    mots_themes[t]=get_max_names(train[t],vect_theme[t],25)

# %%
########################################################################################################
#############            LDA    Gensim        ###########################################################################################
##################################################################################################################

import nltk
import spacy
nlp = spacy.load('fr_core_news_sm',disable=["parser","ner"]) #on charge le modèle français
import gensim
import sys
import re, numpy as np, pandas as pd
from pprint import pprint

# Gensim
from gensim.test.utils import datapath
import logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt
#%%
# NLTK Stop words
from nltk.corpus import stopwords
nltk.set_proxy('http://cache.ritac.i2:32000')
nltk.download('punkt')
nltk.data.load('tokenizers/punkt/french.pickle')
nltk.download('stopwords')
stop_word=nltk.corpus.stopwords.words('french')
# %%
article=X_train[0:3]
mots=[simple_preprocess(i,deacc=True) for i in article] 
#This lowercases, tokenizes, de-accents (optional). – the output are final tokens = unicode strings, that won’t be processed any further.
#Ici le texte est déjà nettoyé donc la fonction ne change pas grand chose
print([len(mots[i]) for i in range(3)])
#%%
# Build the bigram and trigram models
#Automatically detect common phrases – aka multi-word expressions, word n-gram collocations – from a stream of sentences.
bigram = gensim.models.Phrases(mots, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[mots], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
#%%
mots2=[[word for word in z if word not in stop_word] for z in mots] #On vire les stop words
print([len(mots2[i]) for i in range(3)])
#%%
mots2=[bigram_mod[z] for z in mots2] #Donc en fait, ça va chercher les collocations, les bigrammes via la représentation sous forme de vecteur
#élimine également un certain nombre de mots, 343 en l'occurrence, je ne sais pas pourquoi
print([len(mots2[i]) for i in range(3)])
print("Nouveaux mots",[len([i for i in mots2[z] if i not in mots[z]]) for z in range(3)])
#%%
mots3=[trigram_mod[bigram_mod[z]] for z in mots]
print([len(mots2[i]) for i in range(3)])
print("Nouveaux mots",[len([i for i in mots3[z] if i not in mots2[z]]) for z in range(3)])
# %%
#Dictionnaire
id2word = corpora.Dictionary(mots2)
# %%
# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in mots2]
corpus
# %%
#Build LDA model
k=25
lda_model = LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=k, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)
# %%
lda_model[corpus]
# %%
lda_model.save(datapath(chemin+"lda_1"))
lda_model=LdaModel.load(datapath(chemin+"lda_1"))
# %%
for idx,topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic))
    print("\n")
# %%
#########################################################################################################################
#############      Généralisation      LDA    Gensim        ###########################################################################################
##################################################################################################################

theme=list(np.unique(re.sub("[\(\[].*?[\)\]]", "",
    re.sub(","," ",
        re.sub(";"," ",' '.join(np.unique(df.theme.values))))).split(' ')))
theme.remove('ET'),theme.remove('')
theme
# %%
database={}
for i in range(len(theme)):
    database[theme[i]]=df[[True if theme[i] in df.theme[z] else False for z in range(len(df))]]
# %%
mots={}
mots_themes={}

bigram={}
trigram={}
bigram_mod={}
trigram_mod={}

id2word={}
corpus={}
k=25

LDA_theme={}

try:
    os.mkdir(chemin+'LDA_model_theme')
except:
    print('Le dossier existe déjà.')

for t in theme:
    mots[t]=[simple_preprocess(i,deacc=True) for i in database[t].docs]
    print(len(mots[t]))
    bigram[t] = gensim.models.Phrases(mots[t], min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram[t] = gensim.models.Phrases(bigram[t][mots[t]], threshold=100)

    bigram_mod[t] = gensim.models.phrases.Phraser(bigram[t])
    trigram_mod[t] = gensim.models.phrases.Phraser(trigram[t])

    mots_themes[t]=[[word for word in z if word not in stop_word] for z in mots[t]]
    mots_themes[t]=[bigram_mod[t][z] for z in mots_themes[t]]
    mots_themes[t]=[trigram_mod[t][bigram_mod[t][z]] for z in mots_themes[t]]
    print(len(mots_themes[t]),len(mots_themes[t][0]))

    id2word[t] = corpora.Dictionary(mots_themes[t])
    corpus[t] = [id2word[t].doc2bow(text) for text in mots_themes[t]]

    LDA_theme[t] = LdaModel(corpus=corpus[t],
                                           id2word=id2word[t],
                                           num_topics=k, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=1,
                                           passes=20,
                                           alpha='auto',
                                           iterations=100,
                                           per_word_topics=True)
    # lda_model.save(datapath(chemin+"lda_g"))
    # lda_model=LdaModel.load(datapath(chemin+"lda_g"))                                      
pickle.dump(LDA_theme,open(chemin+'LDA_model_theme/LDA_model.pickle','wb'))
#%%
LDA_theme=pickle.load(open(chemin+'LDA_model_theme/LDA_model.pickle','rb'))
# %%    
print(theme[0])

for idx,topic in LDA_theme[theme[0]].print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic))
    print("\n")

# %%
LDA_theme[theme[0]].show_topic(-1,100)
# %%
#########################################################################################################################
#############      Méthodes à tester       ###########################################################################################
## 1. Kmeans
## 2. LDA2vec

# %%
#########################################################################################################################
#############      Gensim explication et exemple       ###########################################################################################
##################################################################################################################

from gensim.models import Phrases

from gensim.models.phrases import Phraser

documents = ["the mayor of new york was there", "machine learning can be useful sometimes","new york mayor was present"]


sentence_stream = [doc.split(" ") for doc in documents]
print(sentence_stream)
#%%
bigram = Phrases(sentence_stream, min_count=1, threshold=2, delimiter=b' ')

bigram_phraser = Phraser(bigram)


print(bigram_phraser)
#%%
for sent in sentence_stream:
    tokens_ = bigram_phraser[sent]

    print(tokens_)
# %%
