#%%

####!!!! FONCTIONNE UNIQUEMENT SOUS PYTHON 3.7 (en tout cas ne marche pas en 3.9)

import pandas as pd
import numpy as np
import re
import os
#import guidedlda
import nltk
import pickle


# NLTK Stop words
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.data.load('tokenizers/punkt/french.pickle')
nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import CountVectorizer
import spacy
nlp = spacy.load('fr_core_news_sm')
stop_words = stopwords.words('french')

stop_words.extend(['avis','environnement','autorite','projet','etude','exploitation','impact','site','dossier','mission','regionale','mrae','mnhn'])
#%%

#A lancer pour preprocess les données une 1ere fois
data_final = pd.read_csv('data_final.csv')
data_final = data_final[data_final['len']>6000]
data_final.drop(['Unnamed: 0','len','url_etude','url_avis','departement'],axis = 1,inplace = True)
#%%
def preprocessing(text):
    sentences = text.split('.')
    textb4stopwords = []
    for sent in sentences:
        textb4stopwords.append(nlp(sent))
    string = []
    for textb4 in textb4stopwords:
        string.append(unidecode.unidecode(' '.join([token.lemma_ for token in textb4 if token.text not in stop_words])))

    return(string)
data_final['text_processed'] = data_final["txt_AAE"].apply(preprocessing)

#%%
data_final.drop(['txt_AAE'],axis= 1,inplace = True)
data_final.to_csv('data_treated.csv')

#%%

#Chargement des données
data_final = pd.read_csv('data_treated.csv')
#%%

#A lancer pour récupérer les données sous forme d'une liste de
# docs 
import ast
data = []
for doc in data_final["text_processed"].values:
    data.append(''.join(sent for sent in doc))
#%%

#Vecto des données
countVecto = CountVectorizer(min_df = 0, max_df = 0.9,
 ngram_range=(1,3), stop_words = stop_words)

#Process de data ou data_final.text_processed selon le besoin
process = countVecto.fit_transform(data)  
X = process.toarray().astype(int)
sum_words = X.sum(axis = 0)
word2id = countVecto.vocabulary_
id2word = {idd:word for word, idd in word2id.items() }
words_freq = [(word, sum_words[idx]) for word, idx in     word2id.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
vocab = tuple(word2id.keys())
vocab_alphaorder = sorted(vocab, key = lambda x : x[1], reverse = False)

#%%
import pickle
import pandas as pd

enjeux = pd.read_csv('enjeux.csv',delimiter = ';')
from unidecode import unidecode

enjeux.applymap(unidecode)
def itsplit(row,delimiter = ', '):
    return (row.split(delimiter))
enjeux.Dictionnaire = enjeux.Dictionnaire.apply(itsplit)

pickle.dump(enjeux,open('Thesaurus.pickle','wb'))
#%%
import re
import unidecode
def preprocessing_mot(text):
    prepro = nlp(text)
    lem = ' '.join(token.lemma_ for token in prepro if token.lemma_ not in stop_words)
    s = re.sub(r'[^\w\s]','',lem)
    s = s.lower()
    s = unidecode.unidecode(s)
    return(s)

Thesaurus = pickle.load(open('Thesaurus.pickle','rb'))
thesau_list_unpro = list(Thesaurus.Dictionnaire.values)
enjeux_list = list(Thesaurus.Enjeux.values)
thesau_list = []
for enjeu in thesau_list_unpro:
    thesau_list.append([preprocessing_mot(mot) for mot in enjeu])

dicoThesau = {k:v for k,v in zip(enjeux_list,thesau_list)}
pickle.dump(dicoThesau,open('Thesaurus_dico.pickle','wb'))


#%%
from guidedUtils import AFE, coverage_all
from scipy.sparse import csr_matrix
from time import time

start = time()
Xsparse = csr_matrix(X)
record = []
covs = []
cooc = csr_matrix.dot(Xsparse.T,Xsparse).toarray()

print(time()-start)
#%%
for k in range(len(cooc.sum(axis=1))):
    if cooc.sum(axis=1)[k] == 0:
        print(k)
#%%
for i in range(1,10):
    test_thesau = []
    for thes in thesau_list:
        mini = []
        k= 0
        for k in range(i):
            mini.append(thes[k])
        test_thesau.append(mini)
    cov = coverage_all(test_thesau,word2id,X)
    covs.append(cov)    
    entrop = AFE(cooc,test_thesau,word2id)
    record.append(entrop)
#%%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1,2)
axes[0].set_title('Entropie')
axes[0].plot([i for i in range(len(record))],record)
axes[1].set_title('Cover')
axes[1].plot([i for i in range(len(covs))],covs)

#%%
import random
record = []
added = []
entrop1 = AFE(cooc,thesau_list,word2id)
record.append(entrop1)
T = thesau_list.copy()
added.append('Knowledge based')
for k in range(0,3000):
    ii = random.randint(0,5)
    bb = random.randint(0,len(words_freq))
    T[ii].append(words_freq[k][0])
    added.append(words_freq[k][0])
    varentrop = AFE(cooc,T,word2id)
    record.append(varentrop)



#%%
import matplotlib.pyplot as plt
record[0] = entrop1
plt.plot(added,record)
#%%
# Normal LDA without seeding
model = guidedlda.GuidedLDA(n_topics=6, n_iter=1000, random_state=7, refresh=200)
model.fit(X)

topic_word = model.topic_word_
n_top_words = 10
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


#%%
# Guided LDA with seed topics.
seed_topic_list = thesau_list

model = guidedlda.GuidedLDA(n_topics=len(enjeux_list), n_iter=1000, random_state=0, refresh=200)
length = 0
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    length+=len(st)
    for word in st:
        try:
            seed_topics[word2id[word]] = t_id
        except:
            print(word,' not in vocab')

print(len(seed_topics)/length)
#%%
model.fit(X, seed_topics=seed_topics, seed_confidence=0.9)
#%%
n_top_words = 20
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
# %%
import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct



# Train the CorEx topic model
topic_model = ct.Corex(n_hidden=len(enjeux_list))  # Define the number of latent (hidden) topics to use.
topic_model.fit(X,words = vocab)
# %%
topics = topic_model.get_topics()
for topic_n,topic in enumerate(topics):
    # w: word, mi: mutual information, s: sign
    topic = [(w,mi,s) if s > 0 else ('~'+w,mi,s) for w,mi,s in topic]
    # Unpack the info about the topic
    words,mis,signs = zip(*topic)    
    # Print topic
    topic_str = str(topic_n+1)+': '+', '.join(words)
    print('\n'+topic_str)
# %%
import scipy.sparse as ss
from corextopic import corextopic as ct

X = np.matrix(X)
topic_model = {}

for k in range(1,10):
    topic_model[k]=ct.Corex(n_hidden=len(enjeux_list))
    topic_model[k].fit(X, words=vocab, anchors=thesau_list, anchor_strength=k)
# %%
topics = topic_model.get_topics()
for topic_n,topic in enumerate(topics):
    # w: word, mi: mutual information, s: sign
    topic = [(w,mi,s) if s > 0 else ('~'+w,mi,s) for w,mi,s in topic]
    # Unpack the info about the topic
    words,mis,signs = zip(*topic)    
    # Print topic
    topic_str = str(enjeux_list[topic_n])+': '+', '.join(words)
    print(topic_str)
 # %%


# %%
