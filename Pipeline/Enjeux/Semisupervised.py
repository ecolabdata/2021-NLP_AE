
# In[353]:



import pandas as pd, numpy as np

import pickle
import unidecode
import spacy
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer

stop_words = stopwords.words('french')
stop_words.extend(['avis','environnement','autorite','projet','etude','exploitation','impact','site','dossier','mission','regionale','mrae','mnhn'])

nlp = spacy.load('fr_core_news_sm')
# Read recipe inputs
avis_Id_Text_Preprocessed_joined = dataiku.Dataset("Avis_Id_Text_Preprocessed_joined")
avis_Id_Text_Preprocessed_joined_df = avis_Id_Text_Preprocessed_joined.get_dataframe()
thesaurus = dataiku.Folder("XXZ13n5V")
thesaurus_info = thesaurus.get_info()
path = thesaurus.get_path()


# ## 0. Préparation du Thesaurus et des données

# Preprocessing a appliquer sur le thesaurus. Attention, il faudra appliquer le même processing sur les textes pour reconnaitre les mots du thesaurus !
# 

# In[354]:


def processing_mot(text):
    prepro = nlp(text)
    lem = ' '.join(token.lemma_ for token in prepro if token.lemma_ not in stop_words and not len(token.text)<=2 )
    s = re.sub(r'[^\w\s]','',lem)
    s = re.sub('  ',' ',s)
    s = s.lower()
    s = unidecode.unidecode(s)
    return(s)


# In[359]:


Thesaurus = pickle.load(open(path+'/Thesaurus1.pickle','rb'))
thesau_list_unpro = list(Thesaurus.Dictionnaire.values)
enjeux_list = list(Thesaurus['Enjeu environnemental'].values)
thesau_list = []
for enjeu in thesau_list_unpro:
    thesau_list.append([processing_mot(mot) for mot in enjeu if mot != ''])

dicoThesau = {k:v for k,v in zip(enjeux_list,thesau_list)}
dicoThesau


# In[360]:


Thesaurus


# In[361]:


docs_df = avis_Id_Text_Preprocessed_joined_df.drop(['url_etude','url_avis','departement','titre','theme'],axis = 1)
docs_df.head(5)


# In[364]:


from tqdm import tqdm
tqdm.pandas(desc="Processing text")
def processing(text):
    text = nlp(text)
    string = unidecode.unidecode(' '.join([token.lemma_ for token in text if token.text not in stop_words and not len(token.text)<=2 ]))
    string = re.sub(r'[^\w\s]','',string)
    string = re.sub(r'  ',' ',string)
    return(string)

docs_df['text_processed'] = docs_df.text_preprocessed.progress_apply(processing)
docs_df.drop(['text_preprocessed'],axis = 1,inplace = True)


# In[365]:


docs_df


# ## 1. Analyse du vocabulaire existant

# On vectorise les données et on crée des listes pratiques pour manipuler les noms de features, etc...
# 

# In[366]:


countVecto = CountVectorizer(min_df = 0, max_df = 0.9, ngram_range=(1,3), stop_words = stop_words)
process = countVecto.fit_transform(docs_df.text_processed.values)  
word2id = countVecto.vocabulary_
vocab = tuple(word2id.keys())
cov = {}
size = [len(vocab)]
for words,enjeu in zip(thesau_list,enjeux_list):
    t = len(words)
    c = 0
    for word in words:
        if word in vocab:
            c+=1
    cov[enjeu] = [c/t]
    
for k in range(1,30):
    countVecto = CountVectorizer(min_df = k, max_df = 0.9, ngram_range=(1,3), stop_words = stop_words)
    process = countVecto.fit_transform(docs_df.text_processed.values)  
    word2id = countVecto.vocabulary_
    vocab = tuple(word2id.keys())
    size.append(len(vocab))
    for words,enjeu in zip(thesau_list,enjeux_list):
        t = len(words)
        c = 0
        for word in words:
            if word in vocab:
                c+=1
        cov[enjeu].append(c/t)


# In[367]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,9))
for enjeu in enjeux_list:
    plt.plot([k for k in range(30)],cov[enjeu],label = enjeu)

plt.legend(loc = 1)
plt.show()
size


# In[380]:


countVecto = CountVectorizer(min_df = 0, max_df = 1, ngram_range=(1,3), stop_words = stop_words)
process = countVecto.fit_transform(docs_df.text_processed.values)  
word2id = countVecto.vocabulary_
vocab = tuple(word2id.keys())
cov = {}
size = [len(vocab)]
for words,enjeu in zip(thesau_list,enjeux_list):
    t = len(words)
    c = 0
    for word in words:
        if word in vocab:
            c+=1
    cov[enjeu] = [c/t]
    
for k in range(1,30):
    countVecto = CountVectorizer(min_df = 0, max_df = 1-k/100, ngram_range=(1,3), stop_words = stop_words)
    process = countVecto.fit_transform(docs_df.text_processed.values)  
    word2id = countVecto.vocabulary_
    vocab = tuple(word2id.keys())
    size.append(len(vocab))
    for words,enjeu in zip(thesau_list,enjeux_list):
        t = len(words)
        c = 0
        for word in words:
            if word in vocab:
                c+=1
        cov[enjeu].append(c/t)


# In[381]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,9))
for enjeu in enjeux_list:
    plt.plot([k/100 for k in range(30)],cov[enjeu],label = enjeu)

plt.legend(loc = 0)
plt.show()
size


# A partir des évaluations précédentes, il faut choisir les paramètres de manière a garder le plus grand nombre de mots des dicos dans le vocabulaire sans pour autant prendre un vocabulaire trop grand.

# In[370]:


countVecto = CountVectorizer(min_df = 3, max_df = 0.95, ngram_range=(1,3), stop_words = stop_words)
process = countVecto.fit_transform(docs_df.text_processed.values)  
X = process.toarray().astype(int)
sum_words = X.sum(axis = 0)
word2id = countVecto.vocabulary_
vocab = tuple(word2id.keys())

id2word = {idd:word for word, idd in word2id.items() }
words_freq = [(word, sum_words[idx]) for word, idx in     word2id.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
vocab_sort = list(vocab)
vocab_sort.sort()

notinvoc = {}
for words,enjeu in zip(thesau_list,enjeux_list):
    t = len(words)
    c = 0
    notinvoc[enjeu] = []
    for word in words:
        if word in vocab:
            c+=1
        else:
            notinvoc[enjeu].append(word)
    cov[enjeu] = c/t

print(cov,len(vocab))


# In[371]:


notinvoc


# In[350]:


mot = 'seul'
for line in vocab_sort:
    if mot in line:
        print(line)


# On constate que certains mots n'apparaissent pas la plupart du temps car :  
# 
#    1/ Le mot transformé du thésaurus n'est pas transformé identiquement dans le corpus  
#    2/ Il y'a parfois plusieurs versions du mot/bigramme/trigramme avec de légères variations   
#     
# Occasionnellement :   
#   
#    Le mot/bi/tri n'apparait pas du tout dans le corpus (autosolisme, analyse cycle vie, nuage toxique, radon...)

# ## 2. Topic modeling semi-supervisé

# In[372]:


import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct


# In[375]:


topic_model = {}
X = np.matrix(X)
for k in range(10):
    topic_model[k] = ct.Corex(n_hidden=len(enjeux_list))
    topic_model[k].fit(X, words=vocab, anchors=thesau_list, anchor_strength=k+1)


# In[379]:


k = 9

topics = topic_model[5].get_topics()
for topic_n,topic in enumerate(topics):
    # w: word, mi: mutual information, s: sign
    topic = [(w,mi,s) if s > 0 else ('~'+w,mi,s) for w,mi,s in topic if w not in dicoThesau[enjeux_list[topic_n]]]
    # Unpack the info about the topic
    words,mis,signs = zip(*topic)    
    # Print topic
    topic_str = str(enjeux_list[topic_n])+': '+', '.join(words)
    print(topic_str)


# In[ ]:


semisupervised_results_df = avis_Id_Text_Preprocessed_joined_df # For this sample code, simply copy input to output


# Write recipe outputs
semisupervised_results = dataiku.Dataset("Semisupervised_results")
semisupervised_results.write_with_schema(semisupervised_results_df)

