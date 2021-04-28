#%%

####!!!! FONCTIONNE UNIQUEMENT SOUS PYTHON 3.7 (en tout cas ne marche pas en 3.9)


import pandas as pd
import numpy as np
import re
import os
#Depuis \Ruben
path = 'Avis_Id_TextProcessed'
done = True

#%%

data_final = pd.read_csv('data_final.csv')
data_final = data_final[data_final['len']>6000]
data_final.drop(['Unnamed: 0','len','url_etude','url_avis','departement'],axis = 1,inplace = True)

#%%
import numpy as np
import guidedlda
X = guidedlda.datasets.load_data(guidedlda.datasets.NYT)
#%%
vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.NYT)
word2id = dict((v, idx) for idx, v in enumerate(vocab))
#%%
print(X.shape)

print(X.sum())
#%%
# NLTK Stop words
from nltk.corpus import stopwords
nltk.set_proxy('http://cache.ritac.i2:32000')
nltk.download('punkt')
nltk.data.load('tokenizers/punkt/french.pickle')
nltk.download('stopwords')
stop_word=nltk.corpus.stopwords.words('french')
from gensim.utils import simple_preprocess
import re
from sklearn.feature_extraction.text import CountVectorizer

stop_words = stopwords.words('french')

stop_words.extend(['avis','environnement','autorite','projet','etude','exploitation','impact','site','dossier','mission','regionale','mrae'])
#%%
def preprocessing(text):
    textb4stopwords = simple_preprocess(str(text), deacc=True)
    string = ' '.join([word for word in textb4stopwords if word not in stop_words])
    return(string)

data_final['text_processed'] = data_final["txt_AAE"].apply(preprocessing)

#%%
countVecto = CountVectorizer(min_df = 15, max_df = 0.9,
 ngram_range=(1,3), stop_words = stop_words)
process = countVecto.fit_transform(data_final["text_processed"].values)


word2id = countVecto.vocabulary_
vocab = tuple(word2id.keys())
#%%
# Normal LDA without seeding
model = guidedlda.GuidedLDA(n_topics=10, n_iter=1000, random_state=7, refresh=200)
model.fit(X)

topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

#%%
# Guided LDA with seed topics.
seed_topic_list = [['eolien'],
                   ['photovoltaique'],
                   ['faune'],
                   ['flore'],
                   ['biodiversite']]

model = guidedlda.GuidedLDA(n_topics=5, n_iter=1000, random_state=7, refresh=200)

seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id

model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)

n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
# %%
