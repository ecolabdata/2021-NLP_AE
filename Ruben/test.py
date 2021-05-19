#%%
import pandas as pd
import lda2vec.nlppipe
import spacy
import numpy as np
import re
import lda2vec.utils
import os
import lda2vec.Lda2vec
#Depuis \Ruben
path = 'Avis_Id_TextProcessed'
done = True
#%%

data_final = pd.read_csv('data_final.csv')
data_final.drop(['Unnamed: 0'],axis = 1)
data_final = data_final[data_final['len']>6000]

themes=list(np.unique(re.sub("[\(\[].*?[\)\]]", "",
    re.sub(","," ",
        re.sub(";"," ",' '.join(np.unique(data_final.theme.values))))).split(' ')))
themes.remove('ET'),themes.remove('')

datatheme = {}
for theme in themes:
    datatheme[theme] = pd.DataFrame(data = None,columns = data_final.columns)

def seprow(row):
    thematiques = list(np.unique(re.sub("[\(\[].*?[\)\]]", "",
    re.sub(","," ",
        re.sub(";"," ",''.join(row.theme)))).split(' ')))
    while 'ET' in thematiques:
        thematiques.remove('ET')
    while '' in thematiques:
        thematiques.remove('')
    for theme in thematiques:
        datatheme[theme] = datatheme[theme].append(row,ignore_index = True)

data_final.apply(seprow,axis = 1)
idx_to_word, word_to_idx, freqs, pivot_ids, target_ids, doc_ids = {},{},{},{},{},{}

#for theme in themes:
#   idx_to_word[theme], word_to_idx[theme], freqs[theme], pivot_ids[theme], target_ids[theme], doc_ids[theme] = None,None,None,None,None,None
#%%
if done == True:
    for theme in themes:
        idx_to_word[theme], word_to_idx[theme], freqs[theme], pivot_ids[theme], target_ids[theme], doc_ids[theme] = lda2vec.utils.load_preprocessed_data(path+'\\'+theme)
else:
    for theme in themes:
        data_processor = lda2vec.nlppipe.Preprocessor(df = datatheme[theme], textcol = 'txt_AAE')
        data_processor.preprocess()
        data_processor.save_data(path+'\\'+theme)
        idx_to_word[theme], word_to_idx[theme],freqs[theme], pivot_ids[theme], target_ids[theme], doc_ids[theme] = lda2vec.utils.load_preprocessed_data(path+'\\'+theme)
    done = True

#%%
import tensorflow.compat.v1 as tf
models = {}
for theme in themes:
    tf.reset_default_graph()
    models[theme] = lda2vec.Lda2vec.Lda2vec(num_unique_documents = len(np.unique(doc_ids[theme])), vocab_size = len(idx_to_word[theme])+1, num_topics = 10)
    models[theme].train(pivot_words = pivot_ids[theme],
                target_words = target_ids[theme],
                doc_ids = doc_ids[theme],
                data_size = len(pivot_ids[theme]),
                num_epochs = int(100),
                switch_loss_epoch = 0)


# %%

import pyLDAvis

topics = model.prepare_topics('document_id', vocab)
prepared = pyLDAvis.prepare(topics)
pyLDAvis.display(prepared)

# %%

enjeux = pd.read_csv('enjeux.csv',delimiter = ';')
from unidecode import unidecode

enjeux.applymap(unidecode)
def itsplit(row,delimiter = ','):
    return (row.split(delimiter))
enjeux.Dictionnaire.apply(split,args = (','))
# %%
