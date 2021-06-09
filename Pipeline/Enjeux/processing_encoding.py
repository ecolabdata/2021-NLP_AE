
# In[353]:



import pandas as pd, numpy as np
import pickle
import unidecode
import spacy
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import matplotlib.pyplot as plt
stop_words = stopwords.words('french')
stop_words.extend(['avis','environnement','autorite','projet','etude','exploitation','impact','site','dossier','mission','regionale','mrae','mnhn'])

nlp = spacy.load('fr_core_news_sm')

if __name__ == "__main__" :
    docs_df = pickle.load(open("Data\Workinprogress\\base_id_avis_txt_sorted",'rb'))
    Thesaurus = pickle.load(open("Data\Thesaurus_csv\Thesaurus1.pickle",'rb'))

#%%
# ## 0. Préparation du Thesaurus et des données

# Preprocessing a appliquer sur le thesaurus. 
# Attention, il faudra appliquer le même processing sur les textes pour reconnaitre les mots du thesaurus !


#%%

def processing_mot(text):
    prepro = nlp(text)
    lem = ' '.join(token.lemma_ for token in prepro if token.lemma_ not in stop_words and not len(token.text)<=2 )
    s = re.sub(r'[^\w\s]','',lem)
    s = re.sub('  ',' ',s)
    s = s.lower()
    s = unidecode.unidecode(s)
    return(s)

def processing(text):
    text = nlp(text)
    string = unidecode.unidecode(' '.join([token.lemma_ for token in text if token.text not in stop_words and not len(token.text)<=2 ]))
    string = re.sub(r'[^\w\s\n\t\r]','',string)
    string = re.sub(r'[\n\t\r]','',string)
    string = re.sub(r'  ',' ',string)
    return(string)


# In[359]:

def clean_thesau(Thesaurus):
    """
    Thesaurus : dataframe avec deux colonnes "Enjeux" et "Dictionnaire"
    """
    thesau_list_unprocessed = list(Thesaurus.Dictionnaire.values)
    thesau_list = []
    for enjeu in thesau_list_unprocessed:
        thesau_list.append([processing_mot(mot) for mot in enjeu if mot != ''])

    enjeux_list = list(Thesaurus['Enjeu environnemental'].values)

    #dicoThesau = {k:v for k,v in zip(enjeux_list,thesau_list)}

    thesau_df = pd.DataFrame({'Enjeux': enjeux_list,'Dictionnaire' : thesau_list})
    return(thesau_df)

if __name__ == "__main__" :
    Thesaurus = clean_thesau(Thesaurus)
    pickle.dump(Thesaurus,open("Data\Thesaurus_csv\\Thesaurus1_clean.pickle",'wb'))

    tqdm.pandas(desc="Processing text")
    docs_df['text_processed'] = docs_df.texte.progress_apply(processing)
    #docs_df.drop(['texte'],axis = 1,inplace = True)

    pickle.dump(docs_df,open('Data/Workinprogress/docs_df.pickle','wb'))


#%%
# ## 1. Analyse du vocabulaire existant

# On vectorise les données et on crée des listes pratiques pour manipuler les noms de features, etc...
if __name__ == "__main__" :
    docs_df = pickle.load(open('Data/Workinprogress/docs_df.pickle','rb'))
    Thesaurus = pickle.load(open("Data\Thesaurus_csv\\Thesaurus1_clean.pickle",'rb'))
#%%

if __name__ == "__main__" :
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



    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,9))
    for enjeu in enjeux_list:
        plt.plot([k for k in range(30)],cov[enjeu],label = enjeu)

    plt.legend(loc = 1)
    plt.show()
    size


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


    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,9))
    for enjeu in enjeux_list:
        plt.plot([k/100 for k in range(30)],cov[enjeu],label = enjeu)

    plt.legend(loc = 0)
    plt.show()
    size


# A partir des évaluations précédentes, il faut choisir les paramètres de manière a garder le plus grand nombre de mots des dicos dans le vocabulaire sans pour autant prendre un vocabulaire trop grand.

# In[370]:


def get_info(vectoriseur,matrix,thesaurus_df):
    """
    Entrées : 
    - vectoriseur : instance de CountVectorizer sklearn

    Sorties :
    - 0 word2id
    - 1 vocabulaire
    - 2 words_freq
    - 3 vocabulaire trié par fréquence
    - 4 mots pas dans le vocabulaire mais dans le thésaurus


    """

    countVecto = vectoriseur
    X = matrix
    sum_words = X.sum(axis = 0).tolist()[0]
    word2id = countVecto.vocabulary_
    vocab = tuple(word2id.keys())

    cov = {}
    id2word = {idd:word for word, idd in word2id.items() }
    words_freq = [(word, sum_words[idx]) for word, idx in     word2id.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    vocab_sort = list(vocab)
    vocab_sort.sort()

    notinvoc = {}
    thesau_list = list(thesaurus_df.Dictionnaire)
    enjeux_list = list(thesaurus_df.Enjeux)
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


    plt.xticks(rotation=90)
    plt.bar([k for k in range(len(cov))],
        [cov[key] for key in cov.keys()], 
        tick_label = [key for key in cov.keys()])

    return(word2id,vocab,words_freq,vocab_sort,notinvoc)



# In[350]:

def isinvocab(mot):
    for line in vocab_sort:
        if mot in line:
            return(True)
    return(False)


# On constate que certains mots n'apparaissent pas la plupart du temps car :  
# 
#    1/ Le mot transformé du thésaurus n'est pas transformé identiquement dans le corpus  
#    2/ Il y'a parfois plusieurs versions du mot/bigramme/trigramme avec de légères variations   
#     
# Occasionnellement :   
#   
#    Le mot/bi/tri n'apparait pas du tout dans le corpus (autosolisme, analyse cycle vie, nuage toxique, radon...)


