
#%%
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


#%%
# ## 0. Préparation du Thesaurus et des données

# Preprocessing a appliquer sur le thesaurus. 
# Attention, il faudra appliquer le même processing sur les textes pour reconnaitre les mots du thesaurus !


#%%

#Fonctions permettant de preprocess le thésaurus
def processing_mot(text):
    prepro = nlp(text)
    lem = ' '.join(token.lemma_ for token in prepro if token.lemma_ not in stop_words and not len(token.text)<=2 )
    s = re.sub(r'[^\w\s]','',lem)
    s = re.sub('  ',' ',s)
    s = s.lower()
    s = unidecode.unidecode(s)
    return(s)

def processing_thesaurus(Thesaurus):
    """
    Thesaurus : dataframe avec deux colonnes "Enjeux" et "Dictionnaire"
    """
    thesau_list_unprocessed = list(Thesaurus.Dictionnaire.values)
    thesau_list = []
    for enjeu in thesau_list_unprocessed:
        thesau_list.append([processing_mot(mot) for mot in enjeu if mot != ''])
    enjeux_list = list(Thesaurus['Enjeu environnemental'].values)
    thesau_df = pd.DataFrame({'Enjeux': enjeux_list,'Dictionnaire' : thesau_list})
    return(thesau_df)


def processing(text):
    text = nlp(text)
    string = unidecode.unidecode(' '.join([token.lemma_ for token in text if token.text not in stop_words and not len(token.text)<=2 ]))
    string = re.sub(r'[^\w\s\n\t\r]','',string)
    string = re.sub(r'[\n\t\r]','',string)
    string = re.sub(r'  ',' ',string)
    return(string)

#%%
# ## 1. Analyse du vocabulaire existant

# On vectorise les données et on crée des listes pratiques pour manipuler les noms de features, etc...

def analysis_min_df(text,Thesaurus,N=15):
    """
    Fonction d'analyse du paramètre min_df du vectoriseur en terme de présence des mots du thésaurus dans le vocabulaire du vectoriseur
    Thesaurus : dataframe avec deux colonnes "Enjeux" et "Dictionnaire"
    """
    thesau_list,enjeux_list = Thesaurus.Dictionnaire.tolist(),Thesaurus.Enjeux.tolist()
    countVecto = CountVectorizer(min_df = 0, max_df = 0.9, ngram_range=(1,3), stop_words = stop_words)
    process = countVecto.fit_transform(text)  
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
        
    for k in range(1,N+1):
        countVecto = CountVectorizer(min_df = k, max_df = 0.9, ngram_range=(1,3), stop_words = stop_words)
        process = countVecto.fit_transform(text)  
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
        plt.plot([k for k in range(N+1)],cov[enjeu],label = enjeu)

    plt.legend(loc = 1)
    plt.show()
    return(cov)

def analysis_max_df(text,Thesaurus,N = 15):
    """
    Fonction d'analyse du paramètre max_df du vectoriseur en terme de présence des mots du thésaurus dans le vocabulaire du vectoriseur
    Thesaurus : dataframe avec deux colonnes "Enjeux" et "Dictionnaire"
    """
    thesau_list,enjeux_list = Thesaurus.Dictionnaire.tolist(),Thesaurus.Enjeux.tolist()
    countVecto = CountVectorizer(min_df = 0, max_df = 1, ngram_range=(1,3), stop_words = stop_words)
    process = countVecto.fit_transform(text)  
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
        
    for k in range(1,N+1):
        countVecto = CountVectorizer(min_df = 0, max_df = 1-k/100, ngram_range=(1,3), stop_words = stop_words)
        process = countVecto.fit_transform(text)  
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
        plt.plot([k/100 for k in range(N+1)],cov[enjeu],label = enjeu)

    plt.legend(loc = 0)
    plt.show()
    size


# A partir des évaluations précédentes, il faut choisir les paramètres de manière a garder le plus grand nombre de mots des dicos
#  dans le vocabulaire sans pour autant prendre un vocabulaire trop grand.

def get_info(vectoriseur,matrix,thesaurus_df):
    """
    Entrées : 
    - vectoriseur : instance de CountVectorizer sklearn
    - matrix : matrice du fit_transform
    - thesaurus sous forme dataframe col Enjeux et Dictionnaire

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


def isinvocab(mot,vocab_sort):
    for line in vocab_sort:
        if mot in line:
            return(True)
    return(False)

