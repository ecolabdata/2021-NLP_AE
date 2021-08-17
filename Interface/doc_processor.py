#%%
#Exemple d'utilisation des pipelines pour générer une base de donnée avec toutes les informations
import numpy as np
import pandas as pd
import pickle
import torch

#%%
#On prend une base de donnée avec les sections découpées
df = pd.read_csv(open("Data\\sections_cool_avecresume.csv",'rb'),sep=";")
num_etudes = np.unique(df.num_etude.tolist())
df.drop(['level_0', 'index', 'section_html',
       'section_clean_1', 'vide', 'longueur', 'pour_resume',
       'resume', 'resume_2', 'pour_resume_2', 'index_resume', 'resume_propre'],axis = 1,inplace = True)
#%%
#On choisit une seule étude ici mais on pourrait appliquer ça a l'ensemble de la BDD
option = num_etudes[0]
df = df[df['num_etude']==option]
df.dropna(inplace = True)
#%%
#Suite de phrases non-nettoyées dans une liste
Paragraphes= df.clean_2.values


#%%
#Ici, on applique le pipeline de résumé automatique
import functools
import operator
import psutil
import gensim
from tqdm import tqdm
from transformers import CamembertTokenizer,CamembertModel
import os

tok=CamembertTokenizer('Data/MLSUM/MLSUM_tokenizer.model')
camem=CamembertModel.from_pretrained("camembert-base")

W2V=gensim.models.Word2Vec.load("Data/MLSUM/W2V_all.model")
import time
import Pipeline.Resume.fats as fats
import pickle

#On se place dans le dossier avec MLSUM pour que l'algo aille chercher le modèle au bon endroit
os.chdir('Data/MLSUM')
#Découpage en phrases
P=[i.split('.') for i in Paragraphes.tolist()]

resu,text_2=fats.Resume(P,
                 DL=False, # True si on veut utiliser des modèles de Deep Learning, False sinon
                 cpu=4, #le nombre de cpu à utiliser, préférez peu de CPU, pour la mémoire
                 type_='TextRankWord2Vec', #le nom du modèle, si DL=False
                 k=2, #Le nombre de phrases
                 choose_model=None, #le nom du modèle de Deep Learning, le cas échéant
                 tok='MLSUM_tokenizer.model', #le nom du tokenizer
                 modele=W2V, #modèle CamemBERT ou W2V selon le modèle choisi
                 get_score_only=True,# est-ce qu'on veut juste le score et pas directement les phrases
                 s=True)
#%%
print(resu) 
text_2

#%%
#On charge le modèle pour le kewyord extraction
from Pipeline.Keywords.keyword_extraction import extract_keywords_doc
from nltk.corpus import stopwords
stop_words = stopwords.words('french')
df['keywords'] = extract_keywords_doc(Paragraphes,
                                maxlen = 1000,
                                language='fr',
                                n_top=4,
                                keyphrases_ngram_max=2,
                                consensus='statistical',
                                models=['keybert','yake','textrank'])

#%%
#On charge un thésaurus
from Pipeline.Enjeux.processing_encoding import processing_thesaurus,processing
Thesaurus = pickle.load(open('Data\Thesaurus_csv\Thesaurus1.pickle','rb'))
#Le préprocessing permet de lemmatiser les mots du thésaurus de la même manière que les mots du texte vont l'être (sinon ils ne seront pas reconnus)
Thesaurus = processing_thesaurus(Thesaurus)

from Pipeline.Enjeux.topicmodeling_pipe import CorExBoosted
extracteur_enjeux = CorExBoosted(df,Thesaurus,model = pickle.load(open('Data/enjeux_section.pickle','rb')))

enjeux = extracteur_enjeux.extract_topics(Paragraphes)

# %%
