#%%
from joblib.parallel import cpu_count
import pandas as pd
import numpy as np
import torch
import pickle
import time
import functools
import operator
from joblib import Parallel,delayed
from functools import partial
import sentencepiece as spm 
import psutil
from tqdm import tqdm
import pickle
import os
import re
from unidecode import unidecode
from bs4 import BeautifulSoup
import gensim

from fats import Make_Embedding, Make_Extractive,Word_Cleaning,Resume,TextRank
#%%
os.chdir("C:\\Users\\theo.roudil-valentin\\Documents\\Resume\\MLSUM")

texts=[i for i in os.listdir("test/") if ('text_clean' in i) and ('test' in i)]
# print(texts)

text_clean=pickle.load(open("test/"+texts[0],'rb'))
text_clean[0]

Paragraphes=text_clean[0]
#introduisez ici une suite de phrases non-nettoyées dans une liste
# Paragraphes = [ phrase1, phrase2, ..., phraseN ]

from transformers import CamembertTokenizer,CamembertModel
tok=CamembertTokenizer('MLSUM_tokenizer.model')
camem=CamembertModel.from_pretrained("camembert-base")

W2V=gensim.models.Word2Vec.load("W2V_all.model")

cpu_max=int(psutil.cpu_count()/6)

WC=Word_Cleaning(n_jobs=cpu_max,
                sentence=True,
                threshold=True,
                seuil=2,
                lemma=False,
                seuil_carac=3)
text=WC.make_documents(Paragraphes)
text=WC.remove_empty(text)
text=functools.reduce(operator.iconcat, text, [])
# ME=Make_Extractive(cpu=psutil.cpu_count())
# sentence=ME.make_w2v_sentences(text)
# text_=Parallel(n_jobs=cpu_max)(delayed(ME.make_splitting)(s) for s in text)

# dico_test=ME.make_encoding(text_clean,tokenizer='MLSUM_tokenizer.model',prefix="test",name="Tokenizer_test",split=1)

# dico=ME.make_encoding(text,
#                       tokenizer='MLSUM_tokenizer.model',
#                       voc_size=12000,
#                       split=1,
#                       training=False)

#%%
TR=TextRank(tok_path='MLSUM_tokenizer.model')
# TRB=partial(TR.make_resume,type='bert',modele=camem,k=2)
TR.make_resume(text,type='bert',modele=camem,k=2)
#%%
TR.make_embedding_bert(text,camem)
#%%
ME=Make_Embedding(tok=CamembertTokenizer('MLSUM_tokenizer.model'),cpu=4)
i,a=ME.make_token(text,4)
em=camem(torch.tensor(i),torch.tensor(a))
#%%
nphrase=2
R=Resume(type_='TextRankBert', #choisissez le type de résumé que vous voulez
        k=nphrase, #le nombre de phrases
        modele=camem, #le modèle d'embedding à utiliser, si besoin
        tok_path='MLSUM_tokenizer.model') # le chemin vers le tokenizer, si besoin
resume=R.resume(text)
resume
# %%
nphrase=2
R=Resume(type_='Random', #choisissez le type de résumé que vous voulez
        k=nphrase, #le nombre de phrases
        modele=None, #le modèle d'embedding à utiliser, si besoin
        tok_path=None) # le chemin vers le tokenizer, si besoin
resume=R.resume(text)
resume
# %%
nphrase=2
R=Resume(type_='TextRankWord2Vec', #choisissez le type de résumé que vous voulez
        k=nphrase, #le nombre de phrases
        modele=W2V, #le modèle d'embedding à utiliser, si besoin
        tok_path=None) # le chemin vers le tokenizer, si besoin
resume=R.resume(text)
resume
# %%
