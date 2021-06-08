#%%
import time
import pickle
from pathlib import Path
import gensim
import pandas as pd
import numpy as np
import sklearn
import re
from unidecode import unidecode
import functools
import operator
import psutil
from joblib import Parallel,delayed
from functools import partial
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import os
os.chdir("C:/Users/theo.roudil-valentin/Documents/Resume/MLSUM/")
#%%
#############################################################################################
########  Téléchargement des données    #############################################################
#############################################################################################
# Pour ma part, cela ne fonctionnait pas : j'avais une erreur WinError 5 Access Denied
# Lorsque le code voulait renommer un dossier qu'il avait lui-même créer pou écrire les données
# Ce que j'ai fait : vu que le téléchargement des données ne pose pas de problème, il faut partir de ça
# Aller chercher le dossier où elles sont au format .jsonl
# Puis faire tourner le code pour les écrire soi-même
# ça prend quelques minutes 

from datasets import load_dataset
dataset = load_dataset(
   'mlsum','fr')

import json_lines
datasets="C:\\Users\\theo.roudil-valentin\\.cache\\huggingface\\datasets\\downloads\\extracted"
directories=os.listdir(datasets)
element={}
for dirr in directories:
   fil=datasets+'\\'+dirr
   name_file=os.listdir(fil)[0]
   fil=fil+'\\'+name_file
   name=name_file[:-6][3:]
   print(name)
   text=[]
   with open(fil, 'rb') as f:
    for item in tqdm(json_lines.reader(f)):
        text.append(item)
   element[name]=text

for name in list(element.keys()):
   base=pd.concat([pd.DataFrame(element[name][i],index=[0]) for i in range(len(element[name]))])
   pickle.dump(base,open('C:\\Users\\theo.roudil-valentin\\Documents\\Resume\\MLSUM\\MLSUM_fr_'+name+'.pickle','wb'))
   print("Base",name,"sauvegardée !")    
#%%
# Ici on va créer les dictionnaires qui contiennent les vecteurs sur 
# lesquels on va entraîner les modèles

for f in os.listdir():
   print(f)
   if f[-3:]!='csv':
      base=pickle.load(open('C:\\Users\\theo.roudil-valentin\\Documents\\Resume\\MLSUM\\'+f,'rb'))
      base.to_csv(f.split('.')[0]+'.csv')
   else:
      continue
#%%
from fats import Word_Cleaning, Make_Extractive
cpu=psutil.cpu_count()

fichiers=[i for i in os.listdir() if (i[-6:]=='pickle') and (i[:5]=='MLSUM')]

for f in fichiers[1:]:
   print(f)
   base=pickle.load(open('C:\\Users\\theo.roudil-valentin\\Documents\\Resume\\MLSUM\\'+f,'rb'))
   name=f.split('_')[-1].split('.')[0]

   print('Début du nettoyage du fichier',name,':')
   cleaning_time=time.time()
   WC=Word_Cleaning(n_jobs=cpu,sentence=True,threshold=True,seuil=2,lemma=False,seuil_carac=3)

   summary=base.summary.values
   summary=WC.make_summary(summary)
   pickle.dump(summary,open('summary_clean_'+name+'.pickle','wb'))

   text=base.text.values
   text=WC.make_documents(text)
   cleaning_end=time.time()
   pickle.dump(text,open('text_clean_'+name+'.pickle','wb'))
   print("Durée du nettoyage :",round((cleaning_end-cleaning_time)/60,2),"minutes.")
#%%
summaries=[i for i in os.listdir() if 'summary' in i]
texts=[i for i in os.listdir() if 'text' in i]

for summary,text in zip(summaries,texts):
   name=summary.split('.')[0].split('_')[-1]
   text=pickle.load(open(text,'rb'))
   summary=pickle.load(open(summary,'rb'))

   print("Début de la création de l'output :")
   ME=Make_Extractive()
   output_time=time.time()
   ME=Make_Extractive(fenetre=20,minimum=1,d=100,epochs=25)
   score,text_2,summary_2,W2V,erreur=ME.make_output(text,summary)
   output_end=time.time()
   print("Durée de la création d'output :",round((output_end-output_time)/60,2),"minutes.")

   pickle.dump(score,open('score_'+name+'.pickle','wb'))
   pickle.dump(text_2,open('text_word_'+name+'_2.pickle','wb'))
   pickle.dump(summary_2,open('summary_word_'+name+'.pickle','wb'))
   pickle.dump(erreur,open('erreur_word_'+name+'.pickle','wb'))
   
#%%
texts=[i for i in os.listdir() if 'text' in i]
scores=[i for i in os.listdir() if 'score' in i]

for text,score in zip(texts,scores):

   name=text.split('.')[0].split('_')[-1]
   text=pickle.load(open(text,'rb'))
   score=pickle.load(open(score,'rb'))
   print("Début de la création de l'encoding")
   encoding_time=time.time()
   ME=Make_Extractive(cpu)
   dico=ME.make_encoding(text,score,voc_size=12000,split=1)
   encoding_end=time.time()
   print("Durée de la création d'output :",round((output_end-output_time)/60,2),"minutes.")
   
   pickle.dump(dico,open('dico_'+name,'wb'))
   print('Les données du',name,"ont été sauvegardées !")
# %%
