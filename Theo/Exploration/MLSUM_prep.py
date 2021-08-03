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

from fats import Word_Cleaning, Make_Extractive
cpu=psutil.cpu_count()
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
# Etant donné que le train est trop gros, on va le découper en plusieurs bouts.

# os.path.getsize('C:\\Users\\theo.roudil-valentin\\Documents\\Resume\\MLSUM\\MLSUM_fr_train.pickle')/(1024**3)
base=pickle.load(open('C:\\Users\\theo.roudil-valentin\\Documents\\Resume\\MLSUM\\MLSUM_fr_train.pickle','rb'))
l=len(base)
q=3
for i in range(q):
   n_1=int(l/4)*i
   n_2=int(l/4)*(i+1)
   print(n_2)
   pickle.dump(base.iloc[n_1:n_2,:],open('C:\\Users\\theo.roudil-valentin\\Documents\\Resume\\MLSUM\\MLSUM_fr_train_'+str(i+1)+'.pickle','wb'))
pickle.dump(base.iloc[n_2:,:],open('C:\\Users\\theo.roudil-valentin\\Documents\\Resume\\MLSUM\\MLSUM_fr_train_4.pickle','wb'))

#%%
fichiers=[i for i in os.listdir() if (i[-6:]=='pickle') and (i[:5]=='MLSUM')]
fichiers_train=[i for i in fichiers if 'train' in i]

for f in fichiers_train[1:]:
   print(f)
   base=pickle.load(open('C:\\Users\\theo.roudil-valentin\\Documents\\Resume\\MLSUM\\'+f,'rb'))
   name=f.split('.')[0].split('fr')[-1][1:]

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
summaries_train=[i for i in os.listdir() if ('summary' in i) and ('train' in i)]
texts_train=[i for i in os.listdir() if ('text' in i) and  ('train' in i)]
#%%
text=[]

for t in texts_train:
   t=pickle.load(open('C:\\Users\\theo.roudil-valentin\\Documents\\Resume\\MLSUM\\'+t,'rb'))
   text=text+t
#%%
ME=Make_Extractive(fenetre=20,minimum=1,d=100,epochs=25)
W2V=ME.make_W2V(text) # 64 minutes
pickle.dump(W2V,open('W2V_train.pickle','wb'))
#%%
W2V=pickle.load(open('W2V_train.pickle','rb'))
ME=Make_Extractive(cpu=cpu)
vocab=list(W2V.wv.vocab.keys())

#%%
temps=[]
for summary,text in zip(summaries_train,texts_train):
   name=summary.split('.')[0].split('_')[-1]
   text=pickle.load(open(text,'rb'))
   summary=pickle.load(open(summary,'rb'))

   start=time.time()
   summary_=ME.make_splitting(summary,vocab)
   end=time.time()
   temps.append((end-start)/60)
   pickle.dump(summary_,open('summary_word_'+name+'.pickle','wb'))
   print("Summary :",temps[-1])

   start=time.time()
   text_=Parallel(n_jobs=cpu)(delayed(ME.make_splitting)(s) for s in text)
   end=time.time()
   pickle.dump(text_,open('text_word_'+name+'.pickle','wb'))
   temps.append((end-start)/60)
   print("Texte :",temps[-1])
#%%
summaries_train=[i for i in os.listdir() if ('summary_word' in i) and ('test' not in i)]
texts_train=[i for i in os.listdir() if ('text_word' in i) and  ('test' not in i)]

n=2
for summary,text in zip(summaries_train,texts_train):
        name='train_'+summary.split('.')[0].split('_')[-1]
        print(name)
        text=pickle.load(open(text,'rb'))
        summary=pickle.load(open(summary,'rb'))
        score_final=[]
        erreur_f=[]
        for l in range(n):
            l_1=int(len(text)*(l)/2)
            l_2=int(len(text)*(l+1)/2)
            print(l_1,l_2)
            text_=text[l_1:l_2]
            summary_=summary[l_1:l_2]
            print("Début de la création de l'output :")
            output_time=time()
            ME=Make_Extractive(cpu=cpu,fenetre=20,minimum=1,d=100,epochs=25)
            score,text_2,summary_2,erreur=ME.make_output(text_,summary_,W2V=W2V)
            output_end=time()
            print("Durée de la création d'output :",round((output_end-output_time)/60,2),"minutes.") #42 minutes pour 15,8K lignes
            print("Nombre d'erreurs pendant l'encodage du test :",round(len(erreur)/len(text_)*100,2),"%")
            score_final+=score
            erreur_f+=erreur
            pickle.dump(score,open('score_'+name+'_'+str(l+1)+'.pickle','wb'))

        pickle.dump(score_final,open('score_'+name+'.pickle','wb'))
        print(score_final)
        #pickle.dump(W2V,open('W2V_'+name+'.pickle','wb'))
        #pickle.dump(text_2,open('text_word_'+name+'.pickle','wb'))
        #pickle.dump(summary_2,open('summary_word_'+name+'.pickle','wb'))
        pickle.dump(erreur_f,open('erreur_word_'+name+'.pickle','wb'))

#%%
for summary,text in zip(summaries_train[:1],texts_train[:1]):
   name='train_'+summary.split('.')[0].split('_')[-1]
   print(name)
   text=pickle.load(open(text,'rb'))
   summary=pickle.load(open(summary,'rb'))

   print("Début de la création de l'output :")
   output_time=time.time()
   score,text_2,summary_2,erreur=ME.make_output(text,summary,W2V)
   output_end=time.time()
   print("Durée de la création d'output :",round((output_end-output_time)/60,2),"minutes.")

   pickle.dump(score,open('score_'+name+'.pickle','wb'))
   pickle.dump(text_2,open('text_word_'+name+'.pickle','wb'))
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
