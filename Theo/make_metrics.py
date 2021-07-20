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
#%%
from fats import BERTScore,TextRank,Random_summary,Lead_3,SMHA_Linear_classifier
from fats import Make_Extractive

os.chdir("C:\\Users\\theo.roudil-valentin\\Documents\\Resume\\MLSUM")
#On va entrainer un nouveau Word2Vec qui comprendra les phrases 
# des données de validation en même temps que celles de l'entraînement
#%%
# PAS BESOIN DE REFAIRE TOURNER
summaries=[i for i in os.listdir() if ('summary_word' in i)]
# summary=pickle.load(open(summaries[0],'rb'))
texts=[i for i in os.listdir() if ('text_word' in i)]# and ('test' in i)
# text=pickle.load(open(texts[0],'rb'))
print(summaries,texts)

start=time.time()
text=pickle.load(open(texts[0],'rb'))
summary=pickle.load(open(summaries[0],'rb'))
assert len(text)==len(summary)
text=functools.reduce(operator.iconcat, text, [])
summary=functools.reduce(operator.iconcat, summary, [])

for i in range(1,len(summaries)):
   text+=functools.reduce(
      operator.iconcat, 
      pickle.load(open(texts[i],'rb')), [])
   summary+=functools.reduce(
      operator.iconcat, 
      pickle.load(open(summaries[i],'rb')), [])
end_1=time.time()
print(round((end_1-start)/60,2),"minutes.")
dim=100
fenetre=20
minimum=1
epo=5
sentence=text.copy()

import gensim
try:
   W2V=gensim.models.Word2Vec(size=dim,window=fenetre,min_count=minimum)
except:
   W2V=gensim.models.Word2Vec(vector_size=dim,window=fenetre,min_count=minimum)
W2V.build_vocab(sentence)

print("Démarrage de l'entraînement du modèle Word2Vec.")
start=time.time()
W2V.train(sentence,total_examples=W2V.corpus_count,epochs=epo)
end=time.time()
print("Le modèle W2V est désormais entraîné et cela a pris :",round((end-start)/60,2),"minutes.")
W2V.save("W2V_all.model")
#%%
W2V=gensim.models.Word2Vec.load("W2V_all.model")
summaries=[i for i in os.listdir("test/") if ('summary_word' in i) and ('test' in i)]
texts=[i for i in os.listdir("test/") if ('text_word' in i) and ('test' in i)]
print(summaries,texts)

start=time.time()
text=pickle.load(open("test/"+texts[0],'rb'))#"test/text_2.pickle",'rb'))
summary=pickle.load(open("test/"+summaries[0],'rb'))#"test/summary_2.pickle",'rb'))
assert len(text)==len(summary)

# text=functools.reduce(operator.iconcat, text, [])
# len(text),len(text[0]),len(text[0][0]),len(text[0][0][0]),len(text[0][0][0][0])
# summary=functools.reduce(operator.iconcat, summary, [])

# # PROBLEME DE VOCABULAIRE WORD2VEC
# cosim=torch.nn.CosineSimilarity(-1)
# import gensim
# gensim.__version__
# score=[]
# verbose=0
# erreur=[]
# for sent in tqdm(text[:100]):
#             score_=[]
#             for s in sent:
#                try:
#                   score_.append(
#                         cosim(torch.as_tensor(W2V.wv[s]),torch.as_tensor(W2V.wv[summary[text.index(sent)]]))
#                         .mean())
#                except Exception as e:
#                   print(e)
#                   break
#                   erreur.append([text.index(sent),sent.index(s)])
#                   if verbose==1:
#                      print("Attention, l'élément",erreur[-1],"n'a pas pu être encodé.")
#                   continue
#             try:
#                score.append(torch.stack(score_))
#             except:
#                score.append(torch.Tensor())


#D'abord, on va transformer le texte pour l'avoir dans le bon format pour le DL
ME=Make_Extractive(cpu=psutil.cpu_count())
# score_=[]
# text_2_=[]
# summary_2_=[]
# erreur_=[]
#%%
for l in range(1,2):
   name='train_'+summaries[0].split('.')[0].split('_')[-1]
   print(name)
   score_final=[]
   erreur_f=[]
   print("Début de la création de l'output :")
   output_time=time.time()
   score,text_2,summary_2,erreur=ME.make_output(text[int(len(text)*(l/2)):int(len(text)*((l+1)/2))],
   summary[int(len(text)*(l/2)):int(len(text)*((l+1)/2))],W2V=W2V,verbose=1)
   output_end=time.time()
   # score_.append(score)
   # text_2_.append(text_2)
   # summary_2_.append(summary_2)
   # erreur_.append(erreur)
   pickle.dump(score,open("test/score_"+str(l)+".pickle",'wb'))
   # pickle.dump(text_2,open("test/text_2.pickle",'wb'))
   # pickle.dump(summary_2,open("test/summary_2.pickle",'wb'))
   pickle.dump(erreur,open("test/erreur_"+str(l)+".pickle",'wb'))

   print("Durée de la création d'output :",round((output_end-output_time)/60,2),"minutes.") #42 minutes pour 15,8K lignes
   print("Nombre d'erreurs pendant l'encodage du test :",round(len(erreur)/len(text[:10])*100,2),"%")

# score=score_[0]+score_[1]
# text_2=text_2_[0]+text_2_[1]
# summary_2=summary_2_[0]+summary_2_[1]
# erreur=erreur_[0]+erreur_[1]

# pickle.dump(score,open("test/score.pickle",'wb'))
# pickle.dump(text_2,open("test/text_2.pickle",'wb'))
# pickle.dump(summary_2,open("test/summary_2.pickle",'wb'))
# pickle.dump(erreur,open("test/erreur.pickle",'wb'))

# score_=pickle.load(open("test/score.pickle",'rb'))
# text_2_=pickle.load(open("test/text_2.pickle",'rb'))
# summary_2_=pickle.load(open("test/summary_2.pickle",'rb'))
# erreur_=pickle.load(open("test/erreur.pickle",'rb'))
# score_=pickle.load(open("test/score.pickle",'rb'))
# text_=pickle.load(open("test/text_2.pickle",'rb'))
# summary_=pickle.load(open("test/summary_2.pickle",'rb'))
# erreur_=pickle.load(open("test/erreur.pickle",'rb'))

# score_2=score_+score
# print(len(score_2))
# text_2_2=text_+text_2
# print(len(text_2_2))
# summary_2_2=summary_+summary_2
# print(len(summary_2_2))
# erreur_2=erreur_+erreur
# print(len(erreur_2))

# pickle.dump(score_2,open("test/score.pickle",'wb'))
# pickle.dump(text_2_2,open("test/text_2.pickle",'wb'))
# pickle.dump(summary_2_2,open("test/summary_2.pickle",'wb'))
# pickle.dump(erreur_2,open("test/erreur.pickle",'wb'))
#%%
summaries=[i for i in os.listdir("test/") if ('summary_clean' in i) and ('test' in i)]
texts=[i for i in os.listdir("test/") if ('text_clean' in i) and ('test' in i)]
print(summaries,texts)

text_clean=pickle.load(open("test/"+texts[0],'rb'))
summary_clean=pickle.load(open("test/"+summaries[0],'rb'))
assert len(text_clean)==len(summary_clean)

score=pickle.load(open("test/score_0.pickle",'rb'))
# pickle.load(open("test/text_2.pickle",'rb'))
# pickle.load(open("test/summary_2.pickle",'rb'))
erreur=pickle.load(open("test/erreur_0.pickle",'rb'))

#%%
for i in erreur:
   print(i)
   try:
      text_clean[i[0]].remove(text_clean[i[0]][i[1]])
   except:
      continue
   
len(score[0]),len(text[0]),len(text_clean[0])

#%%
ME=Make_Extractive(cpu=psutil.cpu_count())
dico_test=ME.make_encoding(text_clean[:7914],score,tokenizer='MLSUM_tokenizer.model',prefix="test",name="Tokenizer_test",split=1)

len(dico_test['input'])
#%%
nombre_phrase_text_clean=[len(i) for i in text_clean]
nombre_phrase_input=[i.count(5) for i in dico_test['input']]
nombre_phrase_text_clean[:10],nombre_phrase_input[:10]
#%%
npi=torch.Tensor(nombre_phrase_input)
trace=[]
x_1=0
x=0
for k in tqdm(range(len(nombre_phrase_text_clean))):
   while npi[x_1:x].sum()!=nombre_phrase_text_clean[k]:
      x+=1
   trace.append([k,x])
   x_1=trace[k][1]
#%%
nphrase=2
TR=TextRank()
BS=BERTScore()

DL_model=SMHA_Linear_classifier(torch.Size([512,768]),8,768)
path='SMHA_Linear_classifier.pt'
DL_model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
#%%
dataloader_2=pickle.load(open('train_loader_2.pickle','rb'))
print("Nombre de batchs :",len(dataloader_2))
for _,batch in enumerate(dataloader_2):
    print("Taille du batch :",len(batch[0]))
    break
batch

#%%
W2V=pickle.load(open("W2V_train.pickle","rb"))
TRB=TR.make_resume(exemple,'bert',k=nphrase)
TRW=TR.make_resume(exemple,'word2vec',W2V=W2V,k=nphrase)
BSR=BS.make_summary(exemple,k=nphrase)
L3=Lead_3(exemple,k=2)
RS=Random_summary(exemple)


#%%




