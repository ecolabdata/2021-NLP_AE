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
from transformers.utils.dummy_pt_objects import CamembertModel
from transformers.utils.dummy_sentencepiece_objects import CamembertTokenizer
from unidecode import unidecode
from bs4 import BeautifulSoup
import gensim

from fats import BERTScore, Make_Embedding,TextRank,Random_summary,Lead_3,SMHA_Linear_classifier
from fats import Make_Extractive

os.chdir("C:\\Users\\theo.roudil-valentin\\Documents\\Resume\\MLSUM")
#On va entrainer un nouveau Word2Vec qui comprendra les phrases 
# des données de validation en même temps que celles de l'entraînement

##############################################################################################################
##### 1. Ré-entraînement d'un Word2Vec pour le fichier test #####################################################
##############################################################################################################

# # PAS BESOIN DE REFAIRE TOURNER
# summaries=[i for i in os.listdir() if ('summary_word' in i)]
# # summary=pickle.load(open(summaries[0],'rb'))
# texts=[i for i in os.listdir() if ('text_word' in i)]# and ('test' in i)
# # text=pickle.load(open(texts[0],'rb'))
# print(summaries,texts)

# start=time.time()
# text=pickle.load(open(texts[0],'rb'))
# summary=pickle.load(open(summaries[0],'rb'))
# assert len(text)==len(summary)
# text=functools.reduce(operator.iconcat, text, [])
# summary=functools.reduce(operator.iconcat, summary, [])

# for i in range(1,len(summaries)):
#    text+=functools.reduce(
#       operator.iconcat, 
#       pickle.load(open(texts[i],'rb')), [])
#    summary+=functools.reduce(
#       operator.iconcat, 
#       pickle.load(open(summaries[i],'rb')), [])
# end_1=time.time()
# print(round((end_1-start)/60,2),"minutes.")
# dim=100
# fenetre=20
# minimum=1
# epo=5
# sentence=text.copy()

# import gensim
# try:
#    W2V=gensim.models.Word2Vec(size=dim,window=fenetre,min_count=minimum)
# except:
#    W2V=gensim.models.Word2Vec(vector_size=dim,window=fenetre,min_count=minimum)
# W2V.build_vocab(sentence)

# print("Démarrage de l'entraînement du modèle Word2Vec.")
# start=time.time()
# W2V.train(sentence,total_examples=W2V.corpus_count,epochs=epo)
# end=time.time()
# print("Le modèle W2V est désormais entraîné et cela a pris :",round((end-start)/60,2),"minutes.")
# W2V.save("W2V_all.model")

##############################################################################################################
##### 2. Adaptation du texte test pour tenseur et comparaison #####################################################
##############################################################################################################


# W2V=gensim.models.Word2Vec.load("W2V_all.model")
# summaries=[i for i in os.listdir("test/") if ('summary_word' in i) and ('test' in i)]
# texts=[i for i in os.listdir("test/") if ('text_word' in i) and ('test' in i)]
# print(summaries,texts)

# start=time.time()
# text=pickle.load(open("test/"+texts[0],'rb'))#"test/text_2.pickle",'rb'))
# summary=pickle.load(open("test/"+summaries[0],'rb'))#"test/summary_2.pickle",'rb'))
# assert len(text)==len(summary)

#D'abord, on va transformer le texte pour l'avoir dans le bon format pour le DL
# ME=Make_Extractive(cpu=psutil.cpu_count())

#########  Pas besoin de refaire tourner ce qui est commenté, une fois suffit
# for l in range(2):
#    name='train_'+summaries[0].split('.')[0].split('_')[-1]
#    print(name)
#    score_final=[]
#    erreur_f=[]
#    print("Début de la création de l'output :")
#    output_time=time.time()
#    score,text_2,summary_2,erreur=ME.make_output(text[int(len(text)*(l/2)):int(len(text)*((l+1)/2))],
#    summary[int(len(text)*(l/2)):int(len(text)*((l+1)/2))],W2V=W2V,verbose=1)
#    output_end=time.time()
#    pickle.dump(score,open("test/score_"+str(l)+".pickle",'wb'))
#    pickle.dump(erreur,open("test/erreur_"+str(l)+".pickle",'wb'))

#    print("Durée de la création d'output :",round((output_end-output_time)/60,2),"minutes.") #42 minutes pour 15,8K lignes
#    print("Nombre d'erreurs pendant l'encodage du test :",round(len(erreur)/len(text[:10])*100,2),"%")

# score_0=pickle.load(open("test/score_0.pickle",'rb'))
# score_1=pickle.load(open("test/score_1.pickle",'rb'))
# score=score_0+score_1
# del score_0, score_1

# erreur_0=pickle.load(open("test/erreur_0.pickle",'rb'))
# erreur_1=pickle.load(open("test/erreur_1.pickle",'rb'))
# erreur=erreur_0+erreur_1
# del erreur_0, erreur_1

# pickle.dump(score,open("test/score.pickle",'wb'))
# pickle.dump(erreur,open("test/erreur.pickle",'wb'))
# os.remove("test/score_0.pickle")
# os.remove("test/score_1.pickle")
# os.remove("test/erreur_0.pickle")
# os.remove("test/erreur_1.pickle")

# score=pickle.load(open("test/score.pickle",'rb'))
# erreur=pickle.load(open("test/erreur.pickle",'rb'))


# summaries=[i for i in os.listdir("test/") if ('summary_clean' in i) and ('test' in i)]
# texts=[i for i in os.listdir("test/") if ('text_clean' in i) and ('test' in i)]
# print(summaries,texts)

# text_clean=pickle.load(open("test/"+texts[0],'rb'))
# summary_clean=pickle.load(open("test/"+summaries[0],'rb'))
# assert len(text_clean)==len(summary_clean)

# On enlève les phrases dans les paragraphes qui n'ont pas de scores associés
# Autrement dit, lors de la création du score, ces phrases ont posé problème
# On les enlève donc des paragraphes

# for i in erreur:
#    print(i)
#    try:
#       text_clean[i[0]].remove(text_clean[i[0]][i[1]])
#    except:
#       continue
   
# len(score[0]),len(text[0]),len(text_clean[0])

# On crée le dictionnaire contenant tout les tenseurs pour l'entraînement DL (Deep Learning)
# ME=Make_Extractive(cpu=psutil.cpu_count())
# start=time.time()
# dico_test=ME.make_encoding(text_clean,score,tokenizer='MLSUM_tokenizer.model',prefix="test",name="Tokenizer_test",split=1)
# end=time.time()
# print(round((end-start)/60,2),"minutes")
# len(dico_test['input'])
# dico_test['trace']


# A partir d'ici on va re-nettoyer pour avoir quelque chose de comparable avec
# les deux familles de modèles : ceux relevant du DL et les autres (BertScore, TextRank etc...)
# Pour cela, on va devoir découper les paragraphes selon le découpage qui a été fait
# pour créer les tenseurs pour les modèles DL

# On compte le nombre de phrases présentes dans chaque paragraphe
# nombre_phrase_text_clean=[len(i) for i in text_clean] 
# On compte le nombre de phrases présentes dans chaque tenseur
# nombre_phrase_input=[i.count(5) for i in dico_test['input']]
# On compare les dix premiers pour voir 
# print(nombre_phrase_text_clean[:10],'\n',nombre_phrase_input[:10])

#On crée un nouvel index pour ne pas prendre en compte les scores vides (problème dans l'encodage)
# index_1=[i for i in range(len(score)) if (score[i].shape!=torch.Size([0]))]
# On enlève de l'index les tenseurs qui n'ont pas de phrases (non encodages encore)
# index_2=[i for i in index_1 if (dico_test['trace'][i]>0)]
# check=[i for i in range(len(score)) if i not in index_2]
# On regarde si cela ne représente pas trop de paragraphes
# print(len(check),"paragraphes n'ont pas de scores, soit",round((len(check)/len(score))*100,2),"%")

# On prend les paragraphes sur lesquels on va vraiment travailler
# text_clean_prime=[text_clean[i] for i in index_2]
# Nombre de phrases dans les paragraphes nettoyés 
# nombre_phrase_text_clean_prime=[len(i) for i in text_clean_prime]
# trace=[(i,dico_test['trace'][i]) for i in range(len(text_clean_prime)) if dico_test['trace'][i]>0]
# Première trace
# trace=[i for i in dico_test['trace'] if i>0]

# Redécoupage des paragraphes pour mettre au même niveau de comparaison
# P=[nombre_phrase_input[0:trace[0]]]
# print(P)
# z=0
# for i in range(1,len(trace)):
#    z+=len(P[-1])
#    # print(z)
#    P.append(nombre_phrase_input[z:(z+trace[i])])

# Dernière trace, chaque élément contient :
# (numéro du paragraphe, 
# nombre de tenseur pour le paragraphe, 
# nombre de phrases dans chaque tenseur)
# trace_finale=[(i,trace[i],P[i]) for i in range(len(index_2))]
# pickle.dump(trace_finale,open('test/trace_test.pickle','wb'))

# def make_new_paragraphes(tcp,trace):
#    paragraphe=[]
#    paragraphe.append(tcp[:trace[-1][0]])
#    if len(trace[-1])>1:
#       for i in range(1,len(trace[-1])):
#          paragraphe.append(tcp[trace[-1][i-1]:(np.sum(trace[-1][:i])+trace[-1][i])])
#    return paragraphe
# # Paragraphes=make_new_paragraphes(text_clean_prime[0],trace_finale[0])

# Paragraphes=Parallel(psutil.cpu_count())(delayed(make_new_paragraphes)(i,j) for i,j in zip(text_clean_prime,trace_finale))
# pickle.dump(Paragraphes,open('test/Paragraphes.pickle','wb'))

##############################################################################################################
##### 3. Métriques des modèles non-neuronaux #################################################################
##############################################################################################################

Paragraphes=pickle.load(open('test/Paragraphes.pickle','rb'))
Paragraphes=functools.reduce(operator.iconcat, Paragraphes, [])

from fats import Make_Embedding
from transformers import CamembertTokenizer,CamembertModel
tok=CamembertTokenizer('MLSUM_tokenizer.model')
camem=CamembertModel.from_pretrained("camembert-base")

nphrase=2
TR=TextRank(tok_path='MLSUM_tokenizer.model')
BS=BERTScore('MLSUM_tokenizer.model')

W2V=gensim.models.Word2Vec.load("W2V_all.model")
TRB=partial(TR.make_resume,type='bert',modele=camem,k=nphrase,get_score_only=True)
TRW=partial(TR.make_resume,type='word2vec',modele=W2V,k=nphrase,get_score_only=True)
BSR=BS.make_score
L3=partial(Lead_3,k=nphrase)
RS=partial(Random_summary,get_index_only=True)


taille=100
start=[]
end=[]

# ME=Make_Embedding(tok)
# dico=ME.make_tokens(Paragraphes[:5],1)
# dico.keys()
# #%%
# def emb_phrase(input_id,att_mask,cam):
#    embeddings=[]
#    for i,a in zip(input_id,att_mask):
#       try:
#          embedding=cam(torch.tensor(i).squeeze(1),torch.tensor(a).squeeze(1))
#          embeddings.append(embedding.last_hidden_state.mean(dim=1))
#       except:
#          embedding=cam(torch.tensor(i).squeeze(0),torch.tensor(a).squeeze(0))
#          embeddings.append(embedding.last_hidden_state.mean(dim=1))
#          #embeddings.append(embedding[0].mean(dim=0).squeeze(0))
#    return embeddings
# ouais=emb_phrase(dico['input_ids'][:2],dico['attention_mask'][:2],camem)
# len(ouais),ouais[1].size()
# #%%
# def emb_phrases(input_ids,att_masks,cam):
#    embedding=[]
#    for input_id,att_mask in zip(input_ids,att_masks):
#       embeddings=emb_phrase(input_id,att_mask,cam)
#       embedding.append(embeddings)
#    return embedding
# ouais=emb_phrases(dico['input_ids'],dico['attention_mask'],camem)
# #%%
# ME=Make_Embedding(tok)
# dico=ME.make_tokens(Paragraphes[0],2)
# #%%
# i,a=ME.make_token(Paragraphes[0],1)
# #%%
# input_ids=dico['input_ids']
# att_mask=dico['attention_mask']
# embedding=camem(torch.tensor(input_ids).squeeze(1),torch.tensor(att_mask).squeeze(1))
# #%%

# embeddings=ME.emb_phrase(input_ids,att_mask,camem)

# #%%
# ME=Make_Embedding(tok)

# def make_embedding_bert(articles,camem):
#    if type(articles[0])==str:
#       input_ids,att_mask=ME.make_token(articles,2)
#       embeddings=camem(torch.tensor(input_ids),
#             torch.tensor(att_mask))
#       return embeddings
#    else:
#       dico=ME.make_tokens(articles,2)
#       input_ids=dico['input_ids']
#       att_mask=dico['attention_mask']
#       embeddings=ME.emb_phrase(input_ids,att_mask,camem)
#       return embeddings,dico
# b,d=TR.make_embedding_bert(Paragraphes[0],camem)
# #%%
# ouais=camem(torch.tensor(dico['input_ids']).squeeze(0),
# torch.tensor(dico['attention_mask']).squeeze(0))
# #%%

# def make_resume(article,modele,k=3,verbose=1,get_score=False,get_score_only=False):
#    b,d=TR.make_embedding_bert(article,modele)
#    mb=[TR.mat_sim(h) for h in b]
#    sb=[TR.scores(m,k=k) for m in mb]
#    if get_score:
#       resume=[[article[i][k] for k in sb[i]] for i in range(len(sb))]
#       return resume, sb
#    elif get_score_only:
#       return sb
#    else:
#       resume=[[article[i][k] for k in sb[i]] for i in range(len(sb))]
#       return resume

# resume=make_resume(Paragraphes[:2],modele=camem)
# resume
#%%
# start.append(time.time())
# TRB_sortie=Parallel(8)(delayed(TRB)(i) for i in tqdm(Paragraphes[:taille]))
# # TRB_sortie=TRB(Paragraphes[0])
# pickle.dump(TRB_sortie,open('test/TRB_sortie.pickle','wb'))
# end.append(time.time())
# print("TRB :",round((end[-1]-start[-1])/60,2),"minutes")

# TRB_sortie_2=[] #pickle.load(open('TRB_sortie.pickle','rb'))
# longueur=[]
# for i in range(len(TRB_sortie_2),len(Paragraphes)):
#     longueur.append(len(Paragraphes[i]))
#     start.append(time.time())
#     TRB_sortie_2.append(TRB(Paragraphes[i]))
#     end.append(time.time())
#     if i%250==0:
#         print("Achevé :",round((i/len(Paragraphes))*100,2),"%")
#         pickle.dump(TRB_sortie_2,open("TRB_sortie.pickle",'wb'))




rendu=pickle.load(open('TRB_sortie.pickle','rb'))#[] if 
print(len(rendu))
import fats
pas=1

for i in tqdm(range(len(rendu),int(len(Paragraphes)/(pas+1)))):
    resu=fats.Resume(Paragraphes[(i*2):(i+pas)*2],
                     DL=False,
                     cpu=1,
                     type_='TextRankBert',k=2,modele=camem,
                     tok="MLSUM_tokenizer.model",get_score_only=True)
    rendu.append(resu)
               
    if i%100==0:
        pickle.dump(rendu,open('TRB_sortie.pickle','wb'))
#%%
TRW_sortie=pickle.load(open('TRW_sortie.pickle','rb'))
len(TRW_sortie)

start.append(time.time())
for i in range(len(TRW_sortie),len(Paragraphes)):
   TRW_sortie.append(TRW(Paragraphes[i]))
   if i%250==0:
      print("Achevé :",round((i/len(Paragraphes))*100,2),"%")
      pickle.dump(TRW_sortie,open('TRW_sortie.pickle','wb'))


# TRW_sortie=Parallel(psutil.cpu_count())(delayed(TRW)(i) for i in tqdm(Paragraphes))
pickle.dump(TRW_sortie,open('test/TRW_sortie.pickle','wb'))
end.append(time.time())
print("TRW :",round((end[-1]-start[-1])/60,2),"minutes")
#%%
# b,_=TR.make_embedding_bert(Paragraphes[0],camem)
# # %%
# VSA=b.mean(dim=0)
# cosim=torch.nn.CosineSimilarity(dim=-1)
# cosim(VSA,b)
#%%
# BSR_sortie=[]
# start.append(time.time())
# for i in tqdm(range(len(Paragraphes))):
#    ouais=BSR(Paragraphes[i])
#    BSR_sortie.append(ouais)
#    if i%100==0:
#       pickle.dump(BSR_sortie,open('test/BSR_sortie.pickle','wb'))

# # BSR_sortie=Parallel(psutil.cpu_count())(delayed(BSR)(i) for i in Paragraphes)
# end.append(time.time())
# print("BSR :",round((end[-1]-start[-1])/60,2),"minutes")

import fats
rendu=pickle.load(open('BSR_sortie.pickle','rb'))#[] if 
print(len(rendu))
ms=fats.BERTScore(tok="MLSUM_tokenizer.model",cpu=4).make_score
# ms(Paragraphes[torch.randint(len(Paragraphes),[1])])
#%%
#pas=1
#s
for i in tqdm(range(len(rendu),len(Paragraphes))):
    resu=ms(Paragraphes[i],s=50)
    #fats.Resume(Paragraphes[(i*2):(i+pas)*2],DL=False,cpu=1,type_='TextRankBert',k=2,modele=camem,tok="MLSUM_tokenizer.model",get_score_only=True)
    rendu.append(resu)
               
    if i%50==0:
        pickle.dump(rendu,open('BSR_sortie.pickle','wb'))
#%%
##############################################################################################################
######    Petite étude du temps d'exécution     ##########################################################################################
###############################################################################################################
TR=TR=TextRank(tok_path='MLSUM_tokenizer.model',cpu=4)
i=torch.randint(len(Paragraphes),[1])
TR.make_embedding_bert(Paragraphes[i],camem=camem)
ms(Paragraphes[i])

ME=Make_Embedding(tok=CamembertTokenizer('MLSUM_tokenizer.model'),cpu=5)
input,att=ME.make_token(Paragraphes[i],5)

s=time.time()
camem(torch.tensor(input),torch.tensor(att))
e=time.time()
print((e-s)/torch.tensor(input).size()[0])

s=time.time()
camem(torch.tensor(input)[:2],torch.tensor(att)[:2])
e=time.time()
print((e-s)/2)
#%%
start.append(time.time())
L3_sortie=[[1,2,3] for i in Paragraphes]
pickle.dump(L3_sortie,open('test/L3_sortie.pickle','wb'))
end.append(time.time())
print("L3 :",round((end[-1]-start[-1])/60,2),"minutes")
#%%
start.append(time.time())
RS_sortie=Parallel(psutil.cpu_count())(delayed(RS)(i) for i in Paragraphes[:taille])
pickle.dump(RS_sortie,open('test/RS_sortie.pickle','wb'))
end.append(time.time())
print("RS :",round((end[-1]-start[-1])/60,2),"minutes")
#%%
TRB_sortie=pickle.load(open('test/TRB_sortie.pickle','rb'))


#%%
##############################################################################################################
##### 4. Métriques des modèles neuronaux #####################################################################
##############################################################################################################

DL_model=SMHA_Linear_classifier(torch.Size([512,768]),8,768)
path='SMHA_Linear_classifier.pt'
DL_model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

#%%




