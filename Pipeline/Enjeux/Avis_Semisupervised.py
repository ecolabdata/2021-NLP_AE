#%%

import pandas as pd, numpy as np
import pickle
import unidecode
import spacy
from nltk.corpus import stopwords
import re
from Pipeline.Enjeux.utils import *
import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct
from Pipeline.Enjeux.utils import *
stop_words = stopwords.words('french')
stop_words.extend(['avis','environnement','autorite','projet','etude','exploitation','impact','site','dossier','mission','regionale','mrae','mnhn'])
from Pipeline.Enjeux.bagging import CorExBoosted
#Charger thésau et data et vecto data

docs_df = pickle.load(open("Data\Workinprogress\docs_df.pickle",'rb'))
Thesaurus = pickle.load(open('Data\Thesaurus_csv\Thesaurus1_clean.pickle','rb'))
# ## 2. Topic modeling semi-supervisé
#%%



#%%
instance = CorExBoosted(docs_df,Thesaurus).encode()

#%%
instance.fit(n_classif=10,strength=2)
#%%

obj = instance.optimize_weights(method = 'SLSQP')
prediction = instance.predict(instance.X,weights=obj.x)
sc2 = evaluate(docs_df,prediction,returnscore=True)

#%%

obj2 = instance.optimize_selectivity(bnds=(0.1,0.9))
prediction = instance.predict(instance.X,selectivity=obj2.x)
sc2 = evaluate(docs_df,prediction,returnscore=True)
#%%
prediction = instance.predict(instance.X)
sc1 = evaluate(docs_df,prediction,returnscore=True)
#delta(sc1,sc2,returnmoy=True)

#%%
#On va utiliser ce modèle comme modèle initial (vérité
# approximative pour raffiner le tout)
k = 2
topic_model = ct.Corex(n_hidden=len(enjeux_list))
topic_model.fit(instance.X, words=instance.vocab, anchors=instance.thesau_list, anchor_strength=k)
mat = topic_model.labels
sc2 = evaluate(docs_df,mat,returnscore=True)
delta(sc1,sc2,returnmoy=True)
#%%
#On va essayer d'optimiser en faisant du stratified sampling
y_true,X_sub,y_pred = separate(instance.docs,instance.X,prediction = instance.predict(instance.X))

from sklearn.metrics import label_ranking_loss

label_ranking_loss(y_true,y_pred)


#%%
sc2 = evaluate(docs_df,test,returnscore=True)

delta(sc1,sc2,returnmoy=True)

#%%
model = ct.Corex(n_hidden=len(enjeux_list))
model.fit(np.matrix(X_res), words=vocab, anchors=thesau_list, anchor_strength=k+1)

#Prédiction et évaluation sur sur toutes les données
test = model.predict(X)

sc3 = evaluate(docs_df,test,returnscore=True)

delta(sc1,sc3,returnmoy=True)

#%%


#%%


test = predict(storage_fitted,X)

sc4 = evaluate(docs_df,test,returnscore=True)

am = [0,0,0,0]
delta_classifmoy = delta(sc1,sc4,returnmoy=True)
for classif in storage_fitted:
    sc = evaluate(docs_df,classif.predict(X),returnscore=True,showgrid=False)
    d =  delta(sc4,sc,returnmoy=True,showgrid=False)
    am = vadd(am,d)
am = np.array(am)/20

#Le bagging surpasse de 6% en moyenne les modèles individuels

#%%
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(base_estimator=ct.Corex(n_hidden=len(enjeux_list)))
clf.fit(X_res,y_res)
#%%
#Tests pour avoir le meilleur anchor_strengh
topic_model_strat = {}
from os import cpu_count
from joblib import Parallel,delayed
from functools import partial

for j in range(10):
    print(j)
    topic_model_strat[j] = ct.Corex(n_hidden=len(enjeux_list))
    topic_model_strat[j].fit(np.matrix(X_sub), words=vocab, anchors=thesau_list, anchor_strength=j+1)

#%%
grid_scores = {}
for j in range(10):
    print(j)
    pred = topic_model_strat[j].predict(X)
    grid_scores[j] = evaluate(docs_df,pred,returnscore=True)

#%%


def sample_add(X,sc1,X_sub2,vocab,anchor,strength,enjeux_list,num_average = 20):
    """
    Pour ajouter des samples sur la base du delta des scores.
    Renvoie True si le F1 est amélioré.
    """
    deltamoy = [0,0,0,0]
    for it in range(num_average):
        model2 = ct.Corex(n_hidden=len(enjeux_list))
        model2.fit(X_sub2, words=vocab, anchors=anchor, anchor_strength=strength)
        #Prédiction et évaluation sur sur toutes les données
        test = model2.predict(X)
        sc2 = evaluate(test,returnscore=True)

        #Calcul du delta des évaluations
        deltamoy = vadd(deltamoy,delta(sc1,sc2,returnmoy=True))
    deltamoy = np.array(deltamoy)/num_average
    if deltamoy[3]>0:
        return(True,sc2)
    else:
        return(False,sc1)

#%%

#tentative pour rajotuer des samples de tail pour
#enrichir l'ensemble d'apprentissage :
#très long a faire tourner et ni vraiment efficace
#ni vraiment opti

X_l = X.tolist()
y = test
y_df_full = pd.DataFrame(y,columns=enjeux_list)
X_sub_mat = X_sub.to_numpy().tolist()
tail = get_tail_label(pd.DataFrame(test,columns=enjeux_list))
sc1 = sc2

#On essaye de rajouter assez naïvement les lignes
#contenant des labels de tail pour équilibrer
for rowX,rowy in zip(X_l,y_df_full.iterrows()):
    if rowX in X_sub_mat:
        pass
    else:
        tail = get_tail_label(y_sub)
        label_in_tail = False
        cols = y_df_full.columns
        for col in cols:
            if rowy[1][col] == True and col in tail:
                label_in_tail = True
                break
        if label_in_tail:
            rowdf = pd.DataFrame([rowX],columns=vocab)
            X_sub2 = X_sub.append(rowdf)
            trueVal, sc1 = sample_add(X,sc1,
            np.matrix(X_sub2.to_numpy()),
            vocab,thesau_list,2,enjeux_list,num_average=1)
            if trueVal:
                X_sub = X_sub2
                y_sub = y_sub.append(rowy[1].map({False:0,True:1}),ignore_index=True)


# %%
#On va enrichir le thésaurus automatiquement en ajoutant les mots
#sémantiquement (word2vec) reliés aux mots de notre thésaurus
#On évaluera (auditeurs) l'interprétabilité des nouvelles listes

from Pipeline.Enjeux.enrichissement import *

inst = makesimilarity(dico_thesau,cosimilarite=pickle.load(open("Data\Workinprogress\cosimilarite_avis.pickle",'rb')))
add = inst.top_words_topic(100)

from Pipeline.Enjeux.processing_encoding import processing_mot

#%%
idx = {}
for k in range(len(enjeux_list)):
    idx[enjeux_list[k]] = k
worst_enj = 'Gestion des déchets'
keep = []

new_thesau = thesau_list.copy()
for k in range(100):
    new_thesau = thesau_list.copy()
    corr = processing_mot(add[worst_enj][k])
    i = idx[worst_enj]
    print(corr)
    if corr not in vocab:
        print('On passe')
        continue
    new_thesau[i].append(corr)
    print('Enrichissement',worst_enj)

    model = ct.Corex(n_hidden=len(enjeux_list))
    model.fit(np.matrix(X), words=vocab, anchors=new_thesau, anchor_strength=5)

    #Prédiction et évaluation sur sur toutes les données
    test = model.predict(X)
    sc2 = evaluate(docs_df,test,returnscore=True)
    d =delta(sc1,sc2,returnmoy=True)
    
    if sc2[worst_enj][3]-sc1[worst_enj][3]>0:
        keep.append(corr)

#%%
worst_enj = 'Gestion des déchets'


new_thesau = thesau_list.copy()
for k in range(1,100):
    new_thesau = thesau_list.copy()
    print('\n')
    print('...................',k)
    i= 0
    for enjeu,thesau in zip(add,thesau_list):
        print(enjeu)
        if enjeu == worst_enj:
            
            corr = processing_mot(add[enjeu][k])
            print(corr)
            if corr not in vocab:
                print('On passe')
                pass
            new_thesau[i].append(corr)
        i+=1
    model = ct.Corex(n_hidden=len(enjeux_list))
    model.fit(np.matrix(X), words=vocab, anchors=new_thesau, anchor_strength=3)

    #Prédiction et évaluation sur sur toutes les données
    test = model.predict(X)
    sc2 = evaluate(docs_df,test,returnscore=True)
    worst_enj = ''
    dummy = [1,1,1,1]
    for e in sc2:
        if sc2[e][3]<dummy[3]:
            worst_enj = e
            dummy[3] = sc2[e][3]
    moy = delta(sc1,sc2,returnmoy=True)

#%%%
ress = []
for row1,row2 in zip(topic_model[k].p_y_given_x,topic_model[k].labels):
    line = []
    for el1,el2 in zip(row1,row2):
        line.append((el1,el2))
    ress.append(line)
ress = np.matrix(ress)
# %%

####TOUT CE QUI EST DESSOUS = PISTES NULLES

#%%
#On va essayer d'optimiser en ajoutant un enjeux poubelle vide

deltamoy = [0,0,0,0]
for it in range(20):
    modelPoub = ct.Corex(n_hidden=len(enjeux_list)+1)
    modelPoub.fit(X, words=vocab, anchors=thesau_list, anchor_strength=2+1)

    #Prédiction et évaluation sur sur toutes les données
    test = modelPoub.predict(X)
    sc2 = evaluate(docs_df,test,returnscore=True)

    #Calcul du delta des évaluations
    deltamoy = vadd(deltamoy,delta(sc1,sc2,returnmoy=True))
deltamoy = np.array(deltamoy)/20 

#deltamoy = [0.02,0.01,-0.02,-0.01]
#En moyenne (20 it), les performances sont moins bonnes !
#%%
#On va tenter de seeder le topic inutile
thesau_list_poub = thesau_list.copy().append(['mrae','administratif','region',
'autorite environnemental','dreal',
    'loi','reglemen'])
deltamoy = [0,0,0,0]
for it in range(20):
    modelPoub2 = ct.Corex(n_hidden=len(enjeux_list)+1)
    modelPoub2.fit(X, words=vocab, anchors=thesau_list_poub, anchor_strength=k+1)

    #Prédiction et évaluation sur sur toutes les données
    test = modelPoub2.predict(X)
    sc2 = evaluate(test,returnscore=True)
    sc1 = evaluate(topic_model[2].labels,returnscore=True)

    #Calcul du delta des évaluations
    
    deltamoy = vadd(deltamoy,delta(sc1,sc2,returnmoy=True))

deltamoy = np.array(deltamoy)/20 

#deltamoy = [0.03,0.02,-0.04,-0.05]
#En moyenne (20 it), les performances sont moins bonnes !