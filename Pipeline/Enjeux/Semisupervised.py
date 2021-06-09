#%%

import pandas as pd, numpy as np
import pickle
import unidecode
import spacy
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from Pipeline.Enjeux.score import *
import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct
stop_words = stopwords.words('french')
stop_words.extend(['avis','environnement','autorite','projet','etude','exploitation','impact','site','dossier','mission','regionale','mrae','mnhn'])

# ## 2. Topic modeling semi-supervisé
#%%
#Charger thésau et data et vecto data
docs_df = pickle.load(open('Data/Workinprogress/docs_df.pickle','rb'))
Thesaurus = pickle.load(open('Data\Thesaurus_csv\Thesaurus1_clean.pickle','rb'))

from Pipeline.Enjeux.processing_encoding import get_info

countVecto = CountVectorizer(min_df = 13, max_df = 0.95, ngram_range=(1,3), stop_words = stop_words)
process = countVecto.fit_transform(docs_df.text_processed.values)  
X = process.toarray().astype(int)
X = np.matrix(X)

word2id,vocab,words_freq,vocab_sort,notinvoc = get_info(countVecto,X,Thesaurus)

enjeux_list = Thesaurus.Enjeux.values
thesau_list = Thesaurus.Dictionnaire.values


#%%

corrige = pd.read_excel("Data\Workinprogress\Dataframe en correction.xlsx")
corrige.dropna(inplace = True)
corrige.drop(['titre', 'url_etude', 'url_avis', 'url_avis_cliquable', 'Status',
        'Biodiversité',
    'Paysage et qualité de vie',
    'Santé et sécurité des personnes',
    'Effets globaux (climat, énergie, ressources...)',
    'Préservation des sols',
    'Qualité de l’eau et ressource',
    'Déplacements', 'Gestion des déchets'], axis = 1, inplace = True)
corrige.id_AAE = corrige.id_AAE.astype(int)

#%%
from distutils.util import strtobool
def cleanstrtobool(x):
    if type(x) != str:
        return(x)
    return(strtobool(x))

def evaluate(y,df_corrige = corrige,returnscore = False):
    labels = pd.concat([docs_df,pd.DataFrame(y[:,:len(enjeux_list)],columns =enjeux_list)],axis=1)
    labels.rename(columns={'id':'id_AAE'},inplace = True)
    labels.dropna(inplace = True)
    labels.id_AAE = labels.id_AAE.astype(int)
    etudes_avis_dep_them_id_df = pd.read_csv("Data\Workinprogress\etudes_avis_dep_them_id.csv")

    final = etudes_avis_dep_them_id_df.merge(labels, on = 'id_AAE', how='inner')
    final = final.drop(['text_processed','theme','departement'],axis = 1)

    final =df_corrige.merge(final, on = 'id_AAE', how='inner')
    
    y_pred = []
    y_true = []
    for enjeu in enjeux_list:
        y_pred.append(final[enjeu].apply(lambda x: cleanstrtobool(x)).values)
        y_true.append(final['True_'+enjeu].apply(lambda x: cleanstrtobool(x)).values)

    y_pred = np.matrix(y_pred).T
    y_true = np.matrix(y_true).T
    
    sc = scores(y_pred,y_true, labels = enjeux_list)

    hotgrid_score(enjeux_list,sc,col = 'seismic')
    hotgrid_corr(enjeux_list,y_true.T)
    if returnscore:
        return(sc)

def separate(X,model,df_corrige = corrige):
    labels = pd.concat([docs_df,pd.DataFrame(model.labels[:,:len(enjeux_list)],columns = enjeux_list.tolist())],axis=1)
    labels.rename(columns={'id':'id_AAE'},inplace = True)
    labels.dropna(inplace = True)
    labels.id_AAE = labels.id_AAE.astype(int)
    etudes_avis_dep_them_id_df = pd.read_csv("Data\Workinprogress\etudes_avis_dep_them_id.csv")

    final = etudes_avis_dep_them_id_df.merge(labels, on = 'id_AAE', how='inner')
    final = final.drop(['text_processed'],axis = 1)
    final['idx_copy'] = final.index

    final =df_corrige.merge(final, on = 'id_AAE', how='inner')
    
    y_pred = []
    y_true = []
    for enjeu in enjeux_list:
        y_pred.append(final[enjeu].apply(lambda x: cleanstrtobool(x)).values)
        y_true.append(final['True_'+enjeu].apply(lambda x: cleanstrtobool(x)).values)

    y_pred = np.matrix(y_pred).T
    y_true = np.matrix(y_true).T
    X_sub = X[final.idx_copy.values,:]
    return(y_pred,y_true,X_sub)

#%%

def topwords(model):
    topics = model.get_topics()
    for topic_n,topic in enumerate(topics):
        # w: word, mi: mutual information, s: sign
        topic = [(w,mi,s) if s > 0 else ('~'+w,mi,s) for w,mi,s in topic if w not in dicoThesau[enjeux_list[topic_n]]]
        # Unpack the info about the topic
        words,mis,signs = zip(*topic)    
        # Print topic
        topic_str = str(enjeux_list[topic_n])+': '+', '.join(words)
        print(topic_str)

def vlin(l1,l2,sign):
    r = []
    for i1,i2 in zip(l1,l2):
        if sign == '-':
            r.append(i1-i2)
        else:
            r.append(i1+i2)
    return(r)

#%%
topic_model = {}
#%%
k = 2
topic_model[k] = ct.Corex(n_hidden=len(enjeux_list))
topic_model[k].fit(X, words=vocab, anchors=thesau_list, anchor_strength=k+1)


#%%
#On va essayer d'optimiser en faisant du stratified sampling

from Pipeline.Enjeux.multilabel_balancing import *

#Stratification a partir des éléments labellisés

#On récupère les y labellisés (vrais), 
# et y prédit et X sur les mêmes indices
y_pred,y_true,X_sub = separate(X,topic_model[k])
y_true_df = pd.DataFrame(y_true,columns = enjeux_list)
X_df = pd.DataFrame(X_sub,columns=vocab)

#On récupère un sous ensemble de ces X 
# avec des proportions équilibrées pour chaque enjeu
X_sub, y_sub = get_minority_instance(X_df, y_true_df)

#On oversample pour équilibrer !!! NE MARCHE PAS
#X_res,y_res = MLSMOTE(X_sub,y_sub,50)

#On entraine un nouveau modèle sur le sous ensemble uniquement

model = ct.Corex(n_hidden=len(enjeux_list))
model.fit(np.matrix(X_sub), words=vocab, anchors=thesau_list, anchor_strength=k+1)

#Prédiction et évaluation sur sur toutes les données
test = model.predict(X)
sc2 = evaluate(test,returnscore=True)
sc1 = evaluate(topic_model[2].labels,returnscore=True)

#Calcul du delta des évaluations
diff = {}
moy = [0,0,0,0]
for enj in enjeux_list:
    diff[enj] = vlin(sc2[enj],sc1[enj],'-')
    moy = vlin(moy,vlin(sc2[enj],sc1[enj],'-'),'+')

for k in range(len(moy)):
    moy[k] = moy[k]/len(enjeux_list)

hotgrid_score(enjeux_list,diff,col='seismic')


#%%

ep = 2

#Initialisation a partir des données corrigées
model = ct.Corex(n_hidden=len(enjeux_list))
model.fit(np.matrix(X_sub), words=vocab, anchors=thesau_list, anchor_strength=k+1)
for i in range(ep):
    pred = model.predict(X)
    pred_df = pd.DataFrame(pred,columns = enjeux_list)
    X_sub,y_sub = get_minority_instance(pd.DataFrame(X,columns=vocab), pred_df)
    model = ct.Corex(n_hidden=len(enjeux_list))
    model.fit(np.matrix(X_sub), words=vocab, anchors=thesau_list, anchor_strength=k+1)

# %%


#%%


#%%%
ress = []
for row1,row2 in zip(topic_model[k].p_y_given_x,topic_model[k].labels):
    line = []
    for el1,el2 in zip(row1,row2):
        line.append((el1,el2))
    ress.append(line)
ress = np.matrix(ress)
# %%
