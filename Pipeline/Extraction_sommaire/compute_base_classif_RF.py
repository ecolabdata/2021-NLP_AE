# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 1. Traitement des résultats du Kmeans

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from bs4 import BeautifulSoup
import time
from unidecode import unidecode
import re

# Read recipe inputs
base_classif_Kmeans_1 = dataiku.Dataset("base_classif_Kmeans_1")
base= base_classif_Kmeans_1.get_dataframe(sampling='head', limit=100000)
base

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 1.1 Vérification des labelisés 1 par le Kmeans

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
N=1000
verif_theo=base.iloc[:N,[0,1,base.shape[1]-1]]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
cleantext = BeautifulSoup(verif_theo.phrase[3], "html.parser").text
cleantext

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
verif_theo['phrase_2']=[BeautifulSoup(i,"html.parser").text for i in verif_theo.phrase]
#[verif_theo.phrase.values[i].split('>')[1].split('<')[0] ]
verif_theo

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for i in verif_theo[verif_theo.label_k==1].index:
    print("Indice :",i,"\n",verif_theo.phrase_2[i])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Bon sur l'échantillon de 1000, les labelisés 1 semblent être tous des "non-titres", donc c'est super !

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 1.2 Vérification des labelisés 0 par le Kmeans
# ### 1.2.1 Essai sur sous-ensemble

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for i in verif_theo[verif_theo.label_k==0].index:
    print("Indice :",i,"\n",verif_theo.phrase_2[i])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
verif_theo.num_etude[0]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def sommaire(s):
    somm=['SOMMAIRE','sommaire','Sommaire']
    s=list(s)
    indice=[s.index(k) for k in somm if k in s]
    #[list(verif_theo[verif_theo.label_k==0][verif_theo.num_etude==verif_theo.num_etude[0]].phrase_2.values).index(k) for k in somm if k in list(verif_theo[verif_theo.label_k==0][verif_theo.num_etude==verif_theo.num_etude[0]].phrase_2.values)]
    return indice
sommaire(verif_theo[verif_theo.label_k==0][verif_theo.num_etude==verif_theo.num_etude[0]].phrase_2.values)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# On peut chercher "sommaire" et tout ce qui est au-dessus on met label=1
# 
# Ensuite on peut tenter le fameux truc : dès qu'un des titres est répété, on label=1 tout ce qui suit ?7
# 
# En fait la question c'est comment isoler uniquement les titres sur un subset ?

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def light_cleaning(s):
    import re
    from unidecode import unidecode
    import numpy as np
    s=unidecode(s)
    s=re.sub(r'[^A-Za-z]',' ',s)
    s=s.lower()
    s=s.split(' ')
    s=list(np.unique(s))
    try:
        s.remove('')
        return s
    except:
        return s

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
len(liste)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
dico_titres={}
for i in np.unique(verif_theo[verif_theo.label_k==0].num_etude):
    liste=verif_theo[verif_theo.label_k==0][verif_theo.num_etude==i].phrase_2.values
    indice_sommaire=sommaire(liste)
    print(indice_sommaire)
    a=light_cleaning(liste[indice[0]+1]) #hypothèse donc que le 1er titre vient après le sommaire
    print(a)
    for j in range(indice[0]+1+1,len(liste)):
        b=light_cleaning(liste[j])
        if a==b:
            dico_titres[i]=[j,liste[j]]
            break
        else:
            continue
print(dico_titres)
titres=liste[indice[0]+1:+dico_titres[1003691][0]+1]
titres

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### 1.2.2 Généralisation

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def sommaire(s):
    somm=['SOMMAIRE','sommaire','Sommaire']
    s=list(s)
    indice=[s.index(k) for k in somm if k in s]
    #[list(verif_theo[verif_theo.label_k==0][verif_theo.num_etude==verif_theo.num_etude[0]].phrase_2.values).index(k) for k in somm if k in list(verif_theo[verif_theo.label_k==0][verif_theo.num_etude==verif_theo.num_etude[0]].phrase_2.values)]
    return indice

def light_cleaning(s):
    s=unidecode(s)
    s=re.sub(r'[^A-Za-z]',' ',s)
    s=s.lower()
    s=s.split(' ')
    s=list(np.unique(s))
    try:
        s.remove('')
        return s
    except:
        return s

def titres_ps(verif):
    verif_theo=verif.copy()
    verif_theo['phrase_2']=[BeautifulSoup(i,"html.parser").text for i in verif_theo.phrase]
    dico_titres={}
    titres={}
    for i in np.unique(verif_theo[verif_theo.label_k==0].num_etude):
        liste=verif_theo[verif_theo.label_k==0][verif_theo.num_etude==i].phrase_2.values
        indice_sommaire=sommaire(liste)
        if len(indice_sommaire)>0:
            print(indice_sommaire)
            a=light_cleaning(liste[indice_sommaire[0]+1]) #hypothèse donc que le 1er titre vient après le sommaire
            print(a)
            for j in range(indice_sommaire[0]+1+1,len(liste)):
                b=light_cleaning(liste[j])
                if a==b:
                    dico_titres[i]=[j,liste[j]]
                    print(dico_titres[i])
                    titres[i]= liste[indice_sommaire[0]+1:+dico_titres[i][0]+1]
                    break
                else:
                    continue
        else:
            dico_titres[i]=['pas de "sommaire" ?']
            titres[i]=['pas de titres']
    return titres,verif_theo

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
titres,base_2=titres_ps(base)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
titres.keys()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import functools
import operator
np.unique(functools.reduce(operator.iconcat, ([titres[i] for i in titres.keys()]), []))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 1.3 Création de la base d'entraînement pour la classification supervisée

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_classif_1=base_2[base_2.label_k==1].reset_index()
base_classif_1['phrase_2']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
TITRE=np.unique(functools.reduce(operator.iconcat, ([titres[i] for i in titres.keys()]), []))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
len(TITRE)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#base_classif_0=
index=[base_2.phrase_2[i] in TITRE for i in base_2.index]
#base_2[base_2.label_k==0][
#    [base_2.num_etude[i] in list(titres.keys()) for i in base_2[base_2.label_k==0].index]][
#    [base_2.phrase_2[i] in TITRE for i in base_2.index]]
base_classif_0=base_2[index].reset_index()#.phrase_2.values

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
len(base_classif_0)+len(base_classif_1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_classif=pd.concat([base_classif_0,base_classif_1])
base_classif

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 2. Classification supervisée

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(base_classif.iloc[:,3:-2],base_classif.iloc[:,-2],test_size=0.25)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Train : proportion de 1 :",round(y_train.sum()/len(y_train)*100,1)," ; de 0 :",round((1-y_train.sum()/len(y_train))*100,1))
print("Test : proportion de 1 :",round(y_test.sum()/len(y_test)*100,1)," ; de 0 :",round((1-y_test.sum()/len(y_test))*100,1))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Arbre de décision

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
k=15
X_train_2=X_train.iloc[:,:(X_train.shape[1]-k)]
X_test_2=X_test.iloc[:,:(X_train.shape[1]-k)]

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(class_weight="balanced")
DTC.fit(X_train_2, y_train)

print("Score train :",round(DTC.score(X_train_2,y_train),3))
print("Score test :",round(DTC.score(X_test_2,y_test),3))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,DTC.predict(X_test_2))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("La variable",X_train_2.columns[-1],"permet d'atteindre un score de 1.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
np.corrcoef(X_train_2.iloc[:,X_train_2.columns[-1]],y_train)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Forêt aléatoire

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
absi=[]
ordo=[]
ordo_test=[]

for i in range(10,X_train.shape[1],10):
    X_train_2=X_train.iloc[:,:i]
    X_test_2=X_test.iloc[:,:i]

    DTC = RandomForestClassifier(n_estimators=10,verbose=0,class_weight="balanced")
    DTC.fit(X_train_2, y_train)

    ordo.append(DTC.score(X_train_2,y_train))
    ordo_test.append(DTC.score(X_test_2,y_test))
    absi.append(i)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import matplotlib.pyplot as plt
f,a=plt.subplots(1,figsize=(12,6))
a.plot(absi,ordo)
a.plot(absi,ordo_test)
a.set(xlabel="Nombre de variables",ylabel='Score',
      title='Score de la random forest en fonction du nombre de variables')
plt.legend(['train','test'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=1,verbose=0,class_weight="balanced")
RFC.fit(X_train, y_train)

print("Score train :",round(RFC.score(X_train,y_train),3))
print("Score test :",round(RFC.score(X_test,y_test),3))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,RFC.predict(X_test))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Forêt aléatoire et bagging

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.ensemble import BaggingClassifier

RFC = RandomForestClassifier(n_estimators=100,verbose=0,class_weight="balanced")
Bagging = BaggingClassifier(RFC,n_estimators=100,verbose=0)
Bagging.fit(X_train, y_train)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Score train :",round(Bagging.score(X_train,y_train),3))
print("Score test :",round(Bagging.score(X_test,y_test),3))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,Bagging.predict(X_test))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Fait très étonnant : la random forest réussit à classer **parfaitement** l'ensemble des unités
# 
# On garde le dernier modèle : bien qu'ils aient tous un score de 1, la double utilisation de Bagging sur une RF permet de s'assurer des bonnes propriétés de généralisation hors échantillon.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 3. Prédiction supervisée de l'ensemble de notre base

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Attention le chargement de la base prend généralement 25 minutes

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base = dataiku.Dataset("base_classif_Kmeans_1").get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
start=time.time()
base['label_RF']=Bagging.predict(base.iloc[:,2:-1])
end=time.time()
print("La prédiction de l'ensemble a pris :",round((end-start)/60,2),"minutes")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_classif_RF_df = base
base_classif_RF = dataiku.Dataset("base_classif_RF")
base_classif_RF.write_with_schema(base_classif_RF_df)