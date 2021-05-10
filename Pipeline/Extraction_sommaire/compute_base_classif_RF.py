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
import pickle

# Read recipe inputs
#base_classif_Kmeans_1 = dataiku.Dataset("base_classif_Kmeans_1")
#base= base_classif_Kmeans_1.get_dataframe()#sampling='head', limit=100000)
da=dataiku.Folder('lZ0B3sSL')
base_finale=pickle.load(open(da.get_path()+"/base_pour_Bagging_final.pickle",'rb'))
base_finale
print('Il y a ',len(np.unique(base_finale.num_etude)),"études")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Attention à bien vérifier la correspondance des labels du Kmeans !!

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 1.1 Vérification des labelisés 1 par le Kmeans

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
N=10000

for i in base_finale[base_finale.label_k_1==1].iloc[:N,:].index:
    print("Indice :",i,"\n",base_finale.phrase_2[i])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
N=10000

for i in base_finale[base_finale.label_k_2==1].iloc[:N,:].index:
    print("Indice :",i,"\n",base_finale.phrase_2[i])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Bon sur l'échantillon de 1000, les labelisés 1 semblent être tous des "non-titres", donc c'est super !

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 1.2 Vérification des labelisés 0 par le Kmeans
### 1.2.1 Essai sur sous-ensemble

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for i in base_finale[base_finale.label_k==1].phrase_2:
    print(i)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
N=10000

for i in base_finale[base_finale.ind_sommaire==1][base_finale.label_k==1].iloc[:N,:].index:
    print("Indice :",i,"\n",base_finale.phrase_2[i])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base=base_finale.iloc[:200000,:]
c=[1 if (base.ind_sommaire[i]==1) and (base.label_k[i]==1) else 0 for i in base.index]
c_index=[i for i in base.index if base.phrase_2[i]=='^' or base.phrase_2[i]==base.iloc[141175,:].phrase_2]
for i in c_index:
    c[i]=0
c[29533:29584]=np.zeros(29584-29533)
c[29901:139926]=np.zeros(139926-29901)
#x1=range(140369,140376,2)
#x2=range(140381,140697,2)
#x3=range(140700,140743,2)
#base[base.phrase_2=='^']
c[141223:171412]=np.zeros(171412-141223)
c[177977:186254]=np.zeros(186254-177977)
base['label']=c
base

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#c_index
#base[base.phrase_2==base.iloc[141175,:].phrase_2]
base.label.sum()/base.shape[0]*100

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base[base.label==1]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for i in base[base.label==1].index:
    print("Indice :",i,"\n",base.phrase_2[i])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Désormais, on a une nouvelle variable : label.

Elle est le croisement de indicatrice_sommaire et label_k puis passe dans un nouveau nettoyage. On peut donc donner les anciennes variables utilisées pour construire notre nouvel algorithme.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base.iloc[:,3:-1]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 2. Classification supervisée

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Attention, désormais, titres ==1 et non_titres==0

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(base.iloc[:,3:-1],base.loc[:,'label'],test_size=0.25)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Train : proportion de 1 :",round(y_train.sum()/len(y_train)*100,1)," ; de 0 :",round((1-y_train.sum()/len(y_train))*100,1))
print("Test : proportion de 1 :",round(y_test.sum()/len(y_test)*100,1)," ; de 0 :",round((1-y_test.sum()/len(y_test))*100,1))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Arbre de décision

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
k=0 # On a plus besoin d'étudier le k
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
#print("La variable",X_train_2.columns[-1],"permet d'atteindre un score de 1.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#np.corrcoef(X_train_2.iloc[:,X_train_2.columns[-1]],y_train)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Forêt aléatoire

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
################ NE PAS LANCER !
######## De même, plus besoin d'étudier cela, je laisse pour l'historique du travail
absi=[]
ordo=[]
ordo_test=[]

for i in range(1,X_train.shape[1]):
    X_train_2=X_train.iloc[:,:i]
    X_test_2=X_test.iloc[:,:i]

    DTC = RandomForestClassifier(n_estimators=10,verbose=0,class_weight="balanced")
    DTC.fit(X_train_2, y_train)

    ordo.append(DTC.score(X_train_2,y_train))
    ordo_test.append(DTC.score(X_test_2,y_test))
    absi.append(i)

import matplotlib.pyplot as plt
f,a=plt.subplots(1,figsize=(12,6))
a.plot(absi,ordo)
a.plot(absi,ordo_test)
a.set(xlabel="Nombre de variables",ylabel='Score',
      title='Score de la random forest en fonction du nombre de variables')
plt.legend(['train','test'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
############## On passe aux forêts :
n=10 # On essaie avec 10 estimateurs

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=n,verbose=0,class_weight="balanced")
RFC.fit(X_train, y_train)

print("Score train :",round(RFC.score(X_train,y_train),3))
print("Score test :",round(RFC.score(X_test,y_test),3))

from sklearn.metrics import confusion_matrix
print("Matrice de confusion sur le train \n",confusion_matrix(y_train,RFC.predict(X_train)))
print("Matrice de confusion sur le test \n",confusion_matrix(y_test,RFC.predict(X_test)))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# On fait encore des erreurs, donc on poursuit, on augmente n

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
n=100 # On essaie avec 100 estimateurs

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=n,verbose=0,class_weight="balanced")
RFC.fit(X_train, y_train)

print("Score train :",round(RFC.score(X_train,y_train),3))
print("Score test :",round(RFC.score(X_test,y_test),3))

from sklearn.metrics import confusion_matrix
print("Matrice de confusion sur le train \n",confusion_matrix(y_train,RFC.predict(X_train)))
print("Matrice de confusion sur le test \n",confusion_matrix(y_test,RFC.predict(X_test)))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Malgré la multiplication par 10 du nombre d'estimateurs, on ne réussit pas à classifier parfaitement. Poursuivons.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Forêt aléatoire et bagging

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# On va faire 100 fois la sélection du meilleur estimateur parmi 100 forêt aléatoire estimée.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.ensemble import BaggingClassifier

RFC = RandomForestClassifier(n_estimators=100,verbose=0,class_weight="balanced")#,bootstrap=False) #à voir
Bagging = BaggingClassifier(RFC,n_estimators=100,verbose=0)
Bagging.fit(X_train, y_train)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Score train :",round(Bagging.score(X_train,y_train),3))
print("Score test :",round(Bagging.score(X_test,y_test),3))

from sklearn.metrics import confusion_matrix
print("Matrice de confusion sur le train \n",confusion_matrix(y_train,Bagging.predict(X_train)))
print("Matrice de confusion sur le test \n",confusion_matrix(y_test,Bagging.predict(X_test)))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Malgré le Bagging, on ne réussit pas à classer **parfaitement**.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base['label_rf']=Bagging.predict(base.iloc[:,3:-1])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for i in base[base.label_rf==1].phrase_2:
    print(i)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
(base.label_rf.sum()/len(base))*100

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base.columns

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# On va sauvegarder la base d'entraînement avec son résultat pour reproduction

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
da=dataiku.Folder('lZ0B3sSL')
import pickle
dico_Bagging={}
dico_Bagging['train']=[X_train,y_train]
dico_Bagging['test']=[X_test,y_test]
dico_Bagging['base']=base
dico_Bagging['modele']=Bagging
pickle.dump(dico_Bagging,open(da.get_path()+"/dico_Bagging.pickle",'wb'))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Vu le travail qu'on a fait de re-labellisation (semi-supervisation) on va sauvegarder ce modèle (et vérifier qu'on l'a bien enregistré en le re-téléchargeant)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#clf_titres_fin=Bagging.copy()
da=dataiku.Folder('lZ0B3sSL')
import pickle
pickle.dump(Bagging,open(da.get_path()+"/Bagging_model.pickle",'wb'))
#da.upload_file("Bagging_model", Bagging)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
clf_titres_fin=pickle.load(open(da.get_path()+"/Bagging_model.pickle",'rb'))
print(clf_titres_fin)
clf_titres_fin.predict(base.iloc[:100,3:-2])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 3. Prédiction supervisée de l'ensemble de notre base

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Attention le chargement de la base prend généralement 25 minutes

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
########### Chargement du modèle
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import time
import pickle
da=dataiku.Folder('lZ0B3sSL')
dico=pickle.load(open(da.get_path()+"/dico_Bagging.pickle",'rb'))
clf_titres_fin=dico['modele']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_finale=pickle.load(open(da.get_path()+"/base_pour_Bagging_final.pickle",'rb'))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_finale.shape

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Attention à bien laisser la dernière variable de la grosse base puisque le modèle utilise label_k pour prédire label

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
start=time.time()
pred=clf_titres_fin.predict(base_finale.iloc[:,3:])
end=time.time()
print("Durée de la prédiction :",round((end-start)/60,2),"minutes")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pred.shape,base_finale.shape

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_finale['label_RF']=pred

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_finale.label_RF.sum()/len(base_finale)*100

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for i in base_finale[base_finale.label_RF==1].iloc[:,:].phrase_2:
    print(i)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
da=dataiku.Folder('lZ0B3sSL')
import pickle
pickle.dump(base_finale,open(da.get_path()+"/Base_label_RF.pickle",'wb'))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_classif_RF_df = base_finale
base_classif_RF = dataiku.Dataset("base_classif_RF")
base_classif_RF.write_with_schema(base_classif_RF_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_finale

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 4. Analyse des titres non détectés (exploratoire, à ne pas faire tourner sauf exception)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Dans un premier temps, quantifions la quantité de documents n'ayant pas de titres détectés.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_finale

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
vide=[i for i in np.unique(base_finale.num_etude) if base_finale[base_finale.num_etude==i].label_RF.sum()==0]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
np.unique(base_finale.num_etude)[0]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
non_vide=[i for i in np.unique(base_finale.num_etude) if i not in vide]
len(non_vide)/len(np.unique(base_finale.num_etude))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
index_vide=[True if base_finale.num_etude[i] in vide else False for i in base_finale.index]
base_relou=base_finale[index_vide]
base_relou

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Proportion des labels kmeans chez les vides,\nnon-titres (1) :",round(base_relou.label_k.sum()/len(base_relou)*100,2))
print("Proportion des labels kmeans chez les non vides,\nnon-titres (1) :",round(base_propre.label_k.sum()/len(base_propre)*100,2))
print("Proportion des labels kmeans total,\nnon-titres (1) :",round(base_finale.label_k.sum()/len(base_finale)*100,2))
numero=np.unique(base_relou.num_etude)
k=0
titres=list(base_relou[base_relou.label_k==1][base_relou.num_etude==numero[k]].phrase_2.values)
for i in titres:
    print('\nIndice :',titres.index(i),i)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
numero=np.unique(base_relou.num_etude)
k=0
titres=list(base_relou[base_relou.ind_sommaire==1][base_relou.num_etude==numero[k]].phrase_2.values)
for i in titres:
    print('\nIndice :',titres.index(i),i)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_relou[base_relou.ind_sommaire==1][base_relou.num_etude==numero[k]]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Problème sur le clustering Kmeans, probablement du à ind_sommaire ?

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
len(base_relou),len(base_finale)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
index_nonvide=[bool(abs(i-1)) for i in index_vide]
base_propre=base_finale[index_nonvide]
distrib_nbligne_propre=[len(base_propre[base_propre.num_etude==i]) for i in np.unique(base_propre.num_etude)]
print("Pourcentage d'études vides :",round(len(non_vide)/len(np.unique(base_finale.num_etude))*100,2),"%")
print('Les documents vides représentent',round(len(base_relou)/len(base_finale)*100,2),'% de la base pour la classification supervisée')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
numero=np.unique(base_propre.num_etude)
k=8
titres=list(base_propre[base_propre.label_k==1][base_propre.num_etude==numero[k]].phrase_2.values)
for i in titres:
    print('\nIndice :',titres.index(i),i)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
distrib_nbligne_vide=[len(base_relou[base_relou.num_etude==i]) for i in np.unique(base_relou.num_etude)]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def stat_des(x,mo,k=2):
    if mo=='mean':
        y=np.mean(x)
    elif mo=='std':
        y=np.std(x)
    elif mo=='med':
        y=np.median(x)
    else:
        raise ValueError('Veuillez spécifier quel type de stat des vous voulez. Valeurs possibles : mean, std ou med')
    z=round(y,k)
    return z

print("Pour les documents non vides, distribution du nombre de lignes html, \nmoyenne :",stat_des(distrib_nbligne_propre,'mean'),"\nécart-type :",stat_des(distrib_nbligne_propre,'std'),'\nmédiane :',stat_des(distrib_nbligne_propre,'med'))
print("Pour les documents vides, distribution du nombre de lignes html, \nmoyenne :",stat_des(distrib_nbligne_vide,'mean'),"\nécart-type :",stat_des(distrib_nbligne_vide,'std'),'\nmédiane :',stat_des(distrib_nbligne_vide,'med'))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import matplotlib.pyplot as plt
figue,hache=plt.subplots(figsize=(18,12))
#sns.kdeplot(distrib_nbligne_propre , bw = 0.5 , fill = True)
#sns.kdeplot(distrib_nbligne_vide , bw = 0.5 , fill = True)
sns.distplot(distrib_nbligne_propre)
sns.distplot(distrib_nbligne_vide)

#hache.hist(distrib_nbligne_propre,density=True,bins=20)
#hache.hist(distrib_nbligne_vide,density=True,bins=20)
plt.legend(['propre','vide'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Maintenant qu'on a analysé le problème et trouvé une source de problème, on repasse la base_relou à la moulinette pour voir si ça s'arrange

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def sommaire(s):
    '''
    Fonction cherchant l'indice du début du sommaire dans la liste de phrase.
    '''
    somm=['SOMMAIRE','sommaire','Sommaire']
    s=list(s)
    indice=[s.index(k) for k in somm if k in s]
    if len(indice)==0:
        indice=[i for i in range(len(s)) if np.sum([1 if som in s[i] else 0 for som in somm])>0]
        if len(indice)>2:
            indice=[indice[0]]
        if len(indice)==0:
            from unidecode import unidecode
            technique=[unidecode('table des matières'.lower()),'non technique']
            indice=[i for i in range(len(s)) if np.sum([1 if som in unidecode(s[i].lower()) else 0 for som in technique])>0]
    #[list(verif_theo[verif_theo.label_k==0][verif_theo.num_etude==verif_theo.num_etude[0]].phrase_2.values).index(k) for k in somm if k in list(verif_theo[verif_theo.label_k==0][verif_theo.num_etude==verif_theo.num_etude[0]].phrase_2.values)]
    return indice

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


def indice_relou(liste,indice_sommaire,seuil,seuil_c):
    '''
    Fonction qui permet de produire l'indicatrice sommaire pour les études récalcitrantes : c'est-à-dire dont le sommaire est difficilement détectable.
    Par exemple à cause de mauvaises répétitions des titres, d'absence de mots indiquants le début ou la fin du sommaire etc...
    '''
    import re
    output=None
    x=0
    #if procede=='basic':
    while output is None:
        x+=1
        a=light_cleaning(liste[indice_sommaire[0]+x])
        for j in range(indice_sommaire[0]+1+1,len(liste)):
            b=light_cleaning(liste[j])
            cnt = 0
            if len(a)==1:
                s=0
                for i in a:
                    if type(b)==str:
                        if re.search(i, b) is not None:
                            cnt = cnt + 1
                cnt=cnt/len(a)
                count=len(set(a)&set(b))/len(a)
            elif len(a)>1:
                aa=''.join([i for i in a])
                for i in aa:
                    if type(b)==str:
                        if re.search(i, b) is not None:
                            cnt = cnt + 1
                cnt=cnt/len(aa)
                count=len(set(aa)&set(b))/len(aa)
                s=0
                if type(b)==str:
                    for i in b:
                        if i in a:
                            s=1

            if (b in a) or (a in b) or (cnt>seuil) or (count>seuil_c) or (s>0):
                output=j
                #vector=np.ones(len([indice_sommaire[0]:j]))
                break
            else:
                continue
  #  elif procede=='further':
   #     a=light_cleaning(liste[indice_sommaire[0]])
    #else:
       # raise ValueError("Hum... Êtes-vous sûr que vous avez correctement spécifié le variable procede ? \nLes valeurs possibles sont 'basic' et 'further' (pour les études récalcitrantes).")

    return output

def ind_titres(verif,brutal=None,seuil_sommaire=50,seuil=0.35,seuil_c=0.35):
    verif_theo=verif.copy()
    titres={}
    vector_=[]
    indexes = np.unique(verif_theo.num_etude, return_index=True)[1]
    print(indexes)
    try:
        interval=[verif_theo.num_etude[index] for index in sorted(indexes)]
    except:
        verif_theo=verif_theo.reset_index(drop=True)
        interval=[verif_theo.num_etude[index] for index in sorted(indexes)]
    print(interval)
    h=0
    for i in interval:
        print("On étudie l'étude :",i)
        liste=verif_theo[verif_theo.num_etude==i].phrase_2.values
        vector=np.zeros(len(liste))
        vector_.append(vector)
        h=h+len(vector)
        #vector_.append(vector)
        indice_sommaire=sommaire(liste)
        if len(indice_sommaire)>0:
            a=light_cleaning(liste[indice_sommaire[0]+1]) #hypothèse donc que le 1er titre vient après le sommaire
            print("L'indice de début de sommaire pour l'étude",i,"est",indice_sommaire[0],a)
            for j in range(indice_sommaire[0]+1+1,len(liste)):
                b=light_cleaning(liste[j])
                if a==b:
                    vector_[interval.index(i)][indice_sommaire[0]:j]=1
                    break
                else:
                    continue
            if np.sum(vector_[interval.index(i)])>0:
                print("Il y a un sommaire pour l'étude :",i)
            elif np.sum(vector_[interval.index(i)])==0:
                print('On tente la nouvelle méthode')
                j=indice_relou(liste,indice_sommaire,seuil,seuil_c)
                if j is not None:
                    vector_[interval.index(i)][indice_sommaire[0]:j]=np.ones(len(range(indice_sommaire[0],j)))
                    ss=np.sum(vector_[interval.index(i)])
                    if ss>seuil_sommaire:
                        print("Il y a un sommaire pour l'étude :",i)
                    elif (ss>0) and (ss<seuil_sommaire) and (brutal is not None):
                        print("Hum... Le sommaire semble petit :",ss)
                        vector_[interval.index(i)][indice_sommaire[0]:brutal]=np.ones(len(range(indice_sommaire[0],brutal)))
                    else:
                        print("Pas de sommaire pour l'étude :",i)
                else:
                    print("Pour l'étude",i,"on n'a pas trouvé d'indice pour débuter le sommaire.")
        else:
            print("Pas de sommaire pour l'étude :",i)
    return vector_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
technique=['non technique',unidecode('Table des matières'.lower())]
numero=np.unique(base_relou.num_etude)
indice=[i for i in range(len(base_relou[base_relou.num_etude==numero[3]].phrase_2.values))
        if np.sum([1 if som in unidecode(base_relou[base_relou.num_etude==numero[3]].phrase_2.values[i].lower()) else 0 for som in technique])>0]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
numero=np.unique(base_relou.num_etude)
print("Il y a ",len(numero),"études vides")
k=8

titres=list(base_relou[base_relou.ind_sommaire==1][base_relou.num_etude==numero[k]].phrase_2.values)
for i in titres:
    print('\nIndice :',titres.index(i),"\n",i)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
numero=np.unique(base_relou.num_etude)
index=[i for i in numero if np.sum(base_relou[base_relou.num_etude==i].ind_sommaire)==0]
base_relou_2=base_relou[[True if base_relou.num_etude[i] in index else False for i in base_relou.index]]
print("Les documents vides tenaces représentent :",round(len(base_relou_2)/len(base_relou)*100,2),'% des lignes de la base relou')
print("On a réussi à traiter ",round((1-len(np.unique(base_relou_2.num_etude))/len(numero))*100,2),"% des documents vides initiaux")
print("Il y a ",len(np.unique(base_relou_2.num_etude)),"études vides")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
numero_2=np.unique(base_relou_2.num_etude)
k=0

titres=list(base_relou_2[base_relou_2.label_k==0][base_relou_2.num_etude==112862].phrase_2.values)
for i in titres:
    print('\nIndice :',titres.index(i),"\n",i)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
liste=base_relou_2[base_relou_2.label_k==1][base_relou_2.num_etude==101872].phrase_2.values
s=sommaire(liste)
print(s)
#ind_titres(base_relou_2)
a=light_cleaning(liste[s[0]+1]) #hypothèse donc que le 1er titre vient après le sommaire
print(a)
print(''.join([i for i in a]))
j=indice_relou(liste,s,0.4,0.4)
j

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
ouais=ind_titres(base_relou_2)
c=functools.reduce(
    operator.iconcat,ouais,[])
print(len(c))
base_relou_2['ind_sommaire']=c

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
numero=np.unique(base_relou_2.num_etude)
print("Il y a ",len(numero),"études vides")
k=9

titres=list(base_relou_2[base_relou_2.ind_sommaire==1][base_relou_2.num_etude==numero[k]].phrase_2.values)
for i in titres:
    print('\nIndice :',titres.index(i),"\n",i)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
numero=np.unique(base_relou_2.num_etude)
index=[i for i in numero if np.sum(base_relou_2[base_relou_2.num_etude==i].ind_sommaire)==0]
base_relou_3=base_relou_2[[True if base_relou_2.num_etude[i] in index else False for i in base_relou_2.index]]
print("Les documents vides tenaces représentent :",round(len(base_relou_3)/len(base_relou_2)*100,2),'% des lignes de la base relou')
print("On a réussi à traiter ",round((1-len(np.unique(base_relou_3.num_etude))/len(numero))*100,2),"% des documents vides initiaux")
print("Il y a ",len(np.unique(base_relou_3.num_etude)),"études vides")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def raffinement_sommaire(base_3,a,b):
    base=base_3.copy()
    c=[1 if len(i)>a or len(i)<b else 0 for i in base.phrase_2]#np.zeros(len(base))
    c_=np.zeros(len(base))
    for i in range(len(base)):
        try:
            k=int(base.phrase_2[i])
            #print(k)
            c_[i]=1
        except:
            continue
    base['sommaire_longueur']=c
    base['sommaire_int']=c_
    return base
base_=raffinement_sommaire(base_relou,180,3)
base_.sommaire_int=base_.sommaire_int*base_.ind_sommaire
base_.sommaire_longueur=base_.sommaire_longueur*base_.ind_sommaire
base_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_.iloc[:,3:]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import pickle
da=dataiku.Folder('lZ0B3sSL')
kmeans=pickle.load(open(da.get_path()+"/kmeans_model.pickle",'rb'))
label_k_relou=kmeans.predict(base_.iloc[:,3:-2])
base_['label_k']=label_k_relou

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
label_RF_relou=Bagging.predict(base_.iloc[:,3:-1])
base_['label_RF']=label_RF_relou

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
numero=np.unique(base_.num_etude)
k=2
titres=list(base_[base_.label_RF==1][base_.num_etude==numero[k]].phrase_2.values)
for i in titres:
    print('\nIndice :',titres.index(i),i)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import seaborn as sns
sns.histplot(distrib_nbligne_propre,stat='frequency',bins=100)