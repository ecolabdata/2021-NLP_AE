# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Notebook pour le clustering non-supervisé des titres
Ce code est divisé en deux parties :
## 1. Ajout, affinage des features
## 2. Algorithme(s) de clustering

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# D'abord on appelle la base :  n'hésitez pas à décommenter la ligne pour n'en avoir qu'une partie (plus rapide pour visualiser) en faisant varier N

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import time

# Read recipe inputs
start=time.time()
base_phrase_clean = dataiku.Dataset("base_phrase_clean")
#N=200000
#base_phrase_clean_df = base_phrase_clean.get_dataframe(sampling='head',limit=N)
base_phrase_clean_df = base_phrase_clean.get_dataframe()
end=time.time()
print('Durée :',round((end-start)/60,2),' minutes')
print('Il y a ',len(np.unique(base_phrase_clean_df.num_etude)),"études")
base_phrase_clean_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 1. Ajout et affinage des features

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### On rajoute les features :
* nombre de mots,
* nombre de caractères
* mais également nombre de caractères de la ligne html

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from bs4 import BeautifulSoup
base_phrase_clean_df['phrase_2']=[BeautifulSoup(i,"html.parser").text for i in base_phrase_clean_df.phrase]
base_phrase_clean_df['f_mots']=[len(i.split(' ')) for i in base_phrase_clean_df.phrase_2]
base_phrase_clean_df['f_carac']=[len(i) for i in base_phrase_clean_df.phrase_2]
base_phrase_clean_df['html_carac']=[len(i) for i in base_phrase_clean_df.phrase]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_phrase_clean_df.shape

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### On rajoute indicatrice sommaire

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
    #print(indexes)
    try:
        interval=[verif_theo.num_etude[index] for index in sorted(indexes)]
    except:
        verif_theo=verif_theo.reset_index(drop=True)
        interval=[verif_theo.num_etude[index] for index in sorted(indexes)]
    #print(interval)
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
            print("L'indice de début de sommaire pour l'étude est",indice_sommaire[0],a)
            for j in range(indice_sommaire[0]+1+1,len(liste)):
                b=light_cleaning(liste[j])
                if a==b:
                    vector_[interval.index(i)][indice_sommaire[0]:j]=1
                    break
                else:
                    continue
            if np.sum(vector_[interval.index(i)])>0:
                print("Il y a un sommaire pour l'étude.")
            elif np.sum(vector_[interval.index(i)])==0:
                print('On tente la nouvelle méthode.')
                j=indice_relou(liste,indice_sommaire,seuil,seuil_c)
                if j is not None:
                    vector_[interval.index(i)][indice_sommaire[0]:j]=np.ones(len(range(indice_sommaire[0],j)))
                    ss=np.sum(vector_[interval.index(i)])
                    if ss>seuil_sommaire:
                        print("Il y a un sommaire pour l'étude.")
                    elif (ss>0) and (ss<seuil_sommaire) and (brutal is not None):
                        print("Hum... Le sommaire semble petit :",ss)
                        vector_[interval.index(i)][indice_sommaire[0]:brutal]=np.ones(len(range(indice_sommaire[0],brutal)))
                    else:
                        print("Pas de sommaire pour l'étude.")
                else:
                    print("Pour l'étude",i,"on n'a pas trouvé d'indice pour débuter le sommaire.")
        else:
            print("Pas de sommaire pour l'étude.")
    return vector_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#start=time.time()
#ouais=ind_titres(base_phrase_clean_df)
#end=time.time()
print("L'indicatrice a pris",round((end-start)/60),"minutes.")
import functools
import operator
c=functools.reduce(
    operator.iconcat,ouais,[])
#print(len(c))
base_phrase_clean_df['ind_sommaire']=c

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
numero=np.unique(base_phrase_clean_df.num_etude)
num_relou=[i for i in numero if np.sum(base_phrase_clean_df[base_phrase_clean_df.num_etude==i].ind_sommaire)==0]
index=[True if base_phrase_clean_df.num_etude[i] in num_relou else False for i in base_phrase_clean_df.index]
base_relou=base_phrase_clean_df[index]
print("Les documents vides représentent :",round(len(base_relou)/len(base_phrase_clean_df)*100,2),'% des lignes de la base initiale')
print("On a réussi à traiter ",round((1-len(np.unique(base_relou.num_etude))/len(numero))*100,2),"% des documents.")
print("Il y a ",len(np.unique(base_relou.num_etude)),"études vides")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_phrase_clean_df[base_phrase_clean_df.ind_sommaire==1].phrase_2

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Après avoir analysé les résultats en bout de pipeline : on a intérêt à bien travailler cette variable, beaucoup de gains possibles à travers elle.

On va donc raffiner cette variable.

De plus, en sortant les phrases liées à l'indicatrice de sommaire, on voit que celle-ci ne peut pas être utilisée comme règle directe, car il y a beaucoup de déchets, plus qu'après tout le pipe (bien qu'il soit assez conséquent, bien que peu couteux en termes de calcul)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
titres=base_phrase_clean_df[base_phrase_clean_df.ind_sommaire==1].phrase_2
for i in titres.index:
    print(titres[i])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
len(base_.sommaire_int*base_.ind_sommaire)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def raffinement_sommaire(base,a,b):
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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
start=time.time()
base_=raffinement_sommaire(base_phrase_clean_df,180,3)
base_.sommaire_int=base_.sommaire_int*base_.ind_sommaire
base_.sommaire_longueur=base_.sommaire_longueur*base_.ind_sommaire
base_
end=time.time()
print("Durée",round((end-start)/60),"minutes.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for i in base_[base_.sommaire_int>0].phrase_2.index:
    print(titres[i])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for i in base_[base_.sommaire_longueur==1].phrase_2.index:
    print(titres[i])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Gestion des caractères spéciaux

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def caractere_spec(base,weird=["□","■"]):
    base['caractere_spec']=[1 if np.mean([int(weird[i] in base.phrase_2[K]) for i in range(len(weird))])>0 else 0
 for K in range(len(base.phrase_2))]
    return base
start=time.time()
base=caractere_spec(base_)
end=time.time()
print("Durée :",round((end-start)/60),"minutes.")
base[base.caractere_spec==1]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Il y a",round((len(base[base.caractere_spec==1])/len(base))*100,2),"% de lignes dans la base avec des caractères spéciaux.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### On a notre matrice d'entraînement
attention à bien gérer les indices au niveau des colonnes pour prendre les bonnes !

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
train=base_phrase_clean_df.iloc[:,2:]
train=pd.concat([train.iloc[:,:-8],train.iloc[:,-7:]],axis=1)
train

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### On va maintenant faire une PCA sur la partie creuse de la matrice puis standardiser

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### Ensuite une PCA sur le ventre creux de la matrice

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
pca=PCA(3)
p=41
#pca.fit(train_.iloc[:,1:p])
#print(pca.explained_variance_ratio_)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Variance expliquée :",round(pca.explained_variance_ratio_.sum()*100,2),"%")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
ordo_p={}
P=range(31,42)
for p in P:
    ordo=[]
    for i in range(1,15):
        pca=PCA(i)
        pca.fit(train_.iloc[:,1:p])
        ordo.append(pca.explained_variance_ratio_.sum())
    ordo_p[p]=ordo

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
len(ordo)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for i in P:
    print(len(ordo_p[i]))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import matplotlib.pyplot as plt
f,a=plt.subplots(1,figsize=(14,8))
absi=range(14)
for i in P:
    a.plot(absi,ordo_p[i])
a.set(xlabel="Nombre de variables",ylabel='Variance expliquée',
      title="Variance expliquée en fonction du nombre d'axes de la PCA")
plt.legend([p for p in P])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# On choisit donc p=33 et i=14

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
pca=PCA(12)
ventre=pca.fit_transform(train_.iloc[:,1:33])
print(ventre.shape)
ventre

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
ventre=pd.DataFrame(ventre,columns=['ventre_'+str(i) for i in range(ventre.shape[1])])
train__=pd.concat([train_.iloc[:,0],pd.DataFrame(ventre),train_.iloc[:,33:]],axis=1)
train__

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
len(np.unique(train.iloc[:,0].values))

def scaling(k,k_min,k_max):
    return (k-k_min)/(k_max-k_min)

def scaler(X,z=10):
    from joblib import Parallel,delayed
    from functools import partial
    X_min=min(X)
    X_max=max(X)
    scal=partial(scaling,k_min=X_min,k_max=X_max)
    X_=Parallel(n_jobs=z)(delayed(scal)(X[i]) for i in range(len(X)))
    return X_

dico_scale={}
train_=train__.copy()
start=time.time()
for i in range(train.shape[1]):
    if len(np.unique(train.iloc[:,i].values))>2:
        dico_scale[i]=scaler(train.iloc[:,i])
        train_.iloc[:,i]=dico_scale[i]
    else:
        continue
end=time.time()
print("Durée :",round((end-start)/60,2),"minutes")

dico_scale.keys()

train_.iloc[:,1:41]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### On vire les colonnes redondantes
A priori aucune après ce nouveau processing

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def pandas_col_linear_dep(A):
    corr=[]
    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            a=np.corrcoef(A.iloc[:,i],A.iloc[:,j])[1][0]
            if (a==1) and (i!=j):
                corr.append([i,j])
    #a=[(np.corrcoef(A.iloc[:,i],A.iloc[:,j])[1][0],i,j) for i in A.shape[1] for j in A.shape[1] if
        #(i!=j) and (np.corrcoef(A.iloc[:,i],A.iloc[:,j])[1][0]==1)]
    return corr

start=time.time()
corr=pandas_col_linear_dep(train__)
end=time.time()
print('Durée :',round((end-start)/60,2),' minutes')

corr

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
if len(corr)>0:
    for i in corr[0:int(len(corr)/2)]:
        try:
            train__=train__.drop([str(i[1])],axis=1)
        except:
            continue

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
train__

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 2. Clustering non-supervisé
On peut faire varier le nombre de cluster, mais pour le moment, je l'ai fixé à 2 : conceptuellement les titres et le reste. Mais en termes machine cela peut être différent, on souhaite que la machine le perçoive comme nous. L'ajout d'un cluster peut être intéressant et aider à travailler sur les phrases qui se rapprochent des titres à cause des problèmes d'OCRisation ou autres par exemple

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
n_cluster=2

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.cluster import KMeans
kmeansmodel=KMeans(n_clusters=n_cluster,n_init=100,max_iter=500)
start=time.time()
kmeans=kmeansmodel.fit(train__)
end=time.time()
print('Durée :',round((end-start)/60,2),' minutes')
train__['label_k_2']=kmeans.labels_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import pickle
da=dataiku.Folder('lZ0B3sSL')
pickle.dump(kmeans,open(da.get_path()+"/kmeans_model_2.pickle",'wb'))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import pickle
da=dataiku.Folder('lZ0B3sSL')
#pickle.dump(kmeans,open(da.get_path()+"/kmeans_model.pickle",'wb'))
kmeans_model=pickle.load(open(da.get_path()+"/kmeans_model.pickle",'rb'))
train__['label_k_1']=kmeans.labels_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Ne pas faire tourner les clusterings suivants

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.cluster import SpectralClustering
sc = SpectralClustering(n_cluster,  n_init=100,
                            assign_labels='discretize')
start=time.time()
sc.fit(train.iloc[:,:-1])
end=time.time()
print('Durée :',round((end-start)/60,2),' minutes')
train['label_sc']=sc.labels_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.cluster import DBSCAN
start=time.time()
clustering = DBSCAN(eps=0.1, min_samples=100).fit(train.iloc[:,:-1])
end=time.time()
print('Durée :',round((end-start)/60,2),' minutes')
train['label_dbs']=clustering.labels_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# On réorganise la base et on la sauvegarde

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
train__

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#train__.columns=["0"]+['ventre_'+str(i) for i in range(ventre.shape[1])]+list(train__.columns[15:].values)
df=pd.concat([base.iloc[:,:2],base.loc[:,'phrase_2'],train__],axis=1)
df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_classif_Kmeans_1_df = df # For this sample code, simply copy input to output
da=dataiku.Folder('lZ0B3sSL')
pickle.dump(base_classif_Kmeans_1_df,open(da.get_path()+"/base_pour_Bagging_final.pickle",'wb'))


# Write recipe outputs
base_classif_Kmeans_1 = dataiku.Dataset("base_classif_Kmeans_1")
base_classif_Kmeans_1.write_with_schema(base_classif_Kmeans_1_df)