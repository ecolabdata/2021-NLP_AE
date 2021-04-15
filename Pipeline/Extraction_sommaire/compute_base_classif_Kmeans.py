# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import time

# Read recipe inputs
start=time.time()
base_phrase_clean = dataiku.Dataset("base_phrase_clean")
base_phrase_clean_df = base_phrase_clean.get_dataframe(sampling='head',limit=10000)
end=time.time()
print('Durée :',round((end-start)/60,2),' minutes')
base_phrase_clean_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### On rajoute les features :
# * nombre de mots,
# * nombre de caractères
# * mais également nombre de caractères de la ligne html

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from bs4 import BeautifulSoup
base_phrase_clean_df['phrase_2']=[BeautifulSoup(i,"html.parser").text for i in base_phrase_clean_df.phrase]
base_phrase_clean_df['f_mots']=[len(i.split(' ')) for i in base_phrase_clean_df.phrase_2]
base_phrase_clean_df['f_carac']=[len(i) for i in base_phrase_clean_df.phrase_2]
base_phrase_clean_df['html_carac']=[len(i) for i in base_phrase_clean_df.phrase]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_phrase_clean_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### On rajoute indicatrice sommaire

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def sommaire(s):
    somm=['SOMMAIRE','sommaire','Sommaire']
    s=list(s)
    indice=[s.index(k) for k in somm if k in s]
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


def ind_titres(verif):
    verif_theo=verif.copy()
    dico_titres={}
    titres={}
    vector_=[]
    indexes = np.unique(verif_theo.num_etude, return_index=True)[1]
    interval=[verif_theo.num_etude[index] for index in sorted(indexes)]
    for i in interval:
        liste=verif_theo[verif_theo.num_etude==i].phrase_2.values
        vector=np.zeros(len(liste))
        indice_sommaire=sommaire(liste)
        if len(indice_sommaire)>0:
            print(indice_sommaire)
            a=light_cleaning(liste[indice_sommaire[0]+1]) #hypothèse donc que le 1er titre vient après le sommaire
            print(a)
            for j in range(indice_sommaire[0]+1+1,len(liste)):
                b=light_cleaning(liste[j])
                if a==b:
                    dico_titres[i]=[j,liste[j]]
                    print(j)
                    print(dico_titres[i])
                    vector[indice_sommaire[0]:j]=1
                    vector_.append(vector)
                    break
                else:
                    continue
        else:
            dico_titres[i]=['pas de "sommaire" ?']
            vector_.append(vector)
    return vector_

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_phrase_clean_df['ind_sommaire']=functools.reduce(
    operator.iconcat,
    ind_titres(base_phrase_clean_df), [])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_phrase_clean_df[base_phrase_clean_df.ind_sommaire==1].phrase_2

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Gestion des caractères spéciaux

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def caractere_spec(base,weird=["□","■"]):
    base['caractere_spec']=[1 if np.mean([int(weird[i] in base.phrase_2[K]) for i in range(len(weird))])>0 else 0
 for K in range(len(base.phrase_2))]
    return base
base=caractere_spec(base_phrase_clean_df)
base[base.caractere_spec==1]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("Il y a",round((len(base[base.caractere_spec==1])/len(base))*100,2),"% de lignes dans la base avec des caractères spéciaux.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### On a notre matrice d'entraînement
# attention à bien gérer les indices au niveau des colonnes pour prendre les bonnes !

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
train=base_phrase_clean_df.iloc[:,2:]
train=pd.concat([train.iloc[:,:-7],train.iloc[:,-5:]],axis=1)
train

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### On vire les colonnes redondantes

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
corr=pandas_col_linear_dep(train)
end=time.time()
print('Durée :',round((end-start)/60,2),' minutes')

corr

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for i in corr[0:int(len(corr)/2)]:
    try:
        train=train.drop([str(i[1])],axis=1)
    except:
        continue

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
train

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### On passe au clustering non-supervisé

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
n_cluster=2

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.cluster import KMeans
kmeansmodel=KMeans(n_clusters=n_cluster,n_init=100,max_iter=500)
start=time.time()
kmeans=kmeansmodel.fit(train)
end=time.time()
print('Durée :',round((end-start)/60,2),' minutes')
train['label_k']=kmeans.labels_

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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df=pd.concat([base.iloc[:,:2],base.loc[:,'phrase_2'],train,base.loc[:,'target']],axis=1)
df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_classif_Kmeans_1_df = df # For this sample code, simply copy input to output


# Write recipe outputs
base_classif_Kmeans_1 = dataiku.Dataset("base_classif_Kmeans_1")
base_classif_Kmeans_1.write_with_schema(base_classif_Kmeans_1_df)