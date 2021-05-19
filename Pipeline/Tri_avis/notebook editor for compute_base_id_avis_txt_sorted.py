
# coding: utf-8

# # 0. Préparation des features

# In[ ]:



import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
base_id_avistxt = dataiku.Dataset("base_id_avistxt")
base_id_avistxt_df = base_id_avistxt.get_dataframe()


# In[ ]:


# Identification naïve des lignes qui sont quasiment certainement des avis non rendus
def target(row):
    score = row.hasAA + row.hasAO + row.hasDI + row.hasPO
    if score >=1:
        return(1)
    else:
        return(0)

base_id_avistxt_df['target'] = base_id_avistxt_df.apply(target,axis=1)
steps = 3

#Fonction longueur "créneau" aux limites arbitraires...
def creneau(rowlen,steps = 3,treshold = 4000, mini = 2000):
    x = np.linspace(mini,treshold,num = steps)
    k= 0
    for rng in x:
        if rowlen<=rng:
             return(k)
        else:
            k+=1
    return(k)

base_id_avistxt_df['creneauLen'] = base_id_avistxt_df['len'].apply(creneau,steps = steps)
base_id_avistxt_df['creneauLen'] = base_id_avistxt_df['creneauLen']/steps


# In[ ]:


# Longueur normalisée
from sklearn.preprocessing import MinMaxScaler

minmax_scale = MinMaxScaler().fit(base_id_avistxt_df[['len']])
base_id_avistxt_df['normLen'] = minmax_scale.transform(base_id_avistxt_df[['len']])


#imprimer un retour à la ligne pour une meilleur clarete de lecture
print('\n********** Normalisation*********\n')

print('Moyenne apres le Min max Scaling :\nMYCT={:.2f}'
.format(base_id_avistxt_df['normLen'].mean()))

print('\n')

print('Valeur minimale et maximale pour la feature apres min max scaling: \nMIN={:.2f}, MAX={:.2f}'
.format(base_id_avistxt_df['normLen'][:].min(), base_id_avistxt_df['normLen'][:].max()))


# # 1. Clustering des avis

# ## 1.1 KMeans sur les features ajoutées

# Repérage de quelques mots clés, longueur normalisée, longueur créneau

# In[ ]:


from sklearn.cluster import KMeans

X = pd.concat([base_id_avistxt_df.iloc[:,3:8]]+[base_id_avistxt_df.creneauLen],axis=1)
inst = KMeans(n_clusters = 3,random_state= 0,)
inst.fit(X)

res = pd.DataFrame(inst.labels_)
base_id_avistxt_df['Kres'] = res
base_id_avistxt_df.sort_values('len').head(200)


# ## 1.2 KMeans sur les features textuelles

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

vecto = TfidfVectorizer()
mat =vecto.fit_transform(base_id_avistxt_df.texte)

X2 = mat
inst2 = KMeans(n_clusters = 3,random_state= 0,)
inst2.fit(X2)

res2 = pd.DataFrame(inst2.labels_)
base_id_avistxt_df['Kres2'] = res2
base_id_avistxt_df.sort_values('len').head(200)


# ## 1.3 DBScan sur les features ajoutées

# In[ ]:


from sklearn.cluster import DBSCAN

X3 = pd.concat([base_id_avistxt_df.iloc[:,3:7],base_id_avistxt_df.creneauLen],axis=1)

params = [x/100 for x in range(1,31)]
clu = []
nois = []
for param in params:
    db = DBSCAN(eps=param, min_samples=len(X3)//15).fit(X3)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    base_id_avistxt_df['DBSCAN']  = db.labels_
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    clu.append(n_clusters_)
    n_noise_ = list(labels).count(-1)
    nois.append(n_noise_)

    #print('Estimated number of clusters: %d' % n_clusters_)
    #print('Estimated number of noise points: %d' % n_noise_)

params2 = [x for x in range(1,31)]
nois2 = []
clu2 = []
for param in params2:
    db = DBSCAN(eps=0.15, min_samples=len(X3)//param).fit(X3)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    base_id_avistxt_df['DBSCAN']  = db.labels_
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    clu2.append(n_clusters_)
    n_noise_ = list(labels).count(-1)
    nois2.append(n_noise_)

    #print('Estimated number of clusters: %d' % n_clusters_)
    #print('Estimated number of noise points: %d' % n_noise_)

import matplotlib.pyplot as plt
figure, axis = plt.subplots(2, 2)


axis[0, 0].plot(params,nois)
axis[0, 0].set_title("Noise wrto Eps")

axis[0, 1].plot(params,clu)
axis[0, 1].set_title("Num cluster wrto Eps")

axis[1, 0].plot(params2,nois2)
axis[1, 0].set_title("Noise wrto min_samples")

axis[1, 1].plot(params2,clu2)
axis[1, 1].set_title("Num cluster wrto min_samples")

plt.show()


# Sélection de quelques paramètres intéressants : eps, min_samples

# In[ ]:


db = DBSCAN(eps=0.15, min_samples=len(X3)//30).fit(X3)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
base_id_avistxt_df['DBSCAN']  = db.labels_+1
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

base_id_avistxt_df.sort_values('len').head(200)


# ## 1.4 Calcul du score de clustering

# Maintenant qu'on a les résultats de 3 clustering, on calcule un score de 0 (avis rendu) à 3 (avis non rendu). Les 0 et 3 constituent le set d'entrainement du classifieur car on considère que le clustering a bien fait son travail. Les autres sont un set de test.
# 
# L'interprétation faite des résultats est donnée dans le tableau suivant. On va transformer les colonnes pour que les résultats coincident.
Kres	Kres2	DBSCAN
  1	      0 	  1     Avis
  2	      2	      2 	Bruit
  0	      1	      0	    Non Avis
# In[ ]:


base_id_avistxt_df.replace(2,3,inplace = True )

base_id_avistxt_df.Kres.replace(0,2,inplace = True )
base_id_avistxt_df.Kres.replace(1,0,inplace = True )

base_id_avistxt_df.Kres2.replace(1,2,inplace = True )

base_id_avistxt_df.DBSCAN.replace(0,2,inplace = True )
base_id_avistxt_df.DBSCAN.replace(1,0,inplace = True )

base_id_avistxt_df.replace(3,1,inplace = True )

base_id_avistxt_df.sort_values('len').head(200)


# In[ ]:


#Calcul du score
base_id_avistxt_df['Score'] = (base_id_avistxt_df.Kres+base_id_avistxt_df.Kres2+base_id_avistxt_df.target+base_id_avistxt_df.DBSCAN)

#Constituion de la BDD des valeurs sûres :
#Non avis certains
df1 = base_id_avistxt_df[base_id_avistxt_df['Score']==7]
#Avis certains
df0 = base_id_avistxt_df[base_id_avistxt_df['Score']==0]
df_train = pd.concat([df1,df0])
df_train = df_train.sample(frac=1)
df_train['Score'] = df_train['Score']/7

#Constitution de la BDD des valeurs incertaines/test

df_test = base_id_avistxt_df[(base_id_avistxt_df['Score']>0)& (base_id_avistxt_df['Score']<7)]


# ## 1.5 Clustering sur clustering

# # 2. Classification supervisée

# ## 2.1 Entrainement sur le set d'avis correctement classés

# ### 2.1.1 Sur les features calculées

# In[ ]:


#Séparation des matrices de features/vecteur target. On classifie sur les indicatrices + la longueur normalisée
X_train,y_train = df_train.iloc[:,3:7],df_train.Score
X_test,y_test = df_test.iloc[:,3:7],df_test.Score

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state = 0)
rfc.fit(X_train,y_train)

#Les non avis sont classés 1, les avis 0
df_test['predict'] = rfc.predict(X_test)
df_test.sort_values('len').head(200)


# ### 2.1.2 Sur les features textuelles

# In[ ]:


#On classifie cette fois sur le texte
vecto2 = TfidfVectorizer(ngram_range = (1,2))
X_train2  =vecto2.fit_transform(df_train.texte)

X_test2 = vecto2.transform(df_test.texte)

rfc2 = RandomForestClassifier(random_state = 0)
rfc2.fit(X_train2,y_train)

#Les non avis sont classés 1, les avis 0
df_test['predict2'] = rfc2.predict(X_test2)
df_test.sort_values('len').head(200)


# Pour predict et predict2, on a 0 si l'avis est considéré comme rendu, 1 si l'avis est considéré comme non rendu. Il semble qu'il n'y ait que trop peu de features pour le premier classifieur : dès que la ligne d'indicatrices de contient pas de 1, l'avis est considéré comme rendu même si ça n'est pas le cas. Le 2eme classifieur (sur le texte) semble plus pertinent => on va conserver ces résultats.
# 
# Résultat très intéressant car le classifieur arrive à éliminer des textes longs, dont on pourrait s'attendre à ce qu'ils soient pertinent, mais qui sont en réalité des listes d'avis non rendus (cf avis 2639113) ou des documents qui ressemblent à des templates non complétés (avis 1299508).
# 
# En revanche quelques avis courts semblant présenter un intérêt sont également éliminés (avis 2061190)
# 
# Sur la "frontière", les avis semblent être bien classés la plupart du temps (ex : 374757 avis long de 4700 caractères bien classé en avis non rendu, alors que l'avis 115486 qui a presque exactement les mêmes caractériques est pertinent)

# ## 3. Création de la database finale

# On crée la database a partir des résultats de la classification supervisée sur les features textuelles

# In[ ]:


part1 = df0
part2 = df_test[df_test['predict2']==0]
base_id_avis_txt_sorted_df = pd.concat([df0,part2]).filter(['id','texte'],axis = 1)

base_id_avis_txt_sorted_df


# In[ ]:


initial = len(base_id_avistxt_df)
moyleninit = base_id_avistxt_df.len.mean()
reste = len(base_id_avis_txt_sorted_df)
moylenfin = (part1.len.sum()+part2.len.sum())/reste
medinit = base_id_avistxt_df.len.median()
medfin = pd.concat([part1,part2]).len.median()

lminav,lmaxav = base_id_avistxt_df.len.min(),base_id_avistxt_df.len.max()
lminap,lmaxap = min(part1.len.min(),part2.len.min()),max(part1.len.max(),part2.len.max())

avistacite = 49+275+65+0+188+65+13+112+61+52+138
avisrendu = 416+338+286+253+251+239+114+112+104+65+40

print(" Part d'avis réellement rendus : ", round(reste/initial*100,1),"%\n",
      "Stat du rapport annuel (cf message slack 15 Mars de Marc) : ",
      round((avisrendu)/(avisrendu+avistacite)*100,1),"% \n")
print("Longueur moyenne :",'\n',"Avant : ", round(moyleninit),'\n',"Après : ", round(moylenfin),'\n')
print("Longueur mediane :",'\n',"Avant : ", round(medinit),'\n',"Après : ", round(medfin))

print("\nLongueur max :", "\t \t Longueur min :"
      '\n',"Avant : ", lmaxav," \t  Avant : ",lminav,
      '\n',"Après : ", lmaxap," \t  Après : ",lminap)


# In[ ]:


# Write recipe outputs
base_id_avis_txt_sorted = dataiku.Dataset("base_id_avis_txt_sorted")
base_id_avis_txt_sorted.write_with_schema(base_id_avis_txt_sorted_df)

