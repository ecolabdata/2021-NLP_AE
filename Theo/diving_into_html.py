#%%
import pandas as pd
import numpy as np
import sklearn
import pickle
import re
import os
chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/EI_txt/"
#%%
fichiers=os.listdir(chemin)
fichiers=[f for f in fichiers if f[-4:]=='.txt']
#%%
# filename = '229223 FEI.txt'

#%%
balise=[]
from bs4 import BeautifulSoup
for k in range(len(fichiers)):
    with open(chemin+fichiers[k],'r', encoding='utf-8') as f:
        soup=BeautifulSoup(f, "html.parser")
        balise.append(np.unique([i.name for i in soup.find_all()]))
#%%
# balise=np.unique(np.array(balise).flatten())
import functools
import operator
balise=np.unique(functools.reduce(operator.iconcat, balise, []))
#%%
################################################################################################################################################################################
########### TITRES AVEC FONT ########################################################################################################################################################################
################################################################################################################################################################################
k=4
with open(chemin+fichiers[k],'r', encoding='utf-8') as f:
    f=f.read()
# print(len(f))
from unidecode import unidecode
##### COUPE AU SOMMAIRE
sommaire=f.split('SOMMAIRE')[1]
print(len(sommaire))
# sommaire=sommaire[:100000]
print(len(sommaire))
#%%
###### Split pour les #Bookmark et prend l'intérieur
bookmark=sommaire.split('#bookmark')[1:-1]
#%%
###### Crée la matrice

#%%
donn=[[i,
    int(re.search('font\d\d',i).group()[-2:]) if re.search('font\d\d',i) is not None else 0,
    int(re.search('italic',i) is not None),
    int(re.search('bold',i) is not None),
    int(re.search('href',i) is not None),
    int(re.search('>\d',i) is not None),] for i in bookmark]
#%%
donn_2=[[int(re.search(h,i) is not None) for h in balise] for i in bookmark]
#%%
df=pd.DataFrame(donn)
df.columns=['html','font','italic','bold','href','>\d']
df_2=pd.DataFrame(donn_2)
df_2.columns=balise
data=pd.concat([df,df_2],axis=1)
data
#%%
from sklearn.cluster import KMeans
kmeansmodel=KMeans(n_clusters=2,n_init=20,max_iter=500)
kmeans=kmeansmodel.fit(data.iloc[:,1:])
data['label']=kmeans.labels_
#%%
font=np.unique(re.findall('font\d\d',sommaire))
# font=[i for i in font if int(i[-2:])>=20]
#%%
titres=np.unique(
    np.concatenate(
        [np.array(
            [c.split('>')[1].split('<')[0] for c in sommaire.split(z)],dtype='object') for z in font]))
# index=[i for i in range(len(titres)) if titres[i].find('&')>0]
titres=[unidecode(c).replace('&nbsp;','') for c in titres]

# %%
# subset_titres=[t for t in titres if len(t)>0 and (re.search('\d',t[0]) is not None or re.search('I',t[0]) is not None)]
# subset_titres
# subset_titres=[i for i in subset_titres if ]
# %%
def TITRES(path,N=10000,encod='utf-8',expr='="font\d\d'):
    import numpy as np
    from unidecode import unidecode
    with open(path,'r', encoding=encod) as f:
        f=f.read()
    font=np.unique(re.findall(expr,f[:N]))
    font=[i for i in font if int(i[-2:])>=20]

    titres=np.unique(
        np.concatenate(
            [np.array(
                [c.split('>')[1].split('<')[0] for c in f.split(z)],dtype='object') for z in font]))
    titres=[unidecode(c).replace('&nbsp;','') for c in titres]
    subset_titres=[t for t in titres if len(t)>0 and (re.search('\d',t[0]) is not None or re.search('I',t[0]) is not None)]
    return titres,subset_titres

def TITRES_all(chemin,N=10000,encod='utf-8',expr='="font\d\d'):
    import os
    fichiers=os.listdir(chemin)
    fichiers=[f for f in fichiers if f[-4:]=='.txt']
    titres={}
    subset_titres={}
    for k in range(len(fichiers)):
        titres[fichiers[k]],subset_titres[fichiers[k]]=TITRES(chemin+fichiers[k],N,encod,expr)
    return titres,subset_titres
# %%
titres,subset_titres=TITRES(chemin+fichiers[4])
titres
# %%
titres,subset_titres=TITRES_all(chemin)
# %%
################################################################################################################################################################################
########### TITRES AVEC LIST ########################################################################################################################################################################
################################################################################################################################################################################

from unidecode import unidecode
titres=np.unique(
    np.array(f.split("<ul"),dtype='object'))
titres2=np.unique(
    np.array(f.split("</ul"),dtype='object'))
# %%

font=np.unique(re.findall('font\d\d',f[:10000]))
font=[i for i in font if int(i[-2:])>=20]

from unidecode import unidecode
titres=np.unique(
    np.concatenate(
        [np.array(
            [c.split('<ul') for c in f.split(z) if len(c)>0],dtype='object') for z in font]))
# index=[i for i in range(len(titres)) if titres[i].find('&')>0]
# titres=[unidecode(c).replace('&nbsp;','') for c in titres]

# %%
