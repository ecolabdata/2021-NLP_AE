################################################################################################
###  Code d'essai de création de variables des docs HTML  ##########################################################################
################################################################################################

# Ce code est une ébauche, je l'ai développé quand nous cherchions à créer les variables
# provenant des caractéristiques des docs HTML.
# L'idée est d'utiliser ces caractéristiques et de les encoder, par exemple dans des indicatrices.
# du type, si tel carac est là, mettre 1 sinon 0 etc...
# Il peut y avoir aussi des variables quantitatives (taille de la police)
# Nous avons au final pris le code de Ruben qui était meilleur 

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
########### généralisation ########################################################################################################################################################################
################################################################################################################################################################################
chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"

df_html=pd.read_csv(chemin+'base_html.csv')
#%%
def balise_html(fichiers):
    balise=[]
    from bs4 import BeautifulSoup
    for k in range(len(fichiers)):
        soup=BeautifulSoup(fichiers[k], "html.parser")
        balise.append(np.unique([i.name for i in soup.find_all()]))
    import functools
    import operator
    balise=np.unique(functools.reduce(operator.iconcat, balise, []))
    return balise

def sommaire(f,H=100000):
    print(len(f))
    from unidecode import unidecode
    import numpy as np
    ##### COUPE AU SOMMAIRE
    try:
        som=f.split('SOMMAIRE')[1]
    except:
        try:
            som=f.split('Sommaire')[1]
        except:
            try:
                som=f.split('sommaire')[1]
            except:
                print('On ne peut pas découper après le mot : sommaire')
                som=f
    longueur=round(len(som)/len(f)*100,2)

    if type(som)==str:
        print("Ce qui vient après le sommaire représente :",longueur,"% du document.")
        som=som[:H]
        print(len(som))
        bookmark=som.split('#bookmark')[1:-1]
        print(len(bookmark))
        if len(bookmark)>0:
            return bookmark
        else:
            bookmark=np.nan
            return bookmark
    else:
        print('Dommage.')
        bookmark=np.nan
        return bookmark

def indicatrice(bookmark,balise):
    import re
    if type(bookmark)==list:
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
        df.columns=['phrase','font','italic','bold','href','>\d']
        df_2=pd.DataFrame(donn_2)
        df_2.columns=balise
        data=pd.concat([df,df_2],axis=1)
        return data
    else:
        print("Petit problème pour la liste bookmark, êtes-vous certain que le découpage du sommaire a pu être effectué ?")

def base_classif_titres(df,balise,H=100000):
    donnees=[]
    fichier_nul=[]
    import pandas
    for i in df.index:
        print(i,df.num_etude[i])
        data=indicatrice(sommaire(df.texte[i],H),balise)
        if type(data)==pandas.core.frame.DataFrame:
            data['num']=df.num_etude[i]
            donnees.append(data)
        else:
            print("Le fichier numéro :",df.num_etude[i]," pose problème, probablement le découpage n'a pas pu être correctement effectué.")
            fichier_nul.append(i)
    base=pd.concat([donnees[i] for i in range(len(donnees))])
    return base,fichier_nul

# %%
base,fichier_nul=base_classif_titres(df_html,balise_html(df_html.texte))
# %%
