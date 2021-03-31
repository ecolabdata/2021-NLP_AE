#%%
import pandas as pd
import numpy as np
import sklearn
import pickle
import re
import os
chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"
#%%
fichiers=os.listdir("C:/Users/theo.roudil-valentin/Documents/Donnees")
fichiers=[f for f in fichiers if f[-4:]=='.txt']
#%%
# filename = '229223 FEI.txt'
k=0
with open(chemin+fichiers[k],'r', encoding='utf-8') as f:
    f=f.read()
#%%
font=np.unique(re.findall('font\d\d',f[:10000]))
font.argmax()

from unidecode import unidecode
titres=np.unique(np.concatenate([np.array([c.split('>')[1].split('<')[0] for c in f.split(z)],dtype='object') for z in font]))
# index=[i for i in range(len(titres)) if titres[i].find('&')>0]
titres=[unidecode(c).replace('&nbsp;','') for c in titres]

# %%
subset_titres=[t for t in titres if len(t)>0 and re.search('\d',t[0]) is not None]

# %%
def TITRES(path,N=10000,encod='utf-8',expr='="font\d\d'):
    import numpy as np
    from unidecode import unidecode
    with open(path,'r', encoding=encod) as f:
        f=f.read()
    font=np.unique(re.findall(expr,f[:N]))

    titres=np.unique(
        np.concatenate(
            [np.array(
                [c.split('>')[1].split('<')[0] for c in f.split(z)],dtype='object') for z in font]))
    titres=[unidecode(c).replace('&nbsp;','') for c in titres]
    subset_titres=[t for t in titres if len(t)>0 and re.search('\d',t[0]) is not None]
    return titres,subset_titres

def TITRES_all(chemin,N=10000,encod='utf-8',expr='="font\d\d'):
    import os
    fichiers=os.listdir(chemin,)
    fichiers=[f for f in fichiers if f[-4:]=='.txt']
    titres={}
    subset_titres={}
    for k in range(len(fichiers)):
        titres[fichiers[k]],subset_titres[fichiers[k]]=TITRES(chemin+fichiers[k],N,encod,expr)
    return titres,subset_titres
# %%
titres,subset_titres=TITRES(chemin+fichiers[1])
# %%
titres,subset_titres=TITRES_all(chemin)
# %%
