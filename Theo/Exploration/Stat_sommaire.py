################################################################################################
###  Quelques statistiques sur les sommaires détectés  ##########################################################################
################################################################################################



#%%
import gensim
import pandas as pd
import numpy as np
import sklearn
import re
from unidecode import unidecode
import functools
import operator
import psutil
import pickle
from joblib import Parallel,delayed
from functools import partial
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import os
#%%
os.chdir('C:/Users/theo.roudil-valentin/Documents/Codes/2021-NLP_AE')
# %%
df_sec=pickle.load(open('Data/Bagging_model/df_sections.pickle','rb'))
# %%
numero=np.unique(df_sec.num_etude)
print("148 études au total\n 7 études n'ont pas de sommaires")
print("Proportion d'études dont un sommaire a été détecté",round((len(numero)/(148-7))*100,2),"%")
# %%
longueur_sec=[len(df_sec[df_sec.num_etude==i]) for i in numero]
longueur_sec
# %%
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

m=stat_des(longueur_sec,'mean')
s=stat_des(longueur_sec,'std')
me=stat_des(longueur_sec,'med')
print("La moyenne du nombre de sections découpées est :",m,"\nL'écart-type est:",s,"\nLa médiane est:",me)
# %%
import matplotlib.pyplot as plt
fig,ax=plt.subplots(3,figsize=(12,6))
ax[0].hist(longueur_sec,density=True,bins=100)
ax[1].hist([i for i in longueur_sec if i<400],density=True,bins=50)
ax[2].hist([i for i in longueur_sec if i<175],density=True,bins=50)

for i in range(3):
    ax[i].axvline(m,color='red',linestyle='--',label='moyenne'),ax[i].axvline(me,color='purple',linestyle='--',label='médiane')
    ax[i].set_xlabel('Nombres de sections extraites'),ax[i].set_ylabel('Fréquence')
plt.legend(['moyenne','médiane'])
# %%
print("Parmi les études dont le sommaire est détecté :")
inf_100=[i for i in longueur_sec if i<100 and i>25]
print("Il y a ",round(len(inf_100)/len(numero)*100,2),"% études qui comportent moins de 100 titres mais plus de 25.")
inf_25=[i for i in longueur_sec if i<=25 and i >0]
print("Il y a ",round(len(inf_25)/len(numero)*100,2),"% études qui comportent moins de 25 titres.")
sup_100=[i for i in longueur_sec if i>100]
print("Il y a ",round(len(sup_100)/len(numero)*100,2),"% études qui comportent plus de 100 titres.")
# egal_0=[i for i in longueur_sec if i==0]
# print("Il y a ",len(egal_0),"études qui ne comportent aucun titres.")
# %%
