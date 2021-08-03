#%%
from PyPDF2 import PdfFileReader
import os
import numpy as np
#%%
os.chdir('C:\\Users\\theo.roudil-valentin\\Documents\\Donnees\\PDF_EI')
fichiers=os.listdir()
fichiers=[i for i in fichiers if i[-3:]=='pdf']
# fichiers
print("Nombre de pdf :",len(np.unique(fichiers)))
# %%
from joblib import Parallel,delayed

def count_page(f):
    with open(f, "rb") as pdf_file:
        try:
            pdf_reader = PdfFileReader(pdf_file)
            return pdf_reader.numPages
        except:
            return np.nan

count_page(fichiers[0])
#%%
import psutil
cpu=psutil.cpu_count()
num_pages=Parallel(n_jobs=cpu)(delayed(count_page)(f) for f in fichiers)
# %%
import matplotlib.pyplot as plt

fig,ax=plt.subplots(figsize=(18,12))
ax.hist(num_pages)
ax.set_title('Distribution du nombre de pages par documents')
# %%
print("Nombre de pages ")
print("Moyenne :",round(np.nanmean(num_pages),2))
print("Ecart-type :",round(np.nanstd(num_pages),2))
print("Médiane :",round(np.nanmedian(num_pages),2))
# %%
os.chdir('C:\\Users\\theo.roudil-valentin\\Documents\\Codes\\2021-NLP_AE\\Data')
import pandas as pd
base=pd.read_csv('sections_cool_avecresume.csv',sep=";")
base
# %%
os.chdir('C:\\Users\\theo.roudil-valentin\\Documents\\Codes\\2021-NLP_AE\\Data\\Bagging_model')
import pandas as pd
base=pd.read_csv('base_classif_RF.csv',sep=";")
base
# %%
liste=[100689, 100707, 102316, 106168, 110277, 114799, 118071, 120638]
base=base[[True if i not in liste else False for i in base.num_etude]]
base[base.label_RF==1].phrase_2
# %%
liste_pascool=base.groupby(['num_etude'])['label_RF'].apply(lambda x : sum(x)).sort_values()[:10].index
base=base[[True if i in liste_pascool else False for i in base.num_etude]]
base
# %%
for i in base.phrase:
    print("\n",i,"\n")
#%%
ouais=base.groupby(['num_etude'])['label_RF'].apply(lambda x : sum(x))#.sort_values()[:50].index
ouais_2=ouais[(ouais>0)]
ouais_index=ouais_2[ouais_2<25].index
ouais_index
#%%
numero=np.unique(base.num_etude)
N=100
for i in liste_pascool:
    print("\n ################################################# \n Indice :",i,"\n")
    for k in base[base.num_etude==i].phrase[:N]:
        print('\n',k,'\n')
# %%
for k in ouais_index:
    print("########################'\n'Indice :",k)
    for i in base[(base.num_etude==k) and (base.label_RF==1)].phrase[:N]:
        print('\n',i,"\n")
# %%
ouais_index
# %%
for i in base[base.num_etude==219528].phrase[:N]:
    print('\n',i,"\n")
# %%
