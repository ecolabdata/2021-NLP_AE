#%%
import pandas as pd
import numpy as np
import sklearn
import pickle
import re

chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"
fichiers=os.listdir(chemin+'EI_txt')
fichiers=[f for f in fichiers if f[-4:]=='.txt']

base={}
for k in range(len(fichiers)):
    with open(chemin+"EI_txt/"+fichiers[k],'r', encoding='utf-8') as f:
        base[k]=[fichiers[k][:-8],f.read()]
base_txt=pd.DataFrame.from_dict(base).T
base_txt.columns=['num_etude','texte']
# %%
chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"
fichiers=os.listdir(chemin+'EI_html')
fichiers=[f for f in fichiers if f[-5:]=='.html']
fichiers

base={}
for k in range(len(fichiers)):
    with open(chemin+"EI_html/"+fichiers[k],'r', encoding='utf-8') as f:
        base[k]=[fichiers[k][:-9],f.read()]
base_html=pd.DataFrame.from_dict(base).T
base_html.columns=['num_etude','texte']
# %%
pickle.dump(base_html,open(chemin+'base_html.pickle','wb'))
pickle.dump(base_txt,open(chemin+'base_txt.pickle','wb'))

# %%
chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"
fichiers=os.listdir(chemin+'EI_html')
fichiers=[f[:-5] for f in fichiers if f[-5:]=='.html']
fichiers_html=fichiers
chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"
fichiers=os.listdir(chemin+'PDF_EI')
fichiers=[f[:-4] for f in fichiers if f[-4:]=='.pdf']
fichiers_pdf=fichiers
pas_bon=[i for i in fichiers_pdf if i not in fichiers_html]
# %%
