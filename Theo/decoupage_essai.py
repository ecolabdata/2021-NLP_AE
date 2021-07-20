#%%
import pandas as pd, numpy as np
from bs4 import BeautifulSoup
import os
import unicodedata
import string
import time
import lxml
from tqdm import tqdm
import pickle
from bs4 import BeautifulSoup
import re
from unidecode import unidecode
# Read recipe inputs
os.chdir('C:/Users/theo.roudil-valentin/Documents/Codes')
base_classif_RF=pd.read_csv("2021-NLP_AE/Data/Bagging_model/base_classif_RF.csv",sep=';')#,nrows=100000)
# base_classif_RF = pickle.load(open("2021-NLP_AE/Data/Bagging_model/Base_label_RF.pickle",'rb'))
# label_rf_2=pickle.load(open("2021-NLP_AE/Data/Bagging_model/label_RF_2.pickle",'rb'))
# label_rf_2.columns=['num_etude','phrase_2','label_rf_2']
# label_rf_3=pickle.load(open("2021-NLP_AE/Data/Bagging_model/label_RF_3.pickle",'rb'))
# label_rf_3.columns=['num_etude','phrase_2','label_rf_3']
# base_classif_RF=pd.concat([base_classif_RF,label_rf_2.label_rf_2,label_rf_3.label_rf_3],axis=1)
base_classif_RF['longueur']=[len(i) if type(i)!=float else 0 for i in base_classif_RF.phrase_2]
base_classif_RF
# %%
seuil=5
base_classif_RF_1=base_classif_RF[(base_classif_RF.label_RF==1) & (base_classif_RF.longueur>seuil)]
# base_classif_RF_1
chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/base_html.csv"
df_html=pd.read_csv(chemin)
# %%
liste_cool=[100689,100707,102316,106168,110277,114799,118071,120638]
titres=base_classif_RF_1[base_classif_RF_1.num_etude==liste_cool[0]].phrase.values
print(titres[0])
titres_2=titres=base_classif_RF_1[base_classif_RF_1.num_etude==liste_cool[0]].phrase_2.values

texte_brut=df_html[df_html.num_etude==liste_cool[0]].texte.values[0]
# print(texte[:1000])
texte=BeautifulSoup(texte_brut,'html.parser').get_text()
# print(texte[:1000])
texte=unidecode(texte).replace('\n','').replace("\'",'')
# print(texte[:1000])
texte
#%%
longueur_bb=[[len(i) for i in texte_brut.split(k)] for k in titres]
longueur_bc=[[len(i) for i in texte_brut.split(BeautifulSoup(k,'html.parser').get_text())] for k in titres]
longueur_cc=[[len(i) for i in texte.split(unidecode(BeautifulSoup(k,'html.parser').get_text()))] for k in titres]
titres_2_bis=[i.replace('\n','') for i in titres_2]
longueur_cc_2=[[len(i) for i in texte.split(k)] for k in titres_2_bis]

for i in range(len(titres)):
    print("Indice :",i,longueur_bb[i],longueur_bc[i],longueur_cc[i],longueur_cc_2[i])
# %%
