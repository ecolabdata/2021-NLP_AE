
# coding: utf-8

# In[ ]:
#Root : 2021-NLP_AE. Dossier Data avec toutes les bases

import pandas as pd, numpy as np
import os
import codecs
# Read recipe inputs£
path = "Data\Avis_txt\\"
listfiles = os.listdir(path)


# In[ ]:
import io

ids = []
texts = []
for file in listfiles:
    if file[-3:] != 'txt':
        pass
    id = ''
    for car in file:
        if car in '0123456789':
            id += car
    ids.append(id)
    with open(path+file,'rb') as f:
        data = f.read().decode('utf-8','ignore')
        texts.append(data)


# In[ ]:


base_id_avistxt_df = pd.DataFrame({'id':ids,'texte':texts})

base_id_avistxt_df['texte'].replace('',np.nan,inplace = True)
base_id_avistxt_df.dropna(subset=['texte'], inplace=True)


#On ajoute la longueur de chaque chaine
base_id_avistxt_df['len']=base_id_avistxt_df['texte'].str.len()

def hasAA(text):
    text = text.lower()
    if "absence d'avis" in text:
        return(1)
    else:
        return(0)

def hasAO(text):
    text = text.lower()
    if "absence d'observation" in text:
        return(1)
    else:
        return(0)

def hasDI(text):
    text = text.lower()
    if "délai imparti" in text:
        return(1)
    else:
        return(0)

def hasPO(text):
    text = text.lower()
    if "pas d'observation" in text:
        return(1)
    else:
        return(0)

sortie = base_id_avistxt_df['texte'].apply(hasAA)
sortie2 = base_id_avistxt_df['texte'].apply(hasAO)
sortie3 = base_id_avistxt_df['texte'].apply(hasDI)
sortie4 = base_id_avistxt_df['texte'].apply(hasPO)
base_id_avistxt_df['hasAA'] = sortie
base_id_avistxt_df['hasAO'] = sortie2
base_id_avistxt_df['hasDI'] = sortie3
base_id_avistxt_df['hasPO'] = sortie4



# In[ ]:
import pickle

path = "Data\Workinprogress\\"
# Write recipe outputs
base_id_avistxt = pickle.dump(base_id_avistxt_df,open(path+"base_id_avistxt.pickle",'wb'))



# %%
