
# coding: utf-8

# In[ ]:



import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import os
import codecs
# Read recipe inputs
avis_txt_projets_env = dataiku.Folder("YoeTW9Sf")
path = dataiku.Folder("YoeTW9Sf").get_path()
avis_txt_projets_env_info = avis_txt_projets_env.get_info()


# In[ ]:


ids = []
texts = []
for file in avis_txt_projets_env.list_paths_in_partition():
    id = ''
    for car in file:
        if car in '0123456789':
            id += car
    ids.append(id)
    with codecs.open(path+file) as f:
        data = f.read()
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
np.sum(sortie)

base_id_avistxt_df.sort_values('len').head(20)


# In[ ]:


# Write recipe outputs
base_id_avistxt = dataiku.Dataset("base_id_avistxt")
base_id_avistxt.write_with_schema(base_id_avistxt_df)

