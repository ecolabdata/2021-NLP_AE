#%%
import streamlit as st
import numpy as np
import pandas as pd

# %%
st.title('Sommaire Augmenté')
df_resume = pd.read_csv(open("Data\sections_cool_avecresume.csv",'rb'),sep=";")
df_resume.rename(columns = {'num_etude':'id_AAE'},inplace =True)
df_enjeux = pd.read_csv(open("Data\Workinprogress\section_test.csv",'r'),sep=";")

#%%
df_enjeux = df_enjeux[df_enjeux.id_AAE != 'True']
df_enjeux['id_AAE'] = df_enjeux.id_AAE.astype(int)

#df_keywords = pd.read_csv()

#%%
df = pd.merge(df_resume,df_enjeux,on=['id_AAE','titres'])

num_etudes = np.unique(df.id_AAE.tolist())
#%%
option = st.selectbox(
    'Quelle étude souhaitez-vous analyser ?',
     num_etudes)

'You selected: ', option

study = df[df['id_AAE']==option]
enjeux_list = study.columns[-8:]
import ast
from streamlit_tags import st_tags

def displayrow(row,enjeux_list = enjeux_list):
    idx = row[0]
    titre = row.titres
    resume = ast.literal_eval(row.resume)
    enjeux = row.enjeux
    present = []
    #keywords = row.keywords
    col1,col2 = st.beta_columns((1,15))
    col2.markdown('## ' + titre)
    edit = col1.checkbox('' ,key=idx,value=True)
    if  edit:
        present_obj = st_tags(
        label='#### Enjeux:',
        text='Entrée pour ajouter un enjeu',
        value= present,
        suggestions=enjeux_list,
        maxtags = len(enjeux_list),
        key=idx)
        if resume[0] != 'tropcourt':
            st.markdown('#### Résumé :')
            for k in resume:
                st.markdown('##### '+ k)
    # if keywords != []:
    #     st.markdown('#### Mots clef :')
    #     for kw in keywords:
    #         st.markdown('#####' + kw)
#%%

study.apply(displayrow,axis=1)

# %%