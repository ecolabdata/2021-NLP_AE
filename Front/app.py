#%%
import streamlit as st
import numpy as np
import pandas as pd

# %%
st.title('Sommaire Augmenté')
df_resume = pd.read_csv(open("Data\\sections_cool_avecresume.csv",'rb'),sep=";")
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
def displayrow(row,enjeux_list = enjeux_list):
    titre = row.titres
    resume = ast.literal_eval(row.resume)
    enjeux = row.iloc[-8:]
    present = []
    k = 0
    for enj in enjeux:
        if enj:
            present.append(enjeux_list[k])
        k = k+1
    st.text(titre)
    if resume[0] != 'tropcourt':
        st.text('Résumé :')
        for k in resume:
            st.text(k)
    if len(present)!=0:
        st.text('Enjeux :')
        for enj in present:
            st.text(enj)
    
#%%

study.apply(displayrow,axis=1)

# %%
