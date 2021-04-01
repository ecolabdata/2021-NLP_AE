#%%
import pandas as pd, numpy as np
import bs4
import requests
from tqdm import tqdm
import re,json
from unidecode import unidecode
import html
# from requests_html import HTMLSession
from html.parser import HTMLParser
#%%
chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"
df=pd.read_csv(chemin+"projets-environnement-diffusion.csv",sep=None)
df_clos=df[df[df.columns[6]]=='clos']

# url='https://www.projets-environnement.gouv.fr/explore/dataset/projets-environnement-diffusion/export/?disjunctive.dc_subject_category&disjunctive.dc_subject_theme&disjunctive.vp_status&disjunctive.dc_type&sort=recordsid'
# url='https://www.google.com/search'
he={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:87.0) Gecko/20100101 Firefox/87.0'}
proxy={'http':'http://cache.ritac.i2:32000',
'https':'http://cache.ritac.i2:32000'
}
#%%
response=requests.get(url,headers=he,proxies=proxy)
response.close()
# resp_parsed = re.sub(r'^jsonp\d+\(|\)\s+$', '', response.text)
# data=json.loads(resp_parsed)
# data=response.json()
texte=response.text
content=response.content
# json.loads(response.content.decode('utf-8'))
#%%
x=30
content[int(str(x)+"000"):int(str(x+1)+"000"):]
#%%
with open(chemin+"my_pdf.pdf", 'wb') as my_data:
    my_data.write(content)

#%%

# On récupère les pdf des études d'impact
try:
    os.mkdir(chemin+'PDF_EI')
except:
    print('Le dossier existe déjà.')

vidage=True
if vidage:
    for f in os.listdir(chemin+'PDF_EI'):
        os.remove(chemin+'PDF_EI/'+f)
#%%
for i in tqdm(df_clos['DC.Relation.Expertise Ã©tudeimpact'].values):
    response=requests.get(i)#,headers=he,proxies=proxy)
    response.close()
    k=list(df_clos['DC.Relation.Expertise Ã©tudeimpact']).index(i)
    with open(chemin+"PDF_EI/"+i.split('/')[-1], 'wb') as my_data:
        my_data.write(response.content)
#%%
# On récupère les pdf des avis

try:
    os.mkdir(chemin+'PDF_Avis')
except:
    print('Le dossier existe déjà.')

if vidage:
    for f in os.listdir(chemin+'PDF_Avis'):
        os.remove(chemin+'PDF_Avis/'+f)

for i in tqdm(df_clos['DC.Relation.Expertise avisae'].values):
    try:
        response=requests.get(i)#,headers=he,proxies=proxy)
        response.close()
        k=list(df_clos['DC.Relation.Expertise avisae']).index(i)
        with open(chemin+"PDF_Avis/"+i.split('/')[-1], 'wb') as my_data:
            my_data.write(response.content)
    except:
        print("Il n'y a pas d'avis")

# %%
