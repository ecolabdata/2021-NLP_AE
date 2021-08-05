#%%
##########################################################################################################################################################
###########       Projets environnements diffusion       ##################################################################################################################################
##########################################################################################################################################################

import pandas as pd, numpy as np
import bs4
import requests
from tqdm import tqdm
import re,json
from unidecode import unidecode
import html
# from requests_html import HTMLSession
from html.parser import HTMLParser
import os
chemin="Data/"

#%%
df=pd.read_csv(chemin+"Garance.csv",sep=";",encoding='unicode_escape')


url='http://garance.e2.rie.gouv.fr/entrepot/documents/documents_2053/'
he={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:87.0) Gecko/20100101 Firefox/87.0'}
proxy={'http':'http://cache.ritac.i2:32000',
'https':'http://cache.ritac.i2:32000'
}
df = df[df['lib_type_evt']=="Avis-Décision"]

import dateutil
df.date_debut = df.date_debut.apply(dateutil.parser.parse)
#On prend les avis émis après 2017 seulement
df = df[df.date_debut>dateutil.parser.parse('2017-01-01')]

dossiers = os.listdir("K:\\03 ECOLAB\\2 - POC DATA et IA\Données NLP\dossiers_etudes_brz_Alfresco\Zip")
dossiers = [int(string[5:11]) for string in dossiers]
#%%
response=requests.get(url)#,headers=he)
response.close()
# resp_parsed = re.sub(r'^jsonp\d+\(|\)\s+$', '', response.text)
# data=json.loads(resp_parsed)
# data=response.json()
texte=response.text
content=response.content
# json.loads(response.content.decode('utf-8'))

#%%
########################
###### On récupère les pdf des études d'impact
########################

try:
    os.mkdir(chemin+'avis_garance')
except:
    print('Le dossier existe déjà.')

vidage=False
if vidage:
    for f in os.listdir(chemin+'PDF_EI/check_erreur'):
        try:
            os.remove(chemin+'PDF_EI/check_erreur/'+f)
        except:
            break
#%%
import PyPDF2
import os

listpasbon_404=[]
listpasbon_vide=[]
for i in tqdm(df['lien_document'].values):
    try:
        response=requests.get(url+i)#,headers=he,proxies=proxy)
        if response.status_code==200:
            response.close()
            with open(chemin+"avis_garance/"+i.split('/')[-1], 'wb') as my_data:
                my_data.write(response.content)
            pdfFileObj = open(chemin+"avis_garance/"+i.split('/')[-1], 'rb')
            try: #On vérifie si le pdf est vide ou pas
                pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
                pdfFileObj.close()
            except:
                pdfFileObj.close(chemin+"avis_garance/"+i.split('/')[-1])
                os.remove()
                listpasbon_vide.append(i.split('/')[-1])
        else:
            listpasbon_404.append(i.split('/')[-1])
    except:
        continue
#%%
from tqdm import tqdm
import PyPDF2
fichiers=os.listdir(chemin+"avis_garance/")
pdfnonvalide=[]
size=[]
for f in tqdm(fichiers):
    try:
        pdfFileObj = open(chemin+"avis_garance/"+f, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    except:
        pdfnonvalide.append(f)
        pass
    size.append(os.path.getsize(chemin+"avis_garance/"+f))
    content = ''
    for page in pdfReader.pages:
        content +=page.extractText()+' '
    pdfFileObj.close()
    txtfile = open(chemin+"/avis_garance_txt/"+f[:-4]+".txt",'w')
    try:
        txtfile.write(content)
    except:
        pdfnonvalide.append(f)

#%%
print("Il y a :",len(pdfnonvalide),"PDF non valides soit",round((len(pdfnonvalide)/len(fichiers))*100,2),"% de l'ensemble")
#%%
import matplotlib.pyplot as plt
f,a=plt.subplots(1,figsize=(12,6))
a.hist(size,bins=100)
a.set(xlabel="Taille (en Mo)",ylabel='Nombre de fichiers',
      title='distribution de la taille des pdf')
#%%
size=[i/1000/1000 for i in size]
print("La taille moyenne est :",np.mean(size))
print("L'écart-type est :",np.std(size))
print("La médiane est :",np.median(size))
# plusieurs choses : la moyenne est quasiment le double de la médiane, 
# la distribution est très fortement concentrée
# D'ailleurs l'écart-type est très très élevé

# %%
from tqdm import tqdm

def nelement(liste,n):
    sort = liste.copy().sort()
    val = sort[-n]
    return(val)

import shutil
import numpy as np
import matplotlib.pyplot as plt
path = "K:\\03 ECOLAB\\2 - POC DATA et IA\Données NLP\dossiers_etudes_brz_Alfresco"
path_do = path + "\Dossiers"
Etudes = os.listdir(path_do)
dossiers = dossiers
c = 0
copied = os.listdir(path+'\\Etudes\\')
print(len(copied))
empty = []
y = []
for f,num in tqdm(zip(Etudes,dossiers)):
    w = os.listdir(path_do+'\\'+f)
    files = os.listdir(path_do+'\\'+f+'\\'+w[0]+'\\a - Dossier')
    size = []
    for pdf in files:
        if pdf in copied:
            pass
        size.append(os.path.getsize(path_do+'\\'+f+'\\'+w[0]+'\\a - Dossier\\'+pdf))
    
    try:
        maxsize = max(size)
        maxindex = size.index(maxsize)
        n = 1
        while 'annexe' in files[maxindex].lower():
            n+=1
            maxsize = nelement(size,n)
            maxindex = size.index(maxsize)
        shutil.copyfile(path_do+'\\'+f+'\\'+w[0]+'\\a - Dossier\\'+files[maxindex],path+'\\Etudes\\'+str(num)+'-'+files[maxindex])
    except:
        empty.append(pdf)
