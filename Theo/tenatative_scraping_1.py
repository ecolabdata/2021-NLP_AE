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

chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"
#%%
df=pd.read_csv(chemin+"projets-environnement-diffusion.csv",sep=None)
df_clos=df[df[df.columns[6]]=='clos']

# url='https://www.projets-environnement.gouv.fr/explore/dataset/projets-environnement-diffusion/export/?disjunctive.dc_subject_category&disjunctive.dc_subject_theme&disjunctive.vp_status&disjunctive.dc_type&sort=recordsid'
#url='https://www.google.com/search'
he={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:87.0) Gecko/20100101 Firefox/87.0'}
proxy={'http':'http://cache.ritac.i2:32000',
'https':'http://cache.ritac.i2:32000'
}
#%%
response=requests.get(url)#,headers=he,proxies=proxy)
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
########################
###### On récupère les pdf des études d'impact
########################

try:
    os.mkdir(chemin+'PDF_EI')
except:
    print('Le dossier existe déjà.')

vidage=True
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
for i in tqdm(df_clos['DC.Relation.Expertise Ã©tudeimpact'].values):
    try:
        response=requests.get(i)#,headers=he,proxies=proxy)
        if response.status_code==200:
            response.close()
            k=list(df_clos['DC.Relation.Expertise Ã©tudeimpact']).index(i)
            with open(chemin+"PDF_EI/check_erreur/"+i.split('/')[-1], 'wb') as my_data:
                my_data.write(response.content)
            pdfFileObj = open(chemin+"PDF_EI/check_erreur/"+i.split('/')[-1], 'rb')
            try: #On vérifie si le pdf est vide ou pas
                pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
                pdfFileObj.close()
            except:
                pdfFileObj.close(chemin+"PDF_EI/check_erreur/"+i.split('/')[-1])
                os.remove()
                listpasbon_vide.append(i.split('/')[-1])
        else:
            listpasbon_404.append(i.split('/')[-1])
    except:
        continue
#%%
fichiers=os.listdir(chemin+"PDF_EI/check_erreur/")
pdfnonvalide=[]
size=[]
for f in fichiers:
    pdfFileObj = open(chemin+"PDF_EI/check_erreur/"+f, 'rb')
    size.append(os.path.getsize(chemin+"PDF_EI/check_erreur/"+f))
    try:
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        pdfFileObj.close()
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


#%%
########################
##### On récupère les pdf des avis
########################


try:
    os.mkdir(chemin+'PDF_Avis/check_erreur')
except:
    print('Le dossier existe déjà.')

vidage=False
if vidage:
    for f in os.listdir(chemin+'PDF_Avis'):
        os.remove(chemin+'PDF_Avis/'+f)

import PyPDF2
import shutil

vidage=True
if vidage:
    for f in os.listdir(chemin+'PDF_Avis/check_erreur'):
        try:
            os.remove(chemin+'PDF_Avis/check_erreur/'+f)
        except:
            break
#%%
listpasbon_avis_404=[]
listpasbon_avis_vide=[]

for i in tqdm(df_clos['DC.Relation.Expertise avisae'].values):
    try:
        response=requests.get(i,headers=he,proxies=proxy)
        if response.status_code==200:
            response.close()
            k=list(df_clos['DC.Relation.Expertise avisae']).index(i)
            with open(chemin+"PDF_Avis/check_erreur/"+i.split('/')[-1], 'wb') as my_data:
                my_data.write(response.content)
            pdfFileObj = open(chemin+"PDF_Avis/check_erreur/"+i.split('/')[-1], 'rb')
            try: #On vérifie si le pdf est vide ou pas
                pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
                pdfFileObj.close()
                print("Avis scrapé !")
            except:
                pdfFileObj.close(chemin+"PDF_Avis/check_erreur/"+i.split('/')[-1])
                os.remove()
                listpasbon_avis_vide.append(i.split('/')[-1])
        else:
            listpasbon_avis_404.append(i.split('/')[-1])
    except:
        print("Il n'y a pas d'avis")

#%%
dico_pasbon={}
# dico_pasbon['EI_404']=listpasbon_404
# dico_pasbon['EI_vide']=listpasbon_vide
dico_pasbon['Avis_404']=listpasbon_avis_404
dico_pasbon['Avis_vide']=listpasbon_avis_vide
import pickle
pickle.dump(dico_pasbon,open(chemin+"PDF_EI/check_erreur/dico_pasbon_avis.pickle",'wb'))
#%%
dico_EI=pickle.load(open(chemin+"PDF_EI/check_erreur/dico_pasbon.pickle",'rb'))
dico_EI
#%%
dico_avis=pickle.load(open(chemin+"PDF_EI/check_erreur/dico_pasbon_avis.pickle",'rb'))
dico_avis
#%%
#######################################################################################################
######### vérification problème OCR #
#############################################################################################
df_clos['num_etude']=[int(i[:-8].split('/')[-1]) for i in df_clos['DC.Relation.Expertise Ã©tudeimpact'].values]
#%%
########## Fichier d'image non reconnu 

#link=df_clos[df_clos.num_etude==2663489]['DC.Relation.Expertise Ã©tudeimpact'].values[0]

# link=df_clos[df_clos.num_etude==1106625]['DC.Relation.Expertise Ã©tudeimpact'].values[0]
#%%
#Un qui marche
link=df_clos[df_clos.num_etude==969856]['DC.Relation.Expertise Ã©tudeimpact'].values[0]
#%%
######### PDF non valide

link=df_clos[df_clos.num_etude==2225340]['DC.Relation.Expertise Ã©tudeimpact'].values[0]
# On a bien status_code==200 donc a priori il y a bien un pdf
#%%
he={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:87.0) Gecko/20100101 Firefox/87.0'}
proxy={'http':'http://cache.ritac.i2:32000',
'https':'http://cache.ritac.i2:32000'
}
url='https://www.google.com/search'

resp=requests.get(link,timeout=(10,200))#,headers=he,proxies=proxy)
resp.status_code==200
#%%
with open(chemin+"PDF_EI/test.pdf", 'wb') as my_data:
                my_data.write(resp.content)
                my_data.close()
#%%
with open(chemin+"PDF_EI/test.pdf", 'rb') as my_data:
                pdf=my_data.read()
#%%
# import subprocess
# subprocess.Popen([chemin+"PDF_EI/test.pdf"],shell=False)
# open(chemin+"PDF_EI/test.pdf")
# try:
#     os.system(chemin+"PDF_EI/test.pdf")
# except:
#     print('ouverture non possible')
import PyPDF2
  
# creating a pdf file object
pdfFileObj = open(chemin+"PDF_EI/969856_FEI.pdf", 'rb')
  
# creating a pdf reader object
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
  
#%%
import PyPDF2
# pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pdfFileObj.close()
import os
#%%

pdfFileObj = open(chemin+"PDF_EI/test.pdf", 'rb')
import PyPDF2
try:
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
except:
    pdfFileObj.close()
    os.remove(chemin+"PDF_EI/test.pdf")

# %%
##########################################################################################################################################################
###########       Garance       ##################################################################################################################################
##########################################################################################################################################################
from pandas_ods_reader import read_ods

chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"
df=read_ods(chemin+"20210325_export_Garance_avisAe_evnts_avis.ods",1)
df_avis=df[df.libc_type_evt=='AVDEC']
# df_clos=df[df[df.columns[6]]=='clos']
#%%
import pandas as pd, numpy as np
import bs4
import requests
from tqdm import tqdm
import re,json
from unidecode import unidecode
import html
from html.parser import HTMLParser

he={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:87.0) Gecko/20100101 Firefox/87.0'}
proxy={'http':'http://cache.ritac.i2:32000',
'https':'http://cache.ritac.i2:32000'
}


link="http://garance.e2.rie.gouv.fr/entrepot/documents/documents_2053/"
#%%
url=link+df.lien_document[0]
response=requests.get(url)#,headers=he,proxies=proxy)
response.close()
# texte=response.text
# content=response.content
with open(chemin+"Garance/"+"essai.pdf", 'wb') as my_data:
    my_data.write(response.content)
#%%
try:
    os.mkdir(chemin+'Garance')
except:
    print('Le dossier existe déjà.')

vidage=True
if vidage:
    for f in os.listdir(chemin+'Garance'):
        os.remove(chemin+'Garance/'+f)
pasbon404=0
df_avis_2=df_avis[[df_avis.lien_document[i] is not None for i in df_avis.index]]
for i in tqdm(df_avis_2.index):
    k=str(int(df_avis_2.dossier[i]))
    h=str(df_avis_2.lib_type_evt[i])
    try:
        response=requests.get(link+df_avis_2.lien_document[i])#,headers=he,proxies=proxy)
        if response.status_code==200
            response.close()
            # k=list(df_clos['DC.Relation.Expertise avisae']).index(i)
            with open(chemin+"Garance/"+k+"_"+h+".pdf", 'wb') as my_data:
                my_data.write(response.content)
        else:
            pasbon404+=1
    except:
        print("Il n'y a pas d'avis")

# %%
##########################################################################################################################################################
###########       LégiFrance   et Thesaurus    ##################################################################################################################################
##########################################################################################################################################################

url="https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000042369329/"
response=requests.get(url)#,headers=he,proxies=proxy)
response.close()
texte=response.text
content=response.content
#%%
from bs4 import BeautifulSoup
soup=BeautifulSoup(content,"html.parser")
soup
legifrance=re.sub('\t',' ',soup.get_text())
#%%
legi=pd.DataFrame(legifrance.split('\n'))
legi=legi[[True if len(legi.iloc[i,:].values[0])>1 else False for i in range(len(legi))]]
legi
# %%
import spacy
nlp = spacy.load('fr_core_news_sm',disable=["parser","ner"]) #on charge le modèle spacy pour le français
#%%%
pipe=nlp.pipe(legi.iloc[:,0])
for i in pipe:
    print(i)

#%%
def represent_word(word):
    import re
    from unidecode import unidecode
    # text = word.lemma_
    # True-case, i.e. try to normalize sentence-initial capitals.
    # Only do this if the lower-cased form is more probable.
    text = str(word).replace("lire page","") #replce "lire page" par "" (en gros delete "lire page")
    text = unidecode(text) #take a unicode object and returns a string
    text = re.sub(r'[^A-Za-z]',' ',str(text)) 
    #means any character that IS NOT a-z OR A-Z
    text = ' '.join(text.split())
    return text

# %%
clean=[" ".join(represent_word(w) for w in doc if not w.is_punct and not w.is_stop and w.pos_ in ["VERB","NOUN","ADJ","PROPN"]) for doc in nlp.pipe(legi.iloc[:,0])]
legi['clean']=clean
# %%
chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"

df=pd.read_csv(chemin+'etudes_dataset_cleaned_prepared.csv',sep=";")

theme=list(np.unique(re.sub("[\(\[].*?[\)\]]", "",
    re.sub(","," ",
        re.sub(";"," ",' '.join(np.unique(df.theme.values))))).split(' ')))
theme.remove('ET'),theme.remove('')
theme=[re.sub('-','',unidecode(i.lower())) for i in theme]
#%%
from gensim.models import Word2Vec
wdow=10
minc=1
sentence=[i.split() for i in clean]
model = Word2Vec(sentence, window=wdow,min_count=minc)
#%%
tn=50
mots={}
for i in theme:
    try:
        mots[i]=[(item[0],round(item[1],2)) for item in model.most_similar(i,topn=tn)]
    except:
        print(i)
# %%
print(mots)
import pickle
pickle.dump(mots,open(chemin+"Thesaurus_LegiFrance.pickle","wb"))
# %%
import pickle
chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"

dico=pickle.load(open(chemin+"Thesaurus_LegiFrance.pickle",'rb'))
# %%
#####################################################################################
import os
liste_1=os.listdir("C:/Users/theo.roudil-valentin/Documents/Donnees/PDF_EI")
liste_2=os.listdir("C:/Users/theo.roudil-valentin/Documents/Donnees/PDF_EI/pasbon_html")
import shutil
liste_1=[i for i in liste_1 if i[-3:]=='pdf']
from tqdm import tqdm
print(liste_1)
for i in tqdm(liste_1):
    if i not in liste_2:
        shutil.copy("C:/Users/theo.roudil-valentin/Documents/Donnees/PDF_EI/"+i,'D:/PDF_EI/'+i)

# %%
