# %%
' '.join([i for i in df.docs.to_list()])
#%%
from bs4 import BeautifulSoup
import os
import unicodedata
import string
import numpy as np
import spacy
import time
from joblib import delayed,Parallel
import sys
from joblib import wrap_non_picklable_objects
from spacy.util import minibatch
# %%
import pandas as pd
chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"

df_html=pd.read_csv(chemin+'base_html.csv')

# %%

#Extraction des textes dans les balises de liens "a"
def toc_extractor(f):
    # file = unicodedata.normalize('NFKC',open(filepath,'rb').read().decode('utf-8'))
    soup = BeautifulSoup(f, features = "html.parser")
    balises = soup.find_all('a',text = True)
    return [line.get_text().replace(u'\xa0',u'') for line in balises]

#%%
#Extraction des strings beautifulsoup : liste des balises (et du texte qu'elles contiennent si il y'en a). Suppression des balises sans texte
# et de la liste des tailles de police (taille de la police i a l'indice i)

# @wrap_non_picklable_objects
def txtBalise_extractor(f):
    # file = unicodedata.normalize('NFKC',open(filepath,'rb').read().decode('utf-8'))
    soup = BeautifulSoup(f, "html.parser")
    first_font = 0
    try:
        fonts = soup.style.contents[0].split('\n')
        for id in range(len(fonts)):
            if 'font' in fonts[id]:
                first_font = id
                break
        fontsclean = fonts[first_font:len(fonts)-2]
        fontsizes = []
        for font in fontsclean:
            deb = font.index('font:')
            fin = font.index('pt')
            size = int(font[deb+5:fin])
            fontsizes.append(size)
        balises = soup.find_all()
    except:
        balises = soup.find_all()
        fontsizes=[]
    return balises,fontsizes
#%%
def txtBalise_test(f):
    # file = unicodedata.normalize('NFKC',open(filepath,'rb').read().decode('utf-8'))
    soup = BeautifulSoup(f, "html.parser")
    balises=soup.find_all()
    return balises
#%%
N=1000 # marche avec 1000 mais pas avec 10 000 :'(
b=[df_html.texte.values[0][:N],df_html.texte.values[1][:N]]
sys.setrecursionlimit(1000000000) #cette solution n'a pas fonctionné
Parallel(n_jobs=2)(delayed(txtBalise_extractor)(str(i)) for i in enumerate(b))
# %%

# partitions=minibatch(df_html.texte,10)
start=time.time()
ouais=Parallel(n_jobs=2,verbose=10)(delayed(txtBalise_extractor)(i) for i in df_html.texte.values)
end=time.time()
print('Durée :',round((end-start)/60,2),' minutes.')
ouais
#%%
from unidecode import unidecode
def puta(s):
    b=[]
    for i in s:
        i=unidecode(i)
        b.append(i)  
    return b
a=['ouaislyuvouciyfxrwea<(-vhjblkn']
# puta(a[0])
# Parallel(n_jobs=2)(delayed(puta)(i) for i in a)
#%%
start=time.time()
ouais=Parallel(n_jobs=8,verbose=10)(delayed(txtBalise_extractor)(i) for i in df_html.texte[:2])
end=time.time()
print('Durée :',round((end-start)/60,2),' minutes.')
ouais

#%%
#Sort l'id de l'étude
def id_extractor(filepath):
    id = ''
    for car in filepath:
        if car in '1234567890':
            id += car
    return(id)


# Pour chaque ligne : on regarde la balise qui encapsule le texte et on extrait ses features (nom, type d'attribut, valeur des attributs)
# Puis on accède au parent, on extrait les mêmes caracs, et on extrait les caracs des enfants du parents après avoir exclu la ligne en cours
# Puis on accède au grand parent et on extrait ses caracs
#%%
from joblib import wrap_non_picklable_objects
@wrap_non_picklable_objects
def features_extractor(b4string,fontsizes):
    
    try:
        parent = b4string.parent
        features = [parent.name]
        try:
            for attr in parent.attrs:
                features.append(attr)
                try:
                    features.append(parent.attrs[attr][0])
                except:
                    features.append(parent.attrs[attr])
        except:
            pass
    except:
        return []
    

    try:
        parent_cont = parent.contents.remove(b4string)
    except:
        parent_cont = parent.contents
    try:
        for child in parent_cont:
            features.append(child.name)
            attrs_child = [attr for attr in child.attrs]
            for attr in attrs_child:
                try :
                    features.append(attr)
                    try:
                        features.append(child.attrs[attr][0])
                    except:
                        features.append(child.attrs[attr])
                    
                except:
                    pass
    except:
        pass

    gd_parent = parent.parent
    try:
        features.append(gd_parent.name)   
        try:
            for attr in gd_parent.attrs:
                features.append(attr)
                try:
                    features.append(gd_parent.attrs[attr][0])
                except:
                    features.append(gd_parent.attrs[attr])
        except:
            pass
    except:
        pass         
    while None in features:
        features.remove(None)      

    # Feature construite à la main pour voir si il y'a un nombre dans la chaine
    for car in b4string.get_text():
            if car in '0123456789':
                features.append('hasNumber')
                break

    # On remplace les numéros de police par leur size, on enlève les numéros de bookmark, enlève les listes 
    for feat in features:
        if 'font' in str(feat):
            try :
                size = fontsizes[int(feat[4:])]
                features.remove(feat)
                features.append(size)
            except:
                print(feat,fontsizes)
        if 'bookmark' in str(feat):
            features.remove(feat)
            features.append('bookmark')
        if type(feat) == list:
            features.replace(feat,feat[0])

    return(features)

import sys
sys.setrecursionlimit(100000)
start=time.time()
ouais=Parallel(n_jobs=8,verbose=10)(delayed(features_extractor)(i,z) for i,z in zip(doc_collection[0],doc_styles[0]))
end=time.time()
print('Durée :',round((end-start)/60,2),' minutes.')


#%%
#Construit la liste des features ligne par ligne
#pour un document contenant une liste d'objets b4string
def features(doc,fontsize):
    features = []
    for chain in doc:
        feats = features_extractor(chain,fontsize)
        features.append(feats)
    return(features)

#Applique la fonction a une collection
def features_collection(doc_collection,fontsizes):
    feature_collection = []
    for doc,font in zip(doc_collection,fontsizes):
        feature_collection.append(features(doc,font))
    return(feature_collection)

#Construit le dico des features uniques
def build_features(ftlists):
    diff_feat = []
    for entry in ftlists:
        for feats in entry:
            for feat in feats:
                diff_feat.append(feat)
    diff_feat = np.unique(diff_feat)
   
    k = 0
    feat_dico = {}
    for feat in diff_feat:
        feat_dico[feat] = k
        k += 1
    return(feat_dico)



#Prend en entrée les features extraites ligne par ligne pour un doc,
#le dico des features, pour construire la matrice des indicatrices
def encode_matrix(doc_feat,features_names):
    matrix = np.zeros((len(doc_feat),len(features_names)))
    k = 0
    for rowfeat in doc_feat:
        for feat in rowfeat:
            ind = features_names[str(feat)]
            matrix[k,ind] = 1
        k += 1
    return(matrix)

#Applique la fonction précédente a une collection, concatène
def encode_collection(docs_feature_collection,feature_names):
    matrix = encode_matrix(docs_feature_collection[0],feature_names)
    i = 0
    for doc_feat in docs_feature_collection[1:]:
        child = encode_matrix(doc_feat,feature_names)
        matrix = np.concatenate((matrix,child),axis = 0)
    return(matrix)


#Nettoie "a la main" les titres pour enlever les éléments qui n'en sont pas

def Title_clean(resultsTitle):
    resultsTitleClean = []
    for result in resultsTitle:
        if 'www' in result:
            pass #Si il y a un "www" on prend pas
        else:
            try:
                int(result) #Si l'élèment est uniquement un chiffre on le prend pas
            except:
                for car in result:
                    if car.isnumeric()==True: #Si il y a au moins un numéro on le prend
                        resultsTitleClean.append(result)
                        break
    return(resultsTitleClean)
                
# title =df_html.texte[0]
#%%
import time
start=time.time()
Title_clean(title)
end=time.time()
print('Durée :',round(end-start,2),'secondes')


# %%

#Encode 1 si c'est un titre, 0 sinon, pour 1 document
def Encode(results,resultsTitle):
    vector = []
    for res in results:
        clean = res.get_text()
        if clean in resultsTitle:
            vector.append(1)
        else:
            vector.append(0)
    return(vector)

#Applique la fonction précédente à la collection de documents, concatène
def Encode_vector(doc_collection,Title_collection):
    vector = []
    for doc,titles in zip(doc_collection,Title_collection):
        vector += Encode(doc,titles)
    return(vector)

# %%
soup = BeautifulSoup(df_html.texte[0], "html.parser")
first_font = 0
fonts = soup.style.contents[0].split('\n')
fonts
#%%
for id in range(len(fonts)):
    if 'font' in fonts[id]:
        first_font = id
        break
fontsclean = fonts[first_font:len(fonts)-2]
fontsizes = []
for font in fontsclean:
    deb = font.index('font:')
    fin = font.index('pt')
    size = int(font[deb+5:fin])
    fontsizes.append(size)
balises = soup.find_all()
#%%
doc_collection = []
doc_styles = []
start=time.time()
for doc in df_html.texte:
    doc, fonts = txtBalise_extractor(doc)
    doc_collection.append(doc)
    doc_styles.append(fonts)
end=time.time()
print("Durée :",round((end-start)/60,2),"minutes.")
#%%
def trans(textes, textes2,function,n_jobs=8,verbose=10):
    import time
    from joblib import Parallel, delayed
    # texts = zip(textes,textes2) # on match les éléments au sein de "texts"
    # partitions = minibatch(texts, size=batch_size) #crée les partitions sur lesquelles on va itérer, de dimensions batch_size sur l'élément texts
    executor = Parallel(n_jobs=n_jobs, prefer="processes",verbose=verbose) # on sépare les tâches en n_jobs, et on crée l'exécuteur
    do = delayed(function) #on prend la fonction transformation_texts et on fixe l'argument nlp (?) 
    #delayed crée une liste de choses à éxécuter petit à petit, il indique qu'on va appliquer la fonction à une "liste" d'éléments
    tasks = (do(i,z) for i,z in zip(textes,textes2)) # on crée la tâche, c'est-à-dire qu'on applique la fonction partial(tranfo_text,nlp) a une liste en lui disant de procéder par étapes (delayed)
    start=time.time()
    ouais=executor(tasks) # on exécute
    end=time.time()
    print("La tâche a pris :", round((end-start)/60,2),"minutes")
    return ouais
#%%
from joblib import Parallel, delayed
from functools import partial
import os
from spacy.util import minibatch

#%%
executor = Parallel(n_jobs=2,verbose=10) # on sépare les tâches en n_jobs, et on crée l'exécuteur
do = delayed(features_extractor)  

#%%
#delayed crée une liste de choses à éxécuter petit à petit, il indique qu'on va appliquer la fonction à une "liste" d'éléments
tasks = (do(i,z) for i,z in zip(doc_collection,doc_styles)) # on crée la tâche, c'est-à-dire qu'on applique la fonction partial(tranfo_text,nlp) a une liste en lui disant de procéder par étapes (delayed)
ouais=executor(tasks) # on exécute
#%%
trans(doc_collection,doc_styles,features)
# %%
import time
def func_async(i, *args):
    return 2 * i


# We have to pass an extra argument with a large list (or another large python
# object).
large_list = list(range(1000000))

t_start = time.time()
wouhou=Parallel(n_jobs=2,verbose=10)(delayed(func_async)(i) for i in range(100))
print("With loky backend and cloudpickle serialization: {:.3f}s"
      .format(time.time() - t_start))
wouhou


#%%
feats_collection = features_collection(doc_collection[:4],doc_styles[:4])

#%%

features_name = build_features(feats_collection)

# %%
Title_collection = []
start=time.time()
for doc in df_html.texte:

    Title_collection.append(Title_clean(toc_extractor(doc)))
end=time.time()
print(end-start)
# %%
start=time.time()
[Title_clean(toc_extractor(i)) for i in df_html.texte]
print(time.time()-start)
# %%

ft_matrix = encode_collection(feats_collection,features_name)

# %%

target_vector = Encode_vector(doc_collection,Title_collection)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(ft_matrix,
                                                target_vector,
                                                test_size=0.2,
                                                random_state=10)

classif = RandomForestClassifier(n_estimators=100,random_state=1,n_jobs=-1)
classif.fit(X_train,y_train)
# %%
print('Score: ',classif.score(X_test,y_test))

y_pred = classif.predict(X_test)

probas = classif.predict_proba(X_test)[:,1]

from sklearn.metrics import recall_score

print('Recall: ',recall_score(y_test,y_pred,pos_label=1))

from sklearn.metrics import f1_score

print('F1: ',f1_score(y_test,y_pred,pos_label=1))

from sklearn.metrics import roc_auc_score

print('AUC Score: ',roc_auc_score(y_test,probas))
# %%

test_mat = encode_matrix(features(doc_collection[0],),features_name)

# %%

classif.predict(test_mat)

# %%