#%%
from bs4 import BeautifulSoup
import os
import unicodedata
import string
import numpy as np
# %%
dirpath = '.\Data\TreatedFEIHTML'
doc_path = os.listdir(dirpath)
# %%

#Extraction des textes dans les balises de liens "a"
def toc_extractor(filepath):
    file = unicodedata.normalize('NFKC',open(filepath,'rb').read().decode('utf-8'))
    soup = BeautifulSoup(file, features = "html")
    balises = soup.find_all('a',text = True)
    return [line.get_text().replace(u'\xa0',u'') for line in balises]

#Extraction des strings beautifulsoup : liste des balises (et du texte qu'elles contiennent si il y'en a). Suppression des balises sans texte
# et de la liste des tailles de police (taille de la police i a l'indice i)
def txtBalise_extractor(filepath):
    file = unicodedata.normalize('NFKC',open(filepath,'rb').read().decode('utf-8'))
    soup = BeautifulSoup(file, features = "html")
    first_font = 0
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
    return balises,fontsizes



# %%
#Sort l'id de l'étude
def id_extractor(filepath):
    id = ''
    for car in filepath:
        if car in '1234567890':
            id += car
    return(id)

# %%
# Pour chaque ligne : on regarde la balise qui encapsule le texte et on extrait ses features (nom, type d'attribut, valeur des attributs)
# Puis on accède au parent, on extrait les mêmes caracs, et on extrait les caracs des enfants du parents après avoir exclu la ligne en cours
# Puis on accède au grand parent et on extrait ses caracs
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

#%%

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
# %%

#Nettoie "a la main" les titres pour enlever les éléments qui n'en sont pas

def Title_clean(resultsTitle):
    resultsTitleClean = []
    for result in resultsTitle:
        if 'www' in result:
            pass
        else:
            try:
                int(result)
            except:
                for car in result:
                    if car.isnumeric()==True:
                        resultsTitleClean.append(result)
                        break
    return(resultsTitleClean)
                


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

doc_collection = []
doc_styles = []
for doc in doc_path:
    doc, fonts = txtBalise_extractor(dirpath+'\\'+doc)
    doc_collection.append(doc)
    doc_styles.append(fonts)



# %%

feats_collection = features_collection(doc_collection,doc_styles)

#%%

features_name = build_features(feats_collection)

# %%
Title_collection = []

for doc in doc_path:
    Title_collection.append(Title_clean(toc_extractor(dirpath+'\\'+doc)))

#%%

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

# %%µ

# %%
