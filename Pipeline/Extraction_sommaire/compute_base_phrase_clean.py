# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from bs4 import BeautifulSoup
import os
import unicodedata
import string
import time
import lxml
import tqdm
import bs4
# Read recipe inputs
path = dataiku.Folder("7SAfnrwf").get_path()
path_of_csv = os.path.join(path, "base_html_06_04.csv")

df_html=pd.read_csv(path_of_csv)
np.unique(df_html.num_etude).shape

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#Extraction des strings beautifulsoup : liste des balises (et du texte qu'elles contiennent si il y'en a). Suppression des balises sans texte
# et de la liste des tailles de police (taille de la police i a l'indice i)


def txtBalise_extractor(file, limit = 3):
    soup = BeautifulSoup(file, features = "lxml")
    txt = soup.body.find_all(text = True,)

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

    txt_clean = []
    features_doc = []
    longueur = len(txt)
    duration = 0.00001
    for it,line in tqdm.tqdm(enumerate(txt),total = len(txt)) :
        stime = time.time()

        features = []
        if not '\n' in line:
            parent = line.parent
            parents = []
            if parent.get_text()==line:
                parents.append(parent)
                features += features_chain2(parent,fontsizes)

                k=0
                for k in range(limit):
                    parent = parent.parent
                    if parent.get_text().strip('\n')==line:
                        parents.append(parent)
                        features += features_chain2(parent,fontsizes)

                    else:
                        break
                txt_clean.append(parents[len(parents)-1])
                features.append(features_added(line))
                features_doc.append(features)
        duration += time.time() - stime
    return txt_clean,features_doc,fontsizes



def features_chain2(b4string,fontsizes):
    if type(b4string) != bs4.element.NavigableString:
        features = []
        features.append(b4string.name)
        attrs = b4string.attrs
        for attr in list(attrs.keys()):

            features.append(attr)
            attrsclean = feat_clean(attrs[attr],fontsizes)
            if type(attrsclean) == list:
                features += attrsclean
            elif type(attrsclean) != type(None):
                features.append(attrsclean)
        return(features)
    else:
        return

def feat_clean(feat,fontsizes):
    if type(feat) == list:
        if 'font' in str(feat):
            try :
                size = fontsizes[int(feat[0][4:])]
                return(size)
            except:
                pass
        return(feat[0])
    if 'bookmark' in str(feat):
        return('bookmark')

    if 'mailto' in str(feat):
        return

    if 'http' in str(feat):
        return

    if '.jpg' in str(feat):
        return

    if 'footnote' in str(feat):
        return

    if '.png' in str(feat):
        return

    if 'width:' in str(feat):
        return


    if type(feat) == str and len(feat.split(';')) >1:
        features = []
        for at in feat.split(';'):
            if at != '':
                features.append(at)
        return(features)

def features_added(stringtxt):
    # Feature construite à la main pour voir si il y'a un nombre dans la chaine
    for car in stringtxt:
            if car in '0123456789':
                return('hasNumber')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#Construit le dico des features uniques
def build_features(ftlists):
    diff_feat = []
    for entry in ftlists:
        for feats in entry:
            for feat in feats:
                diff_feat.append(str(feat))
    diff_feat = np.unique(diff_feat)

    k = 0
    feat_dico = {}
    for feat in diff_feat:
        feat_dico[feat] = k
        k += 1
    return(feat_dico)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#### Prend en entrée les features extraites ligne par ligne pour un doc,
#le dico des features, pour construire la matrice des indicatrices
def encode_matrix(doc_feat,features_names):
    for k in list(features_name.keys()):
        try:
            int(k)
        except:
            a=list(features_name.keys()).index(k)
            break
    matrix = np.zeros((len(doc_feat),len(list(features_names.keys())[a-1:])))
    k = 0
    for rowfeat in doc_feat:
        for feat in rowfeat:
            try:
                    feat=int(feat)
                    matrix[k,0] = feat
            except:
                    ind = features_names[str(feat)]-a
                    matrix[k,ind] = 1
        k += 1
    return(matrix)

#Applique la fonction précédente a une collection, concatène
def encode_collection(docs_feature_collection,feature_names):
    matrix = encode_matrix(docs_feature_collection[0],feature_names)
    i = 0
    for doc_feat in docs_feature_collection[1:]: #Pourquoi tu enlèves l'élément 0 ???
        child = encode_matrix(doc_feat,feature_names)
        matrix = np.concatenate((matrix,child),axis = 0)
    return(matrix)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#Extraction des textes dans les balises de liens "a"
def toc_extractor(file):
    soup = BeautifulSoup(file, features = "html")
    balises = soup.find_all('a',text = True)
    return [line.get_text().replace(u'\xa0',u'') for line in balises]

#Nettoie "a la main" les titres pour enlever les éléments qui n'en sont pas
def Title_clean(resultsTitle):
    resultsTitleClean = []
    for it,result in tqdm.tqdm(enumerate(resultsTitle),total = len(resultsTitle)) :
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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
start=time.time()
doc_collection = []
doc_styles = []
features_collection = []
for doc in df_html.texte:
    doc, features, fonts = txtBalise_extractor(doc)
    doc_collection.append(doc)
    features_collection.append(features)
    doc_styles.append(fonts)
end=time.time()
print('Durée :', round((end-start)/60,2),' minutes')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
len(doc_collection),len(doc_collection[0]),len(doc_collection[0][0]),len(doc_styles),len(doc_styles[0])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
doc_collection

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
start=time.time()

features_name = build_features(features_collection)

end=time.time()
print('Durée :', round((end-start)/60,2),' minutes')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
features_name

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
start=time.time()

Title_collection = []

for doc in df_html.texte:
    Title_collection.append(Title_clean(toc_extractor(doc)))

end=time.time()
print('Durée :', round((end-start)/60,2),' minutes')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
features_collection

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
start=time.time()

ft_matrix = encode_collection(features_collection,features_name)

#target_vector = Encode_vector(doc_collection,Title_collection)


end=time.time()
print('Durée :', round((end-start)/60,2),' minutes')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_html.texte

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
phrases = []
doc_id = []
for ids,doc in zip(df_html.num_etude,doc_collection):
    for phrase in doc:
        phrases.append(str(phrase))
        doc_id.append(ids)

df_phrase_id = pd.DataFrame({'phrase':phrases, 'num_etude':doc_id})
df_phrase_id

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
len(doc_collection),len(doc_collection[0])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df=pd.DataFrame(ft_matrix)
np.unique(df.num_etude)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df=pd.DataFrame(ft_matrix)
#df_target=pd.DataFrame(target_vector)
#df_target.columns=['target']
#df=pd.concat([df,df_target],axis=1)
df=pd.concat([df_phrase_id,df],axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base_phrase_clean_df = df # Compute a Pandas dataframe to write into base_phrase_clean


# Write recipe outputs
base_phrase_clean = dataiku.Dataset("base_phrase_clean")
base_phrase_clean.write_with_schema(base_phrase_clean_df)