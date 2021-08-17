#%%

import pandas as pd, numpy as np
import pickle
import unidecode
import spacy
from nltk.corpus import stopwords
import re
import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct
from Pipeline.Enjeux.utils import *
from Pipeline.Enjeux.topicmodeling_pipe import CorExBoosted
import os

#On charge notre texte et notre thésaurus
if os.path.isfile('Data\Bases_intermediaire\df_avis_clean.pickle'):
    docs_df = pickle.load(open('Data\Bases_intermediaire\df_avis_clean.pickle','rb'))
else:
    docs_df = pickle.load(open("Data\Bases_intermediaire\\df_id_avis_txt_sorted.pickle",'rb'))

if os.path.isfile('Data\Enjeux\\modele_avis.pickle'):
    instance = pickle.load(open('Data\Enjeux\\modele_avis.pickle','rb'))
Thesaurus = pickle.load(open("Data\Enjeux\Thesaurus\Thesaurus1.pickle",'rb'))

metadata = pd.read_csv("Data\Metadata\etudes_avis_dep_them_id.csv")
from Pipeline.Enjeux.processing_encoding import processing_thesaurus
#Le préprocessing permet de lemmatiser les mots du thésaurus de la même manière que les mots du texte vont l'être (sinon ils ne seront pas reconnus)
Thesaurus = processing_thesaurus(Thesaurus)
#%%
#On initialise une instance de CorExBoosted
instance = CorExBoosted(docs_df[:100],Thesaurus)
#On preprocess les textes si ce n'est pas déjà fait dans le dataframe chargé
if 'text_processed' not in docs_df.columns:
    instance.preprocess('texte')

#On dispose d'outils de diagnostic du vocabulaire du thésaurus si nécessaire pour visualiser la couverture du vocabulaire (nombre de mots du dictionnaire réellement présents dans le vocabulaire
# du vectoriseur)
diagnostics = instance.diagnostic()
#On encode
instance.encode()
#On peux accéder a des informations de diagnostic ici encore grace a la méthode encore qui génère des attributs, respectivement:
#Le mapping word-id, la fréquence d'apparition des mots, le vocabulaire trié, les mots du thésaurus qui ne sont pas dans le vocabulaire...
instance.word2id,instance.words_freq,instance.vocab_sort,instance.notinvoc

#%%
#On fit les classifieurs. Ici on stratifie et on stratifie et on augmente les données puisqu'on dispose de données corrigées
corrige = pd.read_excel("Data\Enjeux\Avis_corrige.xlsx")
corrige.dropna(inplace = True)
corrige.drop(['titre', 'url_etude', 'url_avis', 'url_avis_cliquable', 'Status',
        'Biodiversité',
    'Paysage et qualité de vie',
    'Santé et sécurité des personnes',
    'Effets globaux (climat, énergie, ressources...)',
    'Préservation des sols',
    'Qualité de l’eau et ressource',
    'Déplacements', 'Gestion des déchets'], axis = 1, inplace = True)
corrige.id_AAE = corrige.id_AAE.astype(int)

instance.fit(n_classif=2,strength=2,df_corrige = corrige)
#%%
df_corrige = corrige
docs_df = instance.docs
enjeux_list = [c.replace('True_','') for c in df_corrige.columns[1:]]
docs_df.dropna(inplace = True)
docs_df.id_AAE = docs_df.id_AAE.astype(int)
docs_df['idx_copy'] = docs_df.index
df_corrige.dropna(inplace =True)
final =df_corrige.merge(docs_df, on = 'id_AAE', how='inner')
X_df = pd.DataFrame(instance.X,index=instance.docs.idx_copy)
y_true = []
for enjeu in enjeux_list:
    y_true.append(final['True_'+enjeu].apply(lambda x: cleanstrtobool(x)).values)

y_true = np.matrix(y_true).T
X_sub = X_df.iloc[final.idx_copy.values]
#%%
y,X = separate(instance.docs,instance.X,df_corrige=corrige)
ydf = pd.DataFrame(y,columns=instance.enjeux_list)
Xdf = pd.DataFrame(X,columns=instance.vocab)
from Pipeline.Enjeux.multilabel_balancing import get_tail_label
ydf['Biodiversité'].value_counts()
#%%
#Enregistrement du modèle
pickle.dump(instance,open('Data\Enjeux\\modele_avis.pickle','wb'))
#%%
#Exploratoire : optimiser le poids de chaque classifieur
import torch
y_true = torch.tensor(instance.y_sub.to_numpy())
y_predict = torch.tensor(instance.predict(np.matrix(instance.X_sub)).astype(float),requires_grad=True)

def optimize_weights(model,loss, epoch = 10):
    import torch
    import numpy as np
    from Pipeline.Enjeux.boosting import F1_loss    
    from torch.utils.data import DataLoader,RandomSampler
    labels = torch.tensor(model.y_sub.to_numpy())
    n = 10
    #x0 = np.random.rand(n)
    #x0 = x0/x0.sum()
    x0 = np.array([1/n]*n)
    x0 = [torch.tensor(w,requires_grad=True) for w in x0]
    sortie = torch.tensor(model.predict(np.matrix(model.X_sub),
                    weights= [w.detach().numpy() for w in x0]).astype(float))
    l = [loss(sortie,labels)]
    optimizer = torch.optim.Adam(x0,lr=0.0005,amsgrad=True)
    losses = [l[0]]
    for e in range(epoch):
        optimizer.zero_grad()
        sortie = torch.tensor(model.predict(np.matrix(model.X_sub),
                    weights= [w.detach().numpy() for w in x0]).astype(float))
        l = loss(sortie,labels)
        l.backward()
        grad = l.grad
        losses.append(t,l.data[0])
        l.requires_grad = True
        xg = []
        for w in x0:
            w.backward()
            xg.append(w.grad)
        #for k in range(len(x0)):
           # x0[k] += xg[k]*
        optimizer.step()

    print(losses)
    return(optimizer,losses)

from Pipeline.Enjeux.boosting import F1_loss
loss = F1_loss()

opt,losses = optimize_weights(instance,loss,epoch=100)
t=0
for w in opt.param_groups[0]['params']:
    t += w.data

#%%

#Tentative d'optimiser les poids de chaque classifieur pour obtenir un meilleur résultat
#Inefficace pour le moment
obj = instance.optimize_weights(method = 'SLSQP')
#Prédiction des présences ou non des topics dans le corpus
prediction = instance.predict(instance.X,weights=obj.x)
#Evaluation des résultats (precision, accuracy, recall, F1, pour chaque enjeu)
sc2 = evaluate(docs_df,prediction,metadata = metadata,returnscore=True)

#%%

#Optimisation de la sélectivité. Attention a garder une valeur cohérente ! Modifier les bounds (bnds)
#pour garder une sélectivité raisonnable
obj2 = instance.optimize_selectivity(bnds=(0.1,0.9))
prediction = instance.predict(instance.X,selectivity=obj2.x)
sc3 = evaluate(docs_df,prediction,metadata = metadata,returnscore=True)
#%%
prediction = instance.predict(instance.X)
sc1 = evaluate(docs_df,prediction,metadata = metadata,returnscore=True)

#Différence de scores entre le score 2 et le score 1 
# score 1 = initial, score 2 = après modification, on fait final - initial
delta(sc1,sc2,returnmoy=True)

#%%

####TOUT CE QUI EST DESSOUS = PISTES D'AMELIORATIONS
#Parfois un peu brouillon encore

def sample_add(X,sc1,X_sub2,vocab,anchor,strength,enjeux_list,num_average = 20):
    """
    Pour ajouter des samples sur la base du delta des scores.
    Renvoie True si le F1 est amélioré.
    """
    deltamoy = [0,0,0,0]
    for it in range(num_average):
        model2 = ct.Corex(n_hidden=len(enjeux_list))
        model2.fit(X_sub2, words=vocab, anchors=anchor, anchor_strength=strength)
        #Prédiction et évaluation sur sur toutes les données
        test = model2.predict(X)
        sc2 = evaluate(test,metadata = metadata,returnscore=True)

        #Calcul du delta des évaluations
        deltamoy = vadd(deltamoy,delta(sc1,sc2,returnmoy=True))
    deltamoy = np.array(deltamoy)/num_average
    if deltamoy[3]>0:
        return(True,sc2)
    else:
        return(False,sc1)


# %%
#On va enrichir le thésaurus automatiquement en ajoutant les mots
#sémantiquement (word2vec) reliés aux mots de notre thésaurus
#On évaluera (auditeurs) l'interprétabilité des nouvelles listes

from Pipeline.Enjeux.enrichissement import *

inst = makesimilarity(dico_thesau,cosimilarite=pickle.load(open("Data\Enjeux\cosimilarite_avis.pickle",'rb')))
add = inst.top_words_topic(100)

from Pipeline.Enjeux.processing_encoding import processing_mot

#%%
idx = {}
for k in range(len(enjeux_list)):
    idx[enjeux_list[k]] = k
worst_enj = 'Gestion des déchets'
keep = []

new_thesau = thesau_list.copy()
for k in range(100):
    new_thesau = thesau_list.copy()
    corr = processing_mot(add[worst_enj][k])
    i = idx[worst_enj]
    print(corr)
    if corr not in vocab:
        print('On passe')
        continue
    new_thesau[i].append(corr)
    print('Enrichissement',worst_enj)

    model = ct.Corex(n_hidden=len(enjeux_list))
    model.fit(np.matrix(X), words=vocab, anchors=new_thesau, anchor_strength=5)

    #Prédiction et évaluation sur sur toutes les données
    test = model.predict(X)
    sc2 = evaluate(docs_df,test,metadata = metadata, returnscore=True)
    d =delta(sc1,sc2,returnmoy=True)
    
    if sc2[worst_enj][3]-sc1[worst_enj][3]>0:
        keep.append(corr)

#%%
worst_enj = 'Gestion des déchets'


new_thesau = thesau_list.copy()
for k in range(1,100):
    new_thesau = thesau_list.copy()
    print('\n')
    print('...................',k)
    i= 0
    for enjeu,thesau in zip(add,thesau_list):
        print(enjeu)
        if enjeu == worst_enj:
            
            corr = processing_mot(add[enjeu][k])
            print(corr)
            if corr not in vocab:
                print('On passe')
                pass
            new_thesau[i].append(corr)
        i+=1
    model = ct.Corex(n_hidden=len(enjeux_list))
    model.fit(np.matrix(X), words=vocab, anchors=new_thesau, anchor_strength=3)

    #Prédiction et évaluation sur sur toutes les données
    test = model.predict(X)
    sc2 = evaluate(docs_df,test,metadata = metadata,returnscore=True)
    worst_enj = ''
    dummy = [1,1,1,1]
    for e in sc2:
        if sc2[e][3]<dummy[3]:
            worst_enj = e
            dummy[3] = sc2[e][3]
    moy = delta(sc1,sc2,returnmoy=True)

#%%%
ress = []
for row1,row2 in zip(topic_model[k].p_y_given_x,topic_model[k].labels):
    line = []
    for el1,el2 in zip(row1,row2):
        line.append((el1,el2))
    ress.append(line)
ress = np.matrix(ress)

#%%
#On va essayer d'optimiser en ajoutant un enjeux poubelle vide

deltamoy = [0,0,0,0]
for it in range(20):
    modelPoub = ct.Corex(n_hidden=len(enjeux_list)+1)
    modelPoub.fit(X, words=vocab, anchors=thesau_list, anchor_strength=2+1)

    #Prédiction et évaluation sur sur toutes les données
    test = modelPoub.predict(X)
    sc2 = evaluate(docs_df,test,metadata = metadata,returnscore=True)

    #Calcul du delta des évaluations
    deltamoy = vadd(deltamoy,delta(sc1,sc2,returnmoy=True))
deltamoy = np.array(deltamoy)/20 

#deltamoy = [0.02,0.01,-0.02,-0.01]
#En moyenne (20 it), les performances sont moins bonnes !
#%%
#On va tenter de seeder le topic inutile
thesau_list_poub = thesau_list.copy().append(['mrae','administratif','region',
'autorite environnemental','dreal',
    'loi','reglemen'])
deltamoy = [0,0,0,0]
for it in range(20):
    modelPoub2 = ct.Corex(n_hidden=len(enjeux_list)+1)
    modelPoub2.fit(X, words=vocab, anchors=thesau_list_poub, anchor_strength=k+1)

    #Prédiction et évaluation sur sur toutes les données
    test = modelPoub2.predict(X)
    sc2 = evaluate(test,metadata = metadata,returnscore=True)
    sc1 = evaluate(topic_model[2].labels,metadata = metadata,returnscore=True)

    #Calcul du delta des évaluations
    
    deltamoy = vadd(deltamoy,delta(sc1,sc2,returnmoy=True))

deltamoy = np.array(deltamoy)/20 

#deltamoy = [0.03,0.02,-0.04,-0.05]
#En moyenne (20 it), les performances sont moins bonnes !