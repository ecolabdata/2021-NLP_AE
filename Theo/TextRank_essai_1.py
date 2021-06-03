######################################################################################################################################################
#####         Text Rank essai      ################################################################################################################################
######################################################################################################################################################
#%%
import pandas as pd
import numpy as np
import torch
import nltk
import pickle
import time
import functools
import operator
from joblib import Parallel,delayed
from functools import partial
import sentencepiece as spm 
import psutil
from tqdm import tqdm
import pickle

# Tout d'abord donc il nous faut des phrases appartennant à un document (article, paragraphe etc...).
# Pour le moment, on va télécharger les articles d'OrangeSum, on va essayer le TextRank pour voir si on retrouve celles (les phrases) qui ont
# été labellisées comme proche du résumé fourni (abstractif).

chemin_d="C:/Users/theo.roudil-valentin/Documents/OrangeSum/"
#%%
articles=pickle.load(open(chemin_d+'OrangeSum_articles_phrases_clean.pickle','rb'))
print(len(articles))
headings=pickle.load(open(chemin_d+'OrangeSum_headings_phrases_clean.pickle','rb'))
print(len(headings))
# %%
######################################
#####   1. Représentation vectorielle 
######################################

############# 1.1 - Embedding BERT

# Maintenant que nous avons nos phrases et nos sorties, on cherche une manière de représenter nos phrases
# Plusieurs solutions sont possibles : Bag-of-words, embeddings etc...
# On va plutôt se tourner vers de l'Embedding, mais du coup lequel ?
# On va tester deux Embedding différentes : CamemBERT et Word2Vec
# D'autres sont possibles évidemment.

# On a déjà un Tokenizer entraîné, donc on va le réutiliser :
#On rentre notre SentencePiece model dans le tokenizer Camembert

from transformers import CamembertTokenizer
tokenizer=CamembertTokenizer(chemin_d+'FUES.model')
# %%
# On charge le modèle BERT 
from transformers import CamembertModel,BertModel,RobertaModel
camem=CamembertModel.from_pretrained("camembert-base")
camem.config.hidden_size
#%%
camem.eval()
encod=tokenizer(articles[0][0:1])
input_id=torch.tensor(encod['input_ids'])
att_mask=torch.tensor(encod['attention_mask'])
# %%
embedding=camem(input_id,att_mask)
# %%
def emb_phrase(x,tok,cam):
    encod=tok(x)
    input_id=torch.tensor(encod['input_ids'])
    att_mask=torch.tensor(encod['attention_mask'])
    embedding=cam(input_id,att_mask)
    embedding=embedding[0].mean(dim=1).squeeze(-2)
    return embedding#.tolist()

cpu=psutil.cpu_count()
print("Utilisation de ",cpu,"coeurs")
emb_p=partial(emb_phrase,tok=tokenizer,cam=camem)

start=time.time()
embedding_phrase=Parallel(n_jobs=cpu)(delayed(emb_p)([phrase]) for phrase in articles[0])
end=time.time()
print("Durée de l'embedding d'un article :",round((end-start)/60,2),"minutes")

embedding_phrase
# %%
start=time.time()

embedding_section=[]
for i in tqdm(range(len(articles))):
    emb_phrase=Parallel(n_jobs=cpu)(delayed(emb_p)([phrase]) for phrase in articles[i])
    embedding_section.append(emb_phrase)

end=time.time()
print("Durée de l'embedding de tous les articles :",round((end-start)/60,2),"minutes")

#embedding_section=torch.as_tensor(embedding_section)
embedding_section
# %%
pickle.dump(embedding_section,open(chemin_d+'embedding_section_bert.pickle','wb'))
#%%
embedding_section=pickle.load(open(chemin_d+'embedding_section_bert.pickle','rb'))
# %%
emb_len=[len(i) for i in embedding_section]
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(14,12))
ax.hist(emb_len,density=True,bins=30)
#%%
print("Il y a en tout",len(emb_len),'articles')
#%%

############# 1.2 - Embedding Word2Vec

# Ici on va faire quelque chose qui ressemble mais au lieu de prendre la moyenne
# sur les tokens, on va prendre la moyenne des mots, sur une dimension de 100 ou 300.
# D'abord, on va devoir entraîner notre modèle.
from gensim.test.utils import common_texts
print("Format de l'input d'un modèle Word2Vec:",common_texts)
#%%
articles_=[[articles[k][i].split() for i in range(len(articles[k]))] for k in range(len(articles))]
#%%
#### Entraînement du modèle W2V

import gensim

sentence=functools.reduce(operator.iconcat, articles_, [])
fenetre=30
minimum=1
d=100
W2V=gensim.models.Word2Vec(size=d,window=fenetre,min_count=minimum)
W2V.build_vocab(sentence)
W2V.train(sentence,total_examples=W2V.corpus_count,epochs=25)
#%%
#### Création des embeddings par phrase

def get_emb_sentence(art,modele=W2V,di=0):
    word=[modele[w] for w in art]
    word=torch.as_tensor(word).mean(dim=di)
    return word

def get_matrix_section(art):
    mat=[get_emb_sentence(art[i]) for i in range(len(art))]
    mat=[torch.as_tensor(np.nan_to_num(i)) if np.isnan(i).sum()>0 else i for i in mat]
    return mat

cpu=psutil.cpu_count()
embedding_sentence=Parallel(n_jobs=cpu)(delayed(get_matrix_section)(art) for art in articles_[:])
embedding_sentence
#%%
pickle.dump(embedding_sentence,open(chemin_d+'embedding_section_W2V.pickle','wb'))
#%%
embedding_sentence=pickle.load(open(chemin_d+'embedding_section_W2V.pickle','rb'))

# %%
######################################
#####   2. Matrice de similarité 
######################################

############# 2.1 - Similarité cosinus
################## 2.1.1 - avec Embedding BERT

# Une fois qu'on a l'embedding il nous faut une matrice de similarité
# C'est sur cette matrice qu'on va calculer le graph
# Pour chaque document, il va falloir créer une matrice de similarité

#### Essai avec un article 
import torch
cos_sim=torch.nn.CosineSimilarity(dim=0)
cos_sim(embedding_section[0][0],embedding_section[0][1])
# %%
# Calcul de la matrice de similarité
def mat_sim(emb_2,cos_sim=torch.nn.CosineSimilarity(dim=0)):
    ouais=[[cos_sim(emb,y) for y in emb_2] for emb in emb_2]
    return torch.as_tensor(ouais)

cpu=psutil.cpu_count()
# mat_s=partial(mat_sim,emb_2=embedding_section[0])
matrice_sim=Parallel(n_jobs=cpu)(delayed(mat_sim)(emb) for emb in embedding_section)
# matrice_sim=torch.as_tensor(matrice_sim)
matrice_sim
# %%
pickle.dump(matrice_sim,open(chemin_d+'OrangeSum_similarity_matrix_BERT.pickle','wb'))
#%%
################## 2.1.2 - avec Embedding W2V
import torch
cos_sim=torch.nn.CosineSimilarity(dim=0)
cos_sim(embedding_sentence[0][0],embedding_sentence[0][1])
#%%
# Calcul de la matrice de similarité
cpu=psutil.cpu_count()
matrice_sim_w2v=Parallel(n_jobs=cpu)(delayed(mat_sim)(emb) for emb in embedding_sentence)
matrice_sim_w2v

pickle.dump(matrice_sim_w2v,open(chemin_d+'OrangeSum_similarity_matrix_W2V.pickle','wb'))

# %%
######################################
#####   3. Calcul du graph
######################################

############# 3.1 - TextRank via PageRank de networkx

# Calcul du graph pour l'article considéré
import networkx as nx
#%%
graph=nx.from_numpy_array(np.array(matrice_sim[0]))
scores=nx.pagerank_numpy(graph)
# %%
rank=sorted(scores.items(),key=lambda v:(v[1],v[0]),reverse=True)[:3]
rank
# %%
output=pickle.load(open(chemin_d+"Output_OrangeSum.pickle",'rb'))
output=[i for i in output if len(i)>0]
print(len(output))
output=[
    np.nan_to_num(i) if np.isnan(i).sum()>0 else i for i in output]

# Comparaison avec l'output (du W2V)
torch.topk(torch.as_tensor(output[0]),k=3)
# %%
# Définition de la fonction qui va générer un modèle graph pour chaque article
# On lui donne une matrice de similarité, elle nous donne le score par phrase

def TextRank(matrice_similarite,nx=nx):
    graph=nx.from_numpy_array(np.array(matrice_similarite))
    scores=nx.pagerank_numpy(graph)
    return scores

cpu=psutil.cpu_count()

TextRank_scores_bert=Parallel(n_jobs=cpu)(delayed(TextRank)(mat) for mat in matrice_sim)
pickle.dump(TextRank_scores_bert,open(chemin_d+'OrangeSum_TextRank_scores_BERT.pickle','wb'))

TextRank_scores_w2v=Parallel(n_jobs=cpu)(delayed(TextRank)(mat) for mat in matrice_sim_w2v)
pickle.dump(TextRank_scores_w2v,open(chemin_d+'OrangeSum_TextRank_scores_W2V.pickle','wb'))

# %%
######################################
#####   4. Comparaison des résultats
######################################

############# 4.1 - Comparaison numériques
################## 4.1.1 - avec Embedding BERT

# On peut comparer à grande échelle maintenant 
range_output=[i for i in range(len(TextRank_scores_bert)) if len(output[i])>2]
output_top3=[np.array(torch.topk(torch.as_tensor(output[i]),k=3)[1]).tolist() for i in range_output]
TextRank_top3_bert=[[sorted(TextRank_scores_bert[i].items(),key=lambda v:v[1],reverse=True)[z][0] for z in range(3)] for i in range_output]
output_top3[:3],TextRank_top3_bert[:3]
# %%
def TextRank_compar(x,y):
    a=0
    for i in y:
        if i in x:
            a+=1
    a=a/3
    return a
TextRank_accuracy=Parallel(n_jobs=cpu)(delayed(TextRank_compar)(x,y) for x,y in zip(output_top3,TextRank_top3))

TextRank_accuracy=np.array(TextRank_accuracy)
# %%
TextRank_accuracy.mean()
#%%
np.sum(TextRank_accuracy==1),np.sum(TextRank_accuracy==2/3),np.sum(TextRank_accuracy==1/3),np.sum(TextRank_accuracy==0)
#%%
################## 4.1.2 - avec Embedding W2V
range_output=[i for i in range(len(TextRank_scores_w2v)) if len(output[i])>2]
output_top3=[np.array(torch.topk(torch.as_tensor(output[i]),k=3)[1]).tolist() for i in range_output]
TextRank_top3_wv=[[sorted(TextRank_scores_w2v[i].items(),key=lambda v:v[1],reverse=True)[z][0] for z in range(3)] for i in range_output]
TextRank_accuracy_W2V=Parallel(n_jobs=cpu)(delayed(TextRank_compar)(x,y) for x,y in zip(output_top3,TextRank_top3))

TextRank_accuracy_W2V=np.array(TextRank_accuracy_W2V)

print("Accuracy moyenne",TextRank_accuracy_W2V.mean())
#%%
np.sum(TextRank_accuracy_W2V==1),np.sum(TextRank_accuracy_W2V==2/3),np.sum(TextRank_accuracy_W2V==1/3),np.sum(TextRank_accuracy_W2V==0)

# %%
############# 4.2 - Comparaison à l'oeil
def sortie_resume(article,rank):
    try:
        phrases=[article[i] for i in rank]
    except:
        phrases=[]
    return phrases

nul=[]
TR_W2V_resume=[]
TR_BERT_resume=[]
Output_W2V=[]
for i in range_output:
    try:
        TR_BERT_resume.append(sortie_resume(articles[i],TextRank_top3_bert[i]))
        TR_W2V_resume.append(sortie_resume(articles[i],TextRank_top3_wv[i]))
        Output_W2V.append(sortie_resume(articles[i],output_top3[i]))
    except:
        nul.append(i)
        print(i)

# TR_W2V_resume=[sortie_resume(articles[i],TextRank_top3_wv[i]) for i in range_output]
# %%
df_resume_TextRank=pd.DataFrame([articles[:403],Output_W2V,TR_BERT_resume,TR_W2V_resume]).T
df_resume_TextRank.columns=['articles','resume_vrai_W2V','resume_TextRank_BERT','resume_TextRank_W2V']
df_resume_TextRank.to_csv(chemin_d+'TextRank_resume.csv',index=False,sep=';')
# %%
