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
import os
# Tout d'abord donc il nous faut des phrases appartennant à un document (article, paragraphe etc...).
# Pour le moment, on va télécharger les articles d'OrangeSum, on va essayer le TextRank pour voir si on retrouve celles (les phrases) qui ont
# été labellisées comme proche du résumé fourni (abstractif).

os.chdir("C:/Users/theo.roudil-valentin/Documents/Resume/MLSUM/")
from fats import Make_Extractive
cpu=psutil.cpu_count()
#%%
Art=[]
Sum=[]
for k in range(4):
    articles=pickle.load(open('text_clean_train_'+str(k+1)+'.pickle','rb'))
    Art+=articles
    print(len(articles))
    headings=pickle.load(open('summary_clean_train_'+str(k+1)+'.pickle','rb'))
    print(len(headings))
    Sum+=headings
print(len(Sum),len(Art))
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

#On va entrainer un tokenizer :
ME=Make_Extractive(cpu=cpu)
tok=ME.make_tokenizer(Art,12000,'MLSUM_tokenizer',name='MLSUM_tokenizer')


# On a déjà un Tokenizer entraîné, donc on va le réutiliser :
#On rentre notre SentencePiece model dans le tokenizer Camembert
#%%
from transformers import CamembertTokenizer
tok=CamembertTokenizer('MLSUM_tokenizer.model')
# %%
# On charge le modèle BERT 
from transformers import CamembertModel,BertModel,RobertaModel
camem=CamembertModel.from_pretrained("camembert-base")
camem.config.hidden_size
#%%
camem.eval()
encod=tok(articles[0][0:1])
input_id=torch.tensor(encod['input_ids'])
att_mask=torch.tensor(encod['attention_mask'])
# %%
embedding=camem(input_id,att_mask)
# %%
class Make_Embedding():
    def __init__(self,tok=None,cpu=None) -> None:
        super(Make_Embedding,self).__init__
        self.tokenizer=tok
        self.cpu=cpu

    def make_token(self,sequence):
        tokens=self.tokenizer(sequence)
        input_ids=tokens['input_ids']
        att_mask=tokens['attention_mask']
        return input_ids,att_mask

    def make_tokens(self,sequence):
        tokens=Parallel(n_jobs=self.cpu)(delayed(self.make_token)(z) for z in sequence)
        dico={}
        dico['input_ids']=[tokens[i][0] for i in range(len(tokens))]
        dico['attention_mask']=[tokens[i][1] for i in range(len(tokens))]
        return dico

    @staticmethod
    def emb_phrase(input_id,att_mask,cam):
        embeddings=[]
        for i,a in zip(input_id,att_mask):
            embedding=cam(torch.tensor(i).unsqueeze(1),torch.tensor(a).unsqueeze(1))
            embeddings.append(embedding[0].mean(dim=1).squeeze(-2))
        return embeddings
    
    def emb_phrases(self,input_ids,att_masks,cam):
        for input_id,att_mask in zip(input_ids,att_masks):
            embeddings=self.emb_phrase(input_id,att_mask,cam)
        return embeddings
#%%
MEm=Make_Embedding(tok=tok,cpu=psutil.cpu_count())

start=time.time()
tokens=MEm.make_tokens(articles)
end=time.time()
print("Durée de la tokenization des articles :",round((end-start)/60,2),"minutes")
pickle.dump(tokens,open('tokens.pickle','wb'))
#%%

MEm=Make_Embedding()

cpu=psutil.cpu_count()
print("Utilisation de",cpu,"coeurs")
emb_p=partial(MEm.emb_phrase,cam=camem)

start=time.time()
embedding_phrase=Parallel(n_jobs=cpu)(delayed(emb_p)(i,j) for i,j in zip(tokens['input_ids'][:10],tokens['attention_mask'][:10]))
end=time.time()
print("Durée de l'embedding de 10 articles :",round((end-start)/60,2),"minutes")
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
pickle.dump(embedding_section,open('MLSUM_embedding_articles_train_1.pickle','wb'))
#%%
embedding_section=pickle.load(open('embedding_section_bert.pickle','rb'))
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

# TextRank(matrix[0])
#%%
cpu=psutil.cpu_count()

TextRank_scores_bert=Parallel(n_jobs=cpu)(delayed(TextRank)(mat) for mat in matrice_sim)
pickle.dump(TextRank_scores_bert,open(chemin_d+'OrangeSum_TextRank_scores_BERT.pickle','wb'))

TextRank_scores_w2v=Parallel(n_jobs=cpu)(delayed(TextRank)(mat) for mat in matrice_sim_w2v)
pickle.dump(TextRank_scores_w2v,open(chemin_d+'OrangeSum_TextRank_scores_W2V.pickle','wb'))
#%%
matrix=pickle.load(open(chemin_d+'OrangeSum_similarity_matrix_W2V.pickle','rb'))
#%%
mat=matrix[0]
un=torch.ones(mat.shape[0])
wei=torch.div(un,mat.sum(dim=1))
wei=torch.repeat_interleave(wei,mat.shape[0],dim=0).reshape(mat.shape)
mat_div=torch.mul(mat,wei)
score=mat_div@torch.ones(mat.shape[0])#torch.mul(mat_div,torch.ones(mat.shape[0]).unsqueeze(1))
score#.sum(dim=1)
#%%
############# 3.2 - TextRank from scratch

def WS(mat_sim,score,d=0.85):
    un=torch.ones(mat_sim.shape[0])
    weights=torch.div(un,mat_sim.sum(dim=1))
    weights=torch.repeat_interleave(weights,mat_sim.shape[0],dim=0).reshape(mat_sim.shape)
    weights=torch.mul(mat_sim,weights)
    score=torch.mul(d,score)
    score=torch.add((1-d),score)
    return score

WS(matrix[0],torch.rand(matrix[0].shape)).sum(dim=1)
#%%
def make_convergence(mat,score,epsilon=0.001):
    # score_1=torch.ones(mat.shape[0])
    score_1=WS(mat,score)
    while torch.add(score_1,-score).sum()>epsilon:
        print('encore')
        score=score_1
        score_1=WS(mat,score)
    return score_1

make_convergence(matrix[0],torch.rand(matrix[0].shape[0]))



#%%
vrai=[]
for k in range(len(matrix[0])):
    mat=list(matrix[0][k])
    colonne=[matrix[0][i][k] for i in range(matrix[0].shape[0])]
    print(np.sum(mat),np.sum(colonne))
    vrai.append(np.sum(mat)==np.sum(colonne))
np.sum(vrai)/len(matrix)
# colonne






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
