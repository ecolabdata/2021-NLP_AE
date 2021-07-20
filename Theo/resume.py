#%%
from joblib.parallel import cpu_count
import pandas as pd
import numpy as np
import torch
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
import re
from unidecode import unidecode
from bs4 import BeautifulSoup
#%%
os.chdir('C:/Users/theo.roudil-valentin/Documents/Codes')
#%%
W2V=pickle.load(open('C:/Users/theo.roudil-valentin/Documents/Resume/MLSUM/W2V_train.pickle','rb'))
#%%
sections_cool=pickle.load(open('C:/Users/theo.roudil-valentin/Documents/Resume/sections_cool.pickle','rb'))
# %%
sections=pickle.load(open('2021-NLP_AE/Data/Bagging_model/df_sections.pickle','rb'))
liste_cool=[100689,100707,102316,106168,110277,114799,118071,120638]
sections_cool=sections[[True if sections.num_etude[i] in liste_cool else False for i in sections.index]]
sections_cool['clean_2']=[BeautifulSoup(i,'html.parser').get_text() if type(i)==str else 'non' for i in sections_cool.section_html]
sections_cool['vide']=[1 if (sections_cool.clean_2[i]=='') or (sections_cool.clean_2[i]=='non') or (sections_cool.clean_2[i]=='non') 
                       else 0 for i in sections_cool.index]
sections_cool.clean_2=[unidecode(i).replace('\n',' ') for i in sections_cool.clean_2]
pickle.dump(sections_cool,open('C:/Users/theo.roudil-valentin/Documents/Resume/sections_cool.pickle','wb'))
#%%
def clean_section(section,seuil=None,decod=False):
    # section=BeautifulSoup(section,"html.parser").get_text().lower()
    if decod:
        section=unidecode(section)
    section=re.sub(r'[^A-Za-z]',' ',section).lower()
    section=re.sub(' +',' ',section)
    if seuil!=None:
        section=' '.join([i for i in section.split() if len(i)>seuil])
    return section
#%%
def remove_blank(x):
    while '' in x:
        x.remove('')
    return x

import spacy
nlp = spacy.load('fr_core_news_sm',disable=["parser","ner"]) #on charge le modèle spacy pour le français

def lemmatizer(exemple,nlp=nlp):
    article=[]
    for phrase in list(nlp.pipe(exemple)):
        phrase_2=[]
        for mot in phrase:
            # if (not mot.is_punct) and (not mot.is_stop) and 
            if (mot.pos_ in ["VERB","NOUN","ADJ","PROPN"]):
                if mot.lemma_ not in ['le','la','de','l','d','n']:
                    phrase_2.append(mot.lemma_)
        article.append(' '.join([i for i in phrase_2]))
    return article

# #%%
# for doc in nlp.pipe(exemple):
#     # print(doc)
#     for w in doc:
#         if (not w.is_punct) and (not w.is_stop):
#             print(w.lemma_)
#%%
num=105
print(sections_cool[sections_cool.num_etude==100689].titres[num])
exemple=sections_cool[sections_cool.num_etude==100689].clean_2[num]
exemple
#%%
exemple=remove_blank(sections_cool[sections_cool.num_etude==100689].clean_2[101].split('.'))[:-5]
article=[clean_section(exemple[i],seuil=2) for i in range(len(exemple))]
exemple=remove_blank(article)
exemple
#%%
from fats import TextRank,BERTScore,Lead_3,Random_summary
nphrase=2
# resume=[]
TR=TextRank()
BS=BERTScore()
TRB=TR.make_resume(exemple,'bert',k=nphrase)
TRW=TR.make_resume(exemple,'word2vec',W2V=W2V,k=nphrase)
BSR=BS.make_summary(exemple,k=nphrase)
L3=Lead_3(exemple,k=2)
RS=Random_summary(exemple)
resume={'numero' : 100689,'section' : (101,sections_cool[sections_cool.num_etude==100689].titres[101]),
'texte': exemple, 'TextRank_W2V':TRW,'TextRank_Bert' : TRB, 'BertScore' : BSR,
'Lead-3' : L3, "aléatoire" : RS}
#%%
article=lemmatizer(article)
remove_blank(article)

#%%
k=90
for i in sections_cool[sections_cool.num_etude==100689].clean_2[100+(k-10):k+100]:
    indice=list(sections_cool[sections_cool.num_etude==100689].clean_2).index(i)
    print("\n ################################ \n",indice,sections_cool[sections_cool.num_etude==100689].titres[indice],"\n",i)
#%%
exemple_liste=[99,101,105,110,119,123,121,120,137,133,169,168,167,178,179,187 ]
for i in exemple_liste:
    # sections_cool[sections_cool.num_etude==100689].clean_2).index(i)
    print("\n ################################ \n",i,sections_cool[sections_cool.num_etude==100689].titres[i],"\n",sections_cool[sections_cool.num_etude==100689].clean_2[i])
#%%
os.chdir('C:/Users/theo.roudil-valentin/Documents/Codes/2021-NLP_AE/Theo')
from fats import TextRank,BERTScore,Lead_3,Random_summary
nphrase=2
TR=TextRank()
BS=BERTScore()
#%%
resume=[]
exemple_=[]
exemple_vrai=[]
for i in exemple_liste:
    exemple=sections_cool[sections_cool.num_etude==100689].clean_2[i]
    exemple_vrai.append(exemple)
    exemple=[clean_section(i,seuil=2,decod=True) for i in exemple.split('.')]
    # exemple=lemmatizer(exemple)
    exemple=remove_blank(exemple)
    exemple_.append(exemple)
    TRB=TR.make_resume(exemple,'bert',k=nphrase)
    TRW=TR.make_resume(exemple,'word2vec',W2V=W2V,k=nphrase)
    BSR=BS.make_summary(exemple,k=nphrase)
    L3=Lead_3(exemple,k=2)
    RS=Random_summary(exemple)
    resume.append({'numero' : 100689,'section' : (i,sections_cool[sections_cool.num_etude==100689].titres[i]),
    'texte': exemple, 'TextRank_W2V':TRW,'TextRank_Bert' : TRB, 'BertScore' : BSR,
    'Lead-3' : L3, "aléatoire" : RS})
#%%
resume[exemple_liste.index(169)]
#%%
ex_emb=torch.tensor(W2V[[w for w in resume[exemple_liste.index(178)]['texte'][0].split()]]).mean(dim=0)
ex_emb
#%%
pd.DataFrame.from_dict(resume).to_csv('C:/Users/theo.roudil-valentin/Documents/Resume/resume_exemple.csv',sep=';',index=False)
# pickle.dump(pd.DataFrame.from_dict(resume),open('C:/Users/theo.roudil-valentin/Documents/Resume/resume_exemple.pickle','wb'))
#%%
pd.DataFrame.from_dict(resume).to_csv('K:/03 ECOLAB/2 - POC DATA et IA/Données NLP/Data/resume_exemple.csv',sep=';',index=False)
#%%
resume=pd.read_csv('C:/Users/theo.roudil-valentin/Documents/Resume/resume_exemple.csv',sep=';')

# %%
#####################################################################################
# TextRank W2V

os.chdir('C:/Users/theo.roudil-valentin/Documents/Codes/2021-NLP_AE/Theo')
from fats import TextRank#,BERTScore,Lead_3,Random_summary
nphrase=2
TR=TextRank()
pabo=[]
resume_TR=[]
temps=[]

for i in tqdm(range(sections_cool.shape[0])):
    exemple=sections_cool.clean_2[i]
    l=len(exemple)

    if (l>200) and (l<30000):
        exemple=[clean_section(o,seuil=2,decod=True) for o in exemple.split('.')]
        exemple=remove_blank(exemple)
        
        start=time.time()
        try:
            TRR=TR.make_resume(exemple,'word2vec',W2V=W2V,k=nphrase,verbose=0)
        except:
            pabo.append(i)
            TRR=['vide']
        end=time.time()
        temps.append(end-start)

    else:
        TRR=['longueur']
    
    resume_TR.append(TRR)

sections_cool['resume']=resume_TR

sections_cool.to_csv('C:/Users/theo.roudil-valentin/Documents/Resume/sections_cool_avec_resume.csv',sep=';',index=False)
#%%
sections_cool['longueur']=[1 if (len(k)>200) and (len(k)<30000) else 0 for k in sections_cool.clean_2]
sections_cool['pour_resume']=[remove_blank(
    [clean_section(
        o,seuil=2,decod=True) for o in k.split('.')]) for k in sections_cool.clean_2]

def resum(exemple,TR,W2V):
    try:
        TRR=TR.make_resume(exemple,'word2vec',W2V=W2V,k=nphrase,verbose=0)
    except:
        TRR=['vide']
    return TRR
#%%
cpu=psutil.cpu_count()
print(cpu)
resu=partial(resum,TR=TR,W2V=W2V)
resume=Parallel(n_jobs=cpu)(delayed(resu)(i) for i in tqdm(sections_cool[sections_cool.longueur==1].pour_resume.values))
#%%
sections_cool_1=sections_cool.reset_index()[sections_cool.reset_index().longueur==1]
sections_cool_1['resume']=resume
sections_cool_0=sections_cool.reset_index()[sections_cool.reset_index().longueur==0]
sections_cool_0['resume']=[['tropcourt'] for i in range(sections_cool_0.shape[0])]
sections_cool_avecresume=pd.concat([sections_cool_0,sections_cool_1])
sections_cool_avecresume.to_csv('C:/Users/theo.roudil-valentin/Documents/Resume/sections_cool_avecresume.csv',sep=';',index=False)


##########################################################################################
#%%
ouais=pd.read_csv('C:/Users/theo.roudil-valentin/Documents/Resume/sections_cool_avecresume.csv',sep=';')
#%%
def string_to_list(x,k=2):
    try:
        x=[x.strip('][').split(',')[i].strip("''") for i in range(k)]
    except:
        x=[x.strip('][').split(',')[i].strip("''") for i in range(1)]
    return x

def resumenul(x):
    try:
        y=[x[0]]
        for i in range(1,len(x)):
            y=y+[x[i][2:]]
    except:
        y=x
    return y

def make_index_resume(x,z,k=2):
    try:
        y=[x.index(z[i]) for i in range(k)]
    except:
        y=[np.nan]
    return y

def resume_viaindex(x,z,h,k=2):
    try:
        y=[]
        for i in range(k):
            y+=[x.split('.')[z[i]]]
    except:
        y=h
    return y


ouais['resume_2']=[string_to_list(x) for x in ouais.resume]
ouais['pour_resume_2']=[string_to_list(x,k=len(x.split("',"))) for x in ouais.pour_resume]
ouais.pour_resume_2=[resumenul(x) for x in ouais.pour_resume_2]
ouais.resume_2=[resumenul(x) for x in ouais.resume_2]
ouais['index_resume']=[make_index_resume(x,z) for x,z in zip(ouais.pour_resume_2,ouais.resume_2)]
ouais['resume_propre']=[resume_viaindex(ouais.clean_2[i],ouais.index_resume[i],ouais.resume_2[i]) for i in ouais.index]
#%%
ouais.to_csv('C:/Users/theo.roudil-valentin/Documents/Resume/sections_cool_avecresume.csv',sep=';',index=False)

#%%
os.chdir("C:/Users/theo.roudil-valentin/Documents/Resume/MLSUM/")
cpu=psutil.cpu_count()
#%%
Art=[]
Sum=[]
for k in range(1):
    articles=pickle.load(open('text_clean_train_'+str(k+1)+'.pickle','rb'))
    Art+=articles
    print(len(articles))
    headings=pickle.load(open('summary_clean_train_'+str(k+1)+'.pickle','rb'))
    print(len(headings))
    Sum+=headings
#%%
from transformers import CamembertTokenizer
import networkx as nx
#%%
W2V=pickle.load(open('C:/Users/theo.roudil-valentin/Documents/Resume/MLSUM/W2V_train.pickle','rb'))
#%%
class Make_Embedding():
    def __init__(self,tok=None,cpu=psutil.cpu_count()) -> None:
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
            embeddings.append(embedding[0].mean(dim=0).squeeze(0))
        return embeddings
    
    def emb_phrases(self,input_ids,att_masks,cam):
        for input_id,att_mask in zip(input_ids,att_masks):
            embeddings=self.emb_phrase(input_id,att_mask,cam)
        return embeddings

class TextRank():
    def __init__(self):
        super(TextRank,self).__init__
        self.bert_embedding=Make_Embedding(tok=CamembertTokenizer('C:/Users/theo.roudil-valentin/Documents/Resume/MLSUM/MLSUM_tokenizer.model'),cpu=psutil.cpu_count())
        self.camem=CamembertModel(CamembertConfig())
    def make_embedding_bert(self,articles,camem=None):
        if camem==None:
            camem=self.camem
        dico=self.bert_embedding.make_tokens(articles)
        input_ids=dico['input_ids']
        att_mask=dico['attention_mask']
        embeddings=self.bert_embedding.emb_phrase(input_ids,att_mask,camem)
        return embeddings,dico
    
    @staticmethod
    def mat_sim(emb_2,cos_sim=torch.nn.CosineSimilarity(dim=0)):
        ouais=[[cos_sim(emb,y) for y in emb_2] for emb in emb_2]
        return torch.as_tensor(ouais)


    @staticmethod
    def get_emb_sentence(art,modele,di=0):
        word=[modele[w] for w in art]
        word=torch.as_tensor(word).mean(dim=di)
        return word

    def get_matrix_section(self,art,W2V):
        mat=[self.get_emb_sentence(art[i],W2V) for i in range(len(art))]
        mat=[torch.as_tensor(np.nan_to_num(i)) if np.isnan(i).sum()>0 else i for i in mat]
        return mat

    def make_embedding_W2V(self,article,W2V):
        article_=[article[i].split() for i in range(len(article))]
        mat=self.get_matrix_section(article_,W2V)
        return mat
    @staticmethod
    def scores(matrice_similarite,nx=nx,k=3):
        graph=nx.from_numpy_array(np.array(matrice_similarite))
        scores=nx.pagerank_numpy(graph)
        rank=sorted(scores.items(),key=lambda v:(v[1],v[0]),reverse=True)[:k]
        rank=[s[0] for s in rank]
        return rank
    
    def make_resume(self,article,type,W2V=None,k=3):
        if type=='bert':
            b,d=self.make_embedding_bert(article)
            mb=self.mat_sim(b)
            sb=self.scores(mb,k=k)
            resume=[article[i] for i in sb]
            return resume
        elif type=='word2vec':
            assert W2V!=None
            w=TR.make_embedding_W2V(article,W2V)
            mw=TR.mat_sim(w)
            sw=TR.scores(mw,k=k)
            resume=[article[i] for i in sw]
            return resume
        else:
            raise ValueError("Attention, vous devez spécifier le type d'embedding que vous voulez utiliser, soit 'bert' soit 'word2vec'.")

class BERTScore():
    def __init__(self,camem=CamembertModel(CamembertConfig()),
    cosim=torch.nn.CosineSimilarity(dim=-1)) -> None:
        super(BERTScore,self).__init__
        self.make_embedding=TextRank().make_embedding_bert
        self.camem=camem
        self.cosim=cosim

    def make_score(self,article):
        b,_=self.make_embedding(article,self.camem)
        b=torch.stack(b)
        VSA=b.mean(dim=0)
        score=self.cosim(VSA,b)
        return score
    
    def make_summary(self,article,k=3):
        score=self.make_score(article)
        score=score.topk(k=k)[1]
        resume=[article[i] for i in score]
        return resume

#%%
from transformers import CamembertModel,CamembertConfig
camem=CamembertModel(CamembertConfig())#.from_pretrained("camembert-base")
#%%
i=1
TR=TextRank()
b,d=TR.make_embedding_bert(Art[i],camem)
b
#%%
mb=TR.mat_sim(b)
sb=TR.scores(mb)
sb=[s[0] for s in sb]
#%%
w=TR.make_embedding_W2V(Art[0],W2V)
print(len(w),w[0].shape)
mw=TR.mat_sim(w)
sw=TR.scores(mw)
sw=[s[0] for s in sw]
# %%
TR=TextRank()
TR.make_resume(Art[i],type='bert',k=1)
#%%
TR.make_resume(Art[i],type='word2vec',W2V=W2V,k=1)

#%%
BS=BERTScore()
BS.make_score(Art[i])
#%%
BS.make_summary(Art[i],k=1)
#%%
class BERTScore():
    def __init__(self,camem=CamembertModel(CamembertConfig()),
    cosim=torch.nn.CosineSimilarity(dim=-1)) -> None:
        super(BERTScore,self).__init__
        self.make_embedding=TextRank().make_embedding_bert
        self.camem=camem
        self.cosim=cosim

    def make_score(self,article):
        b,_=self.make_embedding(article,self.camem)
        b=torch.stack(b)
        VSA=b.mean(dim=0)
        score=self.cosim(VSA,b)
        return score
    
    def make_summary(self,article):
        score=self.make_score(article)
        score=score.topk(k=3)[1]
        resume=[article[i] for i in score]
        return resume



# %%
