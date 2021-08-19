#%%
import pickle
import os
from joblib import Parallel,delayed
import functools
import operator
import torch
import fats
from tqdm import tqdm
import sys
os.chdir("C:\\Users\\theo.roudil-valentin\\Documents\\Resume\\MLSUM")

index_2=pickle.load(open('index_2.pickle','rb'))
trace=pickle.load(open('test/trace_test.pickle','rb'))
score=pickle.load(open('test/score.pickle','rb'))
score=[score[i] for i in index_2]
assert len(score)==len(trace)
s=Parallel(5)(delayed(fats.make_new_paragraphes)(i,j) for i,j in zip(score,trace))
s=functools.reduce(operator.iconcat,s,[])
P=pickle.load(open('test/Paragraphes_.pickle','rb'))
assert len(P)==len(s)
s=Parallel(5)(delayed(fats.make_new_sortie)(i) for i in s)
#%%
# pickle.dump(s,open('score_final_test.pickle','wb'))
# #%%
# k=4
# N=[f for f in os.listdir('test') if (f.split('.')[-1]=='pickle') and ('sortie' in f)]
# N
#%%
def make_compa(sortie_multi,s):
    # sortie_multi=pickle.load(open(n,'rb'))

    if type(sortie_multi[0])!=torch.Tensor:
        sortie_multi=[torch.tensor(i).to(torch.long) for i in sortie_multi]

    S=[]
    erreur=[]
    if len(sortie_multi)==len(s):
        for i in range(len(s)):
            try:
                S.append(fats.make_new_sortie(s[i],sortie_multi[i]))
            except:
                print("Unexpected error:", sys.exc_info())
                break
                print(i)
                erreur.append(i)

        try:
            if erreur[0]==201:
                s_prime=s[:201]+s[202:]
                Sta=fats.comparaison(torch.cat(S),torch.cat(s_prime))
                return Sta
        except:
            Sta=fats.comparaison(torch.cat(S),torch.cat(s))
            return Sta
    else:
        print(n)
#%%
K=len(s)
ouais=pickle.load(open('test/L3_sortie.pickle','rb'))
print(make_compa(ouais[:K],s[:K]))
ouais=pickle.load(open('test/Net_sortie_4.pickle','rb'))
make_compa(ouais[:K],s[:K])
#%%
ouais=pickle.load(open('test/Net_sortie_4.pickle','rb'))
ouais=Parallel(5)(delayed(fats.make_new_sortie)(i,j) for i,j in zip(s,ouais))
fats.F1_score().true_positive_mean(
    torch.tensor(ouais[0]),
    torch.tensor(s[0]))
# %%
import numpy as np
ouais=[]
for i in range(len(s)):
    r=fats.F1_score().false_positive_mean(sortie[i],s[i])
#%%
nul=torch.stack(ouais).isnan().to(torch.long).topk(torch.stack(ouais).isnan().to(torch.long).sum())[1]
nul
# %%
fats.F1_score().false_positive_mean(sortie[0],s[0])

# %%
import gensim
W2V=gensim.models.Word2Vec.load("W2V_all.model")
TR=fats.TextRank(tok_path='MLSUM_tokenizer.model')
TR.make_resume(P[0],type='word2vec',modele=W2V,get_score_only=True,weights=True,alpha=0.2)
