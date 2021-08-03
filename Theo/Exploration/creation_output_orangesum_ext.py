#%%
#########################################################################################################
#### Création d'un vecteur de similarité article/résumé pour supervisation extractive ###################
#########################################################################################################


chemin_d="C:/Users/theo.roudil-valentin/Documents/OrangeSum/"
import time
import pickle
from pathlib import Path
import gensim
import pandas as pd
import numpy as np
import sklearn
import re
from unidecode import unidecode
import functools
import operator
#%%
with open(chemin_d+"article.txt",'r') as ouais:
    article=ouais.read()
print(len(article))
articles=article.split('\n')
print(len(articles))
articles=[i.split('.') for i in articles] #On coupe pour avoir une liste des listes de phrases de chaque article
print(len(articles))
articles=[i for i in articles if (len(i[0])>0)]
#  [and np.sum([int(len(z)==0) for z in i])==0] #On nettoie un peu
len(articles)

articles_clean=[]
# output_clean=[]
for i in articles:
    arti_clean=[]
    # output_clean_=[]
    # output=output_[articles.index(i)]
    for k in range(len(i)):
        if len(i[k])>0:
            arti_clean.append(i[k])
            # output_clean_.append(output[k])
    articles_clean.append(arti_clean)
    # output_clean.append(output_clean_)
len(articles_clean)


def represent_word(word):
    import re
    from unidecode import unidecode
    # True-case, i.e. try to normalize sentence-initial capitals.
    # Only do this if the lower-cased form is more probable.
    text = word.replace("lire page","") #replce "lire page" par "" (en gros delete "lire page")
    text = unidecode(text) #take a unicode object and returns a string
    text=text.lower()
    text = re.sub(r'[^A-Za-z]',' ',str(text)) 
    #means any character that IS NOT a-z OR A-Z
    text = ' '.join([i for i in text.split() if len(i)>2])
    return text

## liste de type documents x phrases
articles=[[' '.join([represent_word(i) for i in articles[k][z].split() if len(i)>3])
            for z in range(len(articles[k])) if (len(articles[k][z].split())>0)] for k in range(len(articles))]

with open(chemin_d+"heading.txt",'r') as ouais:
    heading=ouais.read()
print(len(heading))
headings=heading.split('\n')
print(len(headings))

headings=[' '.join([represent_word(i) for i in headings[k].split() if len(i)>3]) for k in range(len(headings))]
print(len(headings))

ouais=pd.DataFrame(headings,columns=['ouais'])
index_vide=list(ouais[[True if len(ouais.ouais[i])==0 else False for i in ouais.index]].index)
print(index_vide)


for i in sorted(index_vide,reverse=True):
    try:
        articles.remove(articles[i])
    except:
        print(i)
        continue

print(len(articles))
headings=[i for i in headings if len(i)>0]
print(len(headings))

#%%
pickle.dump(articles,open(chemin_d+"OrangeSum_articles_phrases_clean.pickle",'wb'))
pickle.dump(headings,open(chemin_d+"OrangeSum_headings_phrases_clean.pickle",'wb'))

#%%
#Création du texte entier pour l'ensemble des phrase "résumé" et article
texte_entier_article=[' '.join([i for i in articles[k]]) for k in range(len(articles))]
print(texte_entier_article[0])
texte_entier=[i+' '+j for i,j in zip(texte_entier_article,headings)]
print(texte_entier)
pickle.dump(texte_entier,open(chemin_d+"OrangeSum_texte_entier_clean.pickle",'wb'))

#%%
from gensim.test.utils import common_texts
print("Format de l'input d'un modèle Word2Vec:",common_texts)
#%%
sentence=[]
for i in articles:
    for z in i:
        if len(z.split())>0:
            sentence.append(z.split())
    a=headings[articles.index(i)].split()
    if len(a)>0:
        sentence.append(a)

# sentences =np.array([c for c in texte_entier])
print(sentence)

#%%
#On crée et entraine le modèle d'embedding
fenetre=30
minimum=1
d=300
W2V=gensim.models.Word2Vec(size=d,window=fenetre,min_count=minimum)
W2V.build_vocab(sentence)
W2V.train(sentence,total_examples=W2V.corpus_count,epochs=25)
#%%
def euclid(x):
    import numpy as np
    d=np.sqrt(sum([i**2 for i in x]))
    return d

def cos_sim(x,y):
    a=x@y
    l=euclid(x)*euclid(y)
    sim=a/l
    return sim
#%%
###### Essai 

doc=articles[0]
p_=headings[0]
Alpha_=[]
Lambda=[]
for p in doc:
    Alpha=[]
    for w1 in p.split():
        Alpha.append(
            np.mean(
                [cos_sim(W2V[w1],W2V[w2])
                 for w2 in p_.split()]))# if np.isnan(cos_sim(W2V[w1],W2V[w2]))!=True]))
    Alpha_.append(Alpha)
    Lambda.append(np.mean(Alpha))
#%%
sorted(Lambda)[-5:]
ind=[Lambda.index(i) for i in sorted(Lambda)[-5:] if np.isnan(i)!=True]
for i in ind:
    print(articles[0][i])
print(headings[0])
#%%
def similarite_alpha(phrase,resume,W2V):
    '''
    Cette fonction calcule la similarité moyenne de chaque mot d'une phrase avec chaque mot
    du résumé. Elle produit un vecteur de similarité.
    '''
    Alpha=[]
    from joblib import Parallel,delayed
    from functools import partial
    import psutil
    import time
    cpu=psutil.cpu_count()
    # start_=time.time()
    for j in phrase.split(): #Pour chaque mot de la phrase on va calculer la similarité
        # start=time.time()
        do=partial(cos_sim,y=W2V[j])
        Alpha.append(np.mean(Parallel(n_jobs=cpu)(delayed(do)(W2V[i]) for i in resume.split())))
        # Alpha.append(np.mean([cos_sim(W2V[i],W2V[j]) for i in resume.split()]))
        # end=time.time()
        # print(end-start)
    # end_=time.time()
    # print((end_-start_)/60)
    return Alpha
#Alpha=similarite(articles[0][0],headings[0],W2V)

def similarite_lambda(doc,resume,W2V):
    '''
    Pour un document associé à un résumé, la fonction nous renvoie un vecteur d'importance (similarité)
    des phrases : les phrases proches du résumé ont un coefficient élevé.    
    '''
    from joblib import Parallel,delayed
    from functools import partial
    import time
    import psutil
    cpu=psutil.cpu_count()
    # start=time.time()
    Alpha_=Parallel(n_jobs=cpu)(delayed(similarite_alpha)(doc[i],resume,W2V) for i in range(len(doc)))
    # Alpha_=[similarite_alpha(doc[i],resumre,W2V) for i in range(len(doc))]
    # Lambda=[np.mean(similarite_alpha(doc[i],resume,W2V)) for i in range(len(doc))]
    Lambda=[np.mean(i) for i in Alpha_]
    # end=time.time()
    # print((end-start)/60)
    return Lambda
# m=similarite_lambda(articles[0],headings[0],W2V) #meilleure phrase d'un docu
# m
#%%
def mphrases(doc,Lambda,k=5):
    meilleures_phrases=[doc[Lambda.index(i)] for i in sorted(Lambda)[-k:] if np.isnan(i)!=True]
    return meilleures_phrases
#%%
import pickle
from joblib import Parallel,delayed
from functools import partial
import time
import psutil
cpu=psutil.cpu_count()
do=partial(similarite_lambda,W2V=W2V)
start=time.time()
Output_OrangeSum=Parallel(n_jobs=cpu)(delayed(do)(a,r) for a,r in zip(articles,headings))
end=time.time()
print("Durée :",(end-start)/60)
pickle.dump(Output_OrangeSum,open(chemin_d+"Output_OrangeSum.pickle",'wb'))
#%%
#Ok on a un ouput pour chaque document 
#Mais le nombre de phrases n'est pas le même -> comment on fait ?
#En plus les documents sont constitués de beaucoup de phrases -> quel type d'embedding ?
#sachant que dim<513