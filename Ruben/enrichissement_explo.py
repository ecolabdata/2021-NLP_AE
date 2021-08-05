#%%
import numpy as np

def entropie(proba):
    """
    arguments :
        proba = vecteur des fréquences des features, numpy
    """
    log = np.log2(proba)
    res = -np.sum(proba*log)
    return(res)


def AFE(cooc,topics,word2id,smooth = 1):
    """
    features = matrice des occurences document-termes
    seeds = liste des listes de mots seed
    """
    m = cooc.shape[0]
    somme = 0
    #np.fill_diagonal(cooc,0)
    tot = cooc.sum(axis =1)
    cooc = (cooc+smooth)/tot
    for topic in topics:
        mat = cooc[[word2id[word] for word in topic],:]
        F = mat.sum(axis=0)
        somme += entropie(F)
    somme = somme/m
    return(somme)

def coverage(topic,word2id,features):
    covs = []
    sub = features[:,[word2id[word] for word in topic]]
    covs = sub.sum(axis = 1)
    covs = covs/covs
    covs = np.nan_to_num(covs,0)
    coverage = covs.sum()/features.shape[0]
    return(coverage)


def coverage_all(topics,word2id,features):
    covs = []
    for topic in topics:
        cov = coverage(topic,word2id,features)
        covs.append(cov)
    return(np.mean(covs))
#%%
mat = np.matrix([[1,2,3,3],[1,2,3,3],[1,2,3,3]])
test = np.dot(mat.T,mat)
# %%
