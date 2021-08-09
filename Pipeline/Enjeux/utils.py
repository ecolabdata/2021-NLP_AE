#%%
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

Thesaurus = pickle.load(open('Data\Enjeux\Thesaurus\Thesaurus1_clean.pickle','rb'))

enjeux_list = Thesaurus.Enjeux.values.tolist()
thesau_list = Thesaurus.Dictionnaire.values.tolist()

def scores(y_pred,y_true,labels):
    reslabel = {}
    conf = multilabel_confusion_matrix(y_true,y_pred)
    for label,mat in zip(labels,conf):
        TN = mat[0,0]
        FN = mat[1,0]
        FP = mat[0,1]
        TP = mat[1,1]
        Acc = (TP+TN)/(TP+TN+FP+FP)
        Pre = TP/(TP+FP)
        Rec = TP/(FN+TP)
        F1 = 2*Pre*Rec/(Pre+Rec)
        if TP==0:
            Pre,Rec,F1 = 0,0,0
        reslabel[label] = [Acc,Pre,Rec,F1]
    return(reslabel)

def hotgrid_score(labels,results,col = 'hot'):
    names = ['Accuracy','Precision','Recall','F1']
    resarray = []
    for lab in labels:
        resarray.append(results[lab])
    # Setting the labels of x axis.
    # set the xticks as student-names
    # rotate the labels by 90 degree to fit the names
    plt.xticks(ticks=np.arange(len(names)),labels=names,rotation=90)
    # Setting the labels of y axis.
    # set the xticks as subject-names
    plt.yticks(ticks=np.arange(len(labels)),labels=labels)
    # use the imshow function to generate a heatmap
    # cmap parameter gives color to the graph
    # setting the interpolation will lead to different types of graphs
    hm =plt.imshow(resarray, cmap=col,interpolation="nearest")
    
    plt.colorbar()
    plt.show()

def sampledistrib(y,labels = None):
    vals = y.sum(axis=0)
    try:
        vals = vals.tolist()[0]
    except:
        pass
    if labels:
        plt.bar(x = [k for k in range(len(vals))],height = vals,tick_label= labels,rotation = 90)
        plt.show()
    else:
        print('Provide labels')


def hotgrid_corr(labels,y):
    names = labels
    resarray = np.corrcoef(y)
    # Setting the labels of x axis.
    # set the xticks as student-names
    # rotate the labels by 90 degree to fit the names
    plt.xticks(ticks=np.arange(len(names)),labels=names,rotation=90)
    # Setting the labels of y axis.
    # set the xticks as subject-names
    plt.yticks(ticks=np.arange(len(labels)),labels=labels)
    # use the imshow function to generate a heatmap
    # cmap parameter gives color to the graph
    # setting the interpolation will lead to different types of graphs
    hm =plt.imshow(resarray, cmap='hot',interpolation="nearest")
    plt.colorbar()
    plt.show()
    
def vdiff(l1,l2):
    r = []
    for i1,i2 in zip(l1,l2):
        r.append(i1-i2)
    return(r)

def vadd(l1,l2):
    r = []
    for i1,i2 in zip(l1,l2):
        r.append(i1+i2)
    return(r)

def delta(sc1,sc2,returnmoy = False,showgrid = True):
    """
    Score 2 : final
    Score 1 : initial
    final - initial
    """
    diff = {}
    moy = [0,0,0,0]
    for enj in enjeux_list:
        diff[enj] = vdiff(sc2[enj],sc1[enj])
        moy = vadd(moy,vdiff(sc2[enj],sc1[enj]))
    for k in range(len(moy)):
        moy[k] = moy[k]/len(enjeux_list)
        diff['MOYENNE'] = moy
    l = np.append(enjeux_list,'MOYENNE')
    if showgrid:
        hotgrid_score(l,diff,col='seismic')
    if returnmoy:
        return(moy)


from distutils.util import strtobool

def cleanstrtobool(x):
    if type(x) != str:
        return(x)
    return(strtobool(x))


def evaluate(docs_df,y,df_corrige,returnscore = False,showgrid = True):
    """
    Entrées :
    docs_df = Dataframe avec une colonne 'id_AAE'
    y = sortie de CorEx ou CorExBoosted, sous la forme booléenne, matrice taille (n_docs,n_enjeux)
    df_corrige = 
    metadata = dataframe a ajouter avec une colonne 'id_AAE' pour faire la jointure, pour rajouter des informations
    comme par exemple l'url permettant d'accéder a une étude ou un avis
    """
    enjeux_list = [c.replace('True_','') for c in df_corrige.columns[1:]]
    labels = pd.concat([docs_df,pd.DataFrame(y[:,:len(enjeux_list)],columns =enjeux_list)],axis=1)
    labels.dropna(inplace = True)
    labels.id_AAE = labels.id_AAE.astype(int) # Pour s'assurer que la jointure se fasse bien : même type de données
    final =df_corrige.merge(labels, on = 'id_AAE', how='inner')
    
    y_pred = []
    y_true = []
    for enjeu in enjeux_list:
        y_pred.append(final[enjeu].apply(lambda x: cleanstrtobool(x)).values)
        y_true.append(final['True_'+enjeu].apply(lambda x: cleanstrtobool(x)).values)

    y_pred = np.matrix(y_pred).T
    y_true = np.matrix(y_true).T
    
    sc = scores(y_pred,y_true, labels = enjeux_list)

    moy = [0,0,0,0]
    for enj in enjeux_list:
        moy = vadd(moy,sc[enj])
    for k in range(len(moy)):
        moy[k] = moy[k]/len(enjeux_list)

    sc['MOYENNE'] = moy
    l = np.append(enjeux_list,'MOYENNE')
    if showgrid:
        hotgrid_score(l,sc,col = 'seismic')
        hotgrid_corr(enjeux_list,y_true.T)
    if returnscore:
        return(sc)

def separate(docs_df,X,df_corrige):

    enjeux_list = [c.replace('True_','') for c in df_corrige.columns[1:]]
    docs_df.dropna(inplace = True)
    docs_df.id_AAE = docs_df.id_AAE.astype(int)
    docs_df['idx_copy'] = docs_df.index
    df_corrige.dropna(inplace =True)
    final =df_corrige.merge(docs_df, on = 'id_AAE', how='inner')
    X_df = pd.DataFrame(X,index=docs_df.idx_copy)
    y_true = []
    for enjeu in enjeux_list:
        y_true.append(final['True_'+enjeu].apply(lambda x: cleanstrtobool(x)).values)

    y_true = np.matrix(y_true).T
    X_sub = X_df.loc[final.idx_copy.values]

    return(y_true,X_sub.to_numpy())


def topwords(model):
    topics = model.get_topics()
    for topic_n,topic in enumerate(topics):
        # w: word, mi: mutual information, s: sign
        topic = [(w,mi,s) if s > 0 else ('~'+w,mi,s) for w,mi,s in topic if w not in dicoThesau[enjeux_list[topic_n]]]
        # Unpack the info about the topic
        words,mis,signs = zip(*topic)    
        # Print topic
        topic_str = str(enjeux_list[topic_n])+': '+', '.join(words)
        print(topic_str)


import contextlib
import joblib
from tqdm import tqdm    
from joblib import Parallel, delayed

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close() 


# %%
