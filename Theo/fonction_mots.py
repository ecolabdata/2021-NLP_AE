#%%
import pandas as pd
import numpy as np
import sklearn
import pickle
import re

chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"

df=pd.read_csv(chemin+'etudes_dataset_cleaned_prepared.csv',sep=";")
#%%
######## construction des dico de bases par thèmes

def base_theme(df):
    import numpy as np
    import re
        
    theme=list(np.unique(re.sub("[\(\[].*?[\)\]]", "",
        re.sub(","," ",
            re.sub(";"," ",' '.join(np.unique(df.theme.values))))).split(' ')))
    theme.remove('ET'),theme.remove('')
    print("Les thèmes sont :",theme)

    database={}
    for i in range(len(theme)):
        database[theme[i]]=df[[True if theme[i] in df.theme[z] else False for z in range(len(df))]]
        print("La base du thème", theme[i], "est de dimension :", database[theme[i]].shape)
    return database

####### matrice TF-IDF des documents par thèmes

def vect_theme_tfidf(database,maxf=100000,nrange=(1,2)):
    '''
    @database dictionnaire des bases de données de documents par thèmes
    '''
    from sklearn.feature_extraction.text import TfidfVectorizer

    vect = TfidfVectorizer(analyzer = "word", ngram_range=nrange, 
    tokenizer = None, preprocessor = None, max_features = maxf)
    train={}
    vect_theme={}
    for t in database.keys():
        train[t]=vect.fit_transform(database[t].docs)
        vect_theme[t]=vect
    return train,vect_theme


########### LSA

def LSA_mots(train,vect_theme,ndim=100,niter=1000,n=5):
    '''
    @train matrice TF-IDF des documents dont on veut les mots
    @vect_theme dico des vect tf-idf 
    '''
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=ndim, n_iter=niter) 
    svdm={}
    train_lsa={}
    mots_themes={}
    for t in train.keys():
        svdm[t]=svd.fit(train[t])
        train_lsa[t]=svdm[t].transform(train[t])
        mots=[]
        coeff=[]
        for i, comp in enumerate(svdm[t].components_):
            terms_comp = zip(vect_theme[t].get_feature_names(), comp)
            sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:n]
            print("Topic "+str(i)+": ")
            mots.append([sorted_terms[i][0] for i in range(len(sorted_terms))])
            print(mots[i])
            coeff.append(sorted(svdm[t].components_[i],reverse=True)[:n])
        mots_themes[t]=[[i,np.array(mots)[i],np.array(coeff)[i]] for i, _ in enumerate(svdm[t].components_)]
    return mots_themes


####### mots apparaissant le plus par thème (TF-IDF)

def Nmaxelements(list2,N): 
    final_list = []
    list1=list(list2.copy())
    for i in range(0, N):  
        max1 = 0
          
        for j in range(len(list1)):      
            if list1[j] > max1: 
                max1 = list1[j]; 
                  
        list1.remove(max1); 
        final_list.append(max1) 
          
    return final_list

def get_max_names(train_features,vect,N):
    '''
    @train_features matrice TF-IDF des documents
    @vect vect ayant servi à la transformation
    '''
    matrice=(train_features).toarray().mean(axis=0)
    maxnamesN=[]
    number_of_docs=len(matrice)
    maxn=Nmaxelements(matrice,N)
    maxnames=[]
    for k in range(N):
        maxnames.append(
            vect.get_feature_names()[
                list(matrice).index(maxn[k])])
    maxnamesN.append(maxnames)
    print(number_of_docs,' listes des', N,' plus grands éléments.')
    return maxnamesN,maxn

def mots_themes_tfidf(features,vect,N):
    '''
    @features matrice TF-IDF des documents par thèmes
    @vect vecteur ayant servi à la transformation
    '''
    mots_theme={}
    df=[]
    for t in features.keys():
        mots_theme[t]=get_max_names(features[t],vect[t],N)
    
        df_=pd.concat([pd.DataFrame.from_dict(mots_theme[t][0]).T,
                pd.DataFrame.from_dict(mots_theme[t][1])],axis=1)
        df_.columns=['mots','importance']
        df_['theme']=t
        df.append(df_)
    df=pd.concat([df[i] for i in range(len(df))])
    
    return df

####### Fonction LDA (sklearn)
def LDA_mots(train,vect_theme,k=15,n=10):        
    from sklearn.decomposition import LatentDirichletAllocation
    mots_themes={}
    for t in train.keys():
        mots=[]
        lda = LatentDirichletAllocation(n_components=k)
        lda.fit(train[t])
        coeff=[]
        for i in range(k):
            mots.append(pd.Series(
                vect_theme[t].get_feature_names())[
                    lda.components_[i].argsort()[:n]].values)
            coeff.append(sorted(lda.components_[i],reverse=True)[:n])
        mots_themes[t]=[[i,np.array(mots)[i], np.array(coeff)[i]] for i in range(k)]
    return mots_themes


####### Fonction W2V KMeans
def euclid(x):
    import numpy as np
    d=np.sqrt(sum([i**2 for i in x]))
    return d

def cos_sim(x,y):
    a=x@y
    l=euclid(x)*euclid(y)
    sim=a/l
    return sim

def W2VKMEANS_mots(database,fenetre=15,minimum=1,d=300,len_min=2,racine=3,num_clust=3):
    import gensim
    import numpy as np
    from unidecode import unidecode
    mots_themes={}
    for t in database.keys():
        print(t)
        sentences = np.array([str(c).split() for c in list(database[t].docs)],dtype=object)
        W2V=gensim.models.Word2Vec(sentences,size=d,window=fenetre,min_count=minimum)
    
        vocabulaire=[v for v in list(set(W2V.wv.vocab)) if len(v)>len_min]
        vocab_theme=[v for v in vocabulaire if v[:racine]==unidecode(t.lower())[:racine]]
        vocab_theme_wv=np.array([W2V.most_similar(z)[i][0] for z in vocab_theme for i in range(10)]).flatten()
        vocab_theme_wv=[v for v in vocab_theme_wv if len(v)>len_min]
        vocab_theme_wv #mots les plus similaires des mots qui partagent la racine du thème
        print(len(vocab_theme_wv))
        if len(vocab_theme_wv)!=0:
            vector=pd.concat([pd.DataFrame(W2V[v]).T for v in vocab_theme_wv])
            vect_moyen=np.array([W2V[v].T for v in vocab_theme]).mean(axis=0)
            vector['index']=vocab_theme_wv
            vector.set_index(keys='index',inplace=True)

            from sklearn.cluster import KMeans
            kmeansmodel=KMeans(n_clusters=num_clust,n_init=20,max_iter=500)
            kmeans=kmeansmodel.fit(vector)
            vector['label']=kmeans.labels_

            groupe_theme=np.array([vector.label[vector.index==v].values[0] for v in vocab_theme_wv if v in vector.index])
            mots_themes[t]=[vector.index[vector.label==np.bincount(groupe_theme).argmax()],
                            [cos_sim(vect_moyen,W2V[v]) for v in vector.index[vector.label==np.bincount(groupe_theme).argmax()]]]
        else:
            mots_themes[t]=['empty','empty']
    return mots_themes
# %%
num_enjeux=5
num_mots=5
database=base_theme(df)
features,vect_theme=vect_theme_tfidf(database)
#%%
# sub={k:features[k] for k in list(features.keys())[:2]}
mots_themes_LSA=LSA_mots(features,vect_theme,num_enjeux*num_mots,n=10)
#%%
base_TFIDF=mots_themes_tfidf(features,vect_theme,num_mots)

#%%
mots_themes_LDA=LDA_mots(features,vect_theme,num_enjeux,num_mots)
#%%
sub2={k:database[k] for k in list(database.keys())[:2]}
mots_themes_WKMeans=W2VKMEANS_mots(sub2)
mots_themes_WKMeans['AGRICULTURE']
# %%
theme='AGRICULTURE'
df=pd.DataFrame.from_dict([mots_themes_LSA[theme][i][1] for i in range(len(mots_themes_LSA[theme]))]).T
df.columns=["topic_"+str(i) for i in range(df.shape[1])]
df=df.reset_index()
df=df.melt(id_vars=['index'],var_name='topic',value_name='mots')
#%%
imp=pd.DataFrame.from_dict([mots_themes_LSA[theme][i][2] for i in range(len(mots_themes_LSA[theme]))]).T
imp.columns=["topic_"+str(i)+"_importance" for i in range(imp.shape[1])]
imp=imp.reset_index()
imp=imp.melt(id_vars=['index'],var_name='topic_imp',value_name='importance')
base_LDA=pd.concat([df,imp],axis=1)
base_LDA=base_LDA[['topic','mots','importance']]
base_LDA['methode']='LSA'
base_LDA['theme']=theme
base_LDA
# pd.DataFrame.from_dict(mots_themes_LDA['AGRICULTURE'])
# pd.DataFrame.from_dict(mots_themes_LDA).T
# %%
def func_base(mots_themes,methode):
    baselist=[]

    if methode=="WKmeans":
        for t in mots_themes.keys():
            df=pd.DataFrame.from_dict(mots_themes_WKMeans[t]).T
            df.columns=['mots','proximite']
            df['theme']=t
            baselist.append(df)
        df=pd.concat([baselist[i] for i in range(len(baselist))])
        df['methode']=methode

    else:
        for theme in mots_themes.keys():
            df=pd.DataFrame.from_dict([mots_themes[theme][i][1] for i in range(len(mots_themes[theme]))]).T
            df.columns=["topic_"+str(i) for i in range(df.shape[1])]
            df=df.reset_index()
            df=df.melt(id_vars=['index'],var_name='topic',value_name='mots')

            imp=pd.DataFrame.from_dict([mots_themes[theme][i][2] for i in range(len(mots_themes[theme]))]).T
            imp.columns=["topic_"+str(i)+"_importance" for i in range(imp.shape[1])]
            imp=imp.reset_index()
            imp=imp.melt(id_vars=['index'],var_name='topic_imp',value_name='importance')

            base_LDA=pd.concat([df,imp],axis=1)
            base_LDA=base_LDA[['topic','mots','importance']]
            base_LDA['methode']=methode
            base_LDA['theme']=theme
            baselist.append(base_LDA)
        df=pd.concat([baselist[i] for i in range(len(baselist))])
    return df
#%%
base_LDA=func_base(mots_themes_LDA,'LDA')
base_LSA=func_base(mots_themes_LSA,'LSA')
base_WK=func_base(mots_themes_WKMeans,'WKmeans')
base_finale=pd.concat([base_LDA,base_LSA,base_TFIDF,base_WK])
base_finale

# %%
