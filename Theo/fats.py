from typing import Text
import numpy as np
import os
from torch._C import Value  
from unidecode import unidecode
import re
from joblib import Parallel,delayed
from functools import partial
import functools
import operator
from tqdm import tqdm
import torch
import pickle
import warnings
from time import time
from transformers import CamembertModel,BertModel,RobertaModel,CamembertTokenizer,CamembertConfig
import networkx as nx
import psutil 
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import sys
import gc

class Word_Cleaning():
      '''
      Classe permettant le nettoyage du texte.
      @n_jobs : nombre de cpu, attention, plutôt ne pas en mettre beaucoup (2 par exemple) à cause de la mémoire.
      @sentence : dummy pour indiquer si on coupe les paragraphes, fixé à True, doit l'être sir vous ne mettez qu'un paragraphe à la fois.
      @threshold : dummy pour indiquer l'activation de la sélection des mots suffisamment grand, fixé à True.
      @seuil : seuil pour la taille des mots, fixé à 2.
      @lemma : activation de la lemmatisation, fixé à False
      @seuil_carac : seuil pour le nombre de caractères dans la phrase, fixé à 3.
      '''
      def __init__(self,n_jobs,sentence=False,threshold=False,seuil=None,lemma=False,seuil_carac=None):
        super(Word_Cleaning,self).__init__
        self.cpu=n_jobs
        self.seuil=seuil
        self.lemma=lemma
        self.threshold=threshold
        self.seuil_carac=seuil_carac
        self.sentence=sentence
    
      @staticmethod
      def remove_empty(text):
        #trace_empty=[]
        if type(text[0])==str:
            #for i in range(len(text)):
             #   if text[i]=='':
              #      trace_empty.append(i)
            while '' in text:
                  text.remove('')
            while ' ' in text:
                  text.remove(' ')
        elif type(text[0])==list:
            #for i in range(len(text)):
             #   if text[i]==[]:
              #      trace_empty.append(i)
            while [] in text:
                text.remove([])
            while [''] in text:
                text.remove([''])
            while [' '] in text:
                text.remove([' '])
        return text#,trace_empty
    
      def make_sentence(self,phrases):
        phrases_2=phrases.split('.')
        if self.seuil_carac!=None:
            phrases_2=[i for i in phrases_2 if len(i)>=self.seuil_carac]
        return phrases_2

      @staticmethod
      def represent_word(word):
        text = unidecode(word) #take a unicode object and returns a string
        #remove accents
        text=text.lower()
        text = re.sub(r'[^A-Za-z]',' ',str(text)) 
        #remove any character that IS NOT a-z OR A-Z
        return text
    
      def make_word(self,phrase):
        phrase_2=[self.represent_word(t) for t in phrase]
        return phrase_2

      def word_threshold(self,text):
         text = ' '.join([i for i in text.split() if len(i)>self.seuil])
         return text
      
      def sentence_threshold(self,phrases):
         phrases_2=[self.word_threshold(i) for i in phrases]
         return phrases_2

      @staticmethod
      def make_lemma(phrases):
         '''
         @docs : liste de strings
         '''
         try:
            import spacy
         except:
            raise ValueError("Vous n'avez pas installé Spacy.")
         try:
            nlp = spacy.load('fr_core_news_sm',disable=["parser","ner"])
         except:
            raise ValueError("Etes-vous certain d'avoir installé le modèle français de Spacy ?\nSinon, rendez-vous sur le site du package.")
         doc_2=[]
         for w in nlp(phrases):
               doc_2.append(w.lemma_)
         doc_2=' '.join([i for i in doc_2])
         return doc_2
      
      def lemma_docs(self,phrases):
         phrases=[self.make_lemma(i) for i in phrases]
         return phrases

      def make_summary(self,phrases):
         array=Parallel(self.cpu)(delayed(self.represent_word)(i) for i in phrases)
         if self.threshold:
            array=Parallel(self.cpu)(delayed(self.word_threshold)(i) for i in array)
         if self.lemma:
            array=Parallel(self.cpu)(delayed(self.make_lemma)(i) for i in array)
         return array

      def make_documents(self,array):
        '''
        @array : liste dont les éléments sont des phrases séparables par des points.
        '''
        from joblib import Parallel,delayed
        from functools import partial
        if self.sentence:
            array=Parallel(self.cpu)(delayed(self.make_sentence)(i) for i in array)


        array=Parallel(self.cpu)(delayed(self.make_word)(i) for i in array)
        
        if self.threshold:
            # threshold=partial(self.word_threshold,a=self.seuil)
            array=Parallel(self.cpu)(delayed(self.sentence_threshold)(i) for i in array)
        
        if self.lemma:
            array=Parallel(self.cpu)(delayed(self.lemma_docs)(i) for i in array)
        
        return array

      

def encod_articles(article,output,tokenizer,dim=512):
   #tokenizer=CamembertTokenizer(tokenizer)
   encod_article=[]
   encod_mask=[]
   encod_phrase=[]

   segs=[]
   encod_segs=[]

   clss_=[0]
   encod_clss=[]

   out=[]
   output_=[]
   try:
      if len(article)==len(output):


         # On prend chaque phrase dans l'article considéré
         for phrase in article:
               #On encode la phrase en dimension libre (nbr de tokens)
               encod=tokenizer(phrase)
               encod=encod['input_ids'] #On prend le vecteur des ids

               #Tant qu'on peut additionner les phrases sans dépasser 512 on fait ça :
               if (len(encod_phrase)+len(encod))<dim:        
                  encod_phrase=encod_phrase+encod #On ajoute la nouvelle phrase aux précédentes
                  #Pour avoir les phrases les unes après les autres séparées par les bons tokens
                  #On crée le vecteur de segments
                  if (article.index(phrase)%2==0): #Si la phrase est d'index paire
                     seg=list(np.repeat(0,len(encod))) #On lui associe nombre de tokens fois des zéros
                  else:
                     seg=list(np.repeat(1,len(encod))) #Sinon des 1 
                  segs=segs+seg #On ajoute pour que le vecteur de segment suive le vecteur des tokens
                  clss=len(encod) 
                  if article.index(phrase)!=(len(article)-1):
                     clss_=clss_+[clss_[-1]+clss] #Via ce vecteur, on veut garder la trace des premiers tokens de chaque phrase
                  #Du coup on prend le token 0, puis le premier token de chaque phrase, donc pour cela
                  #on ajoute la longueur des nouvelles phrases (en tokens)
                  out=out+[output[article.index(phrase)]]

               else: #Si la dimension dépasse 512, on s'arrête là pour le moment
                  index=dim-len(encod_phrase) #On prend la dim qui sépare de 512
                  
                  segs=segs+list(np.repeat(abs(segs[-1]-1),index)) #On rajoute les segments manquants pour avoir 512 du chiffre opposé du dernier 
                  encod_segs.append(segs) #On stock le segment des phrases considérées
                  segs=list(np.repeat(0,len(encod)))
                  #Pour l'attention_mask on met des 1 pour le nombre de vrais tokens, 0 sinon 
                  attention_mask=list(np.repeat(1,len(encod_phrase)))+list(np.repeat(0,index))
                  encod_mask.append(attention_mask) #Idem on stock
                  #On rajoute des 1 sur les places manquantes pour avoir dim=512
                  #1 étant le token de remplissage associé à rien, qui va disparaitre via l'attention_mask de toute façon
                  encod_phrase=encod_phrase+list(np.repeat(1,index))
                  encod_article.append(encod_phrase)
                  encod_phrase=encod #On a stocké le vecteur qui allait être trop grand (>512), donc maintenant
                  #On peut repartir avec la nouvelle phrase (encod donc)

                  encod_clss.append(clss_)
                  clss_=[0] #On réinitialise 

                  output_.append(out)
                  out=[output[article.index(phrase)]]

         #Ensuite une fois qu'on a terminé de passer en revue toutes les phrases de l'article
         #on va stocker les derniers vecteurs, donc
         #le seul si on a jamais dépassé dim 512
         # le dernier si on a déjà dû en stocker quelqu'uns
         index=dim-len(encod_phrase)
         # try:
         segs=segs+list(np.repeat(abs(segs[-1]-1),index))
         encod_segs.append(segs)
         # except:
         #     segs=segs+list(np.repeat(0,index))
         #     encod_segs.append(segs)
               
         attention_mask=list(np.repeat(1,len(encod_phrase)))+list(np.repeat(0,index))
         encod_mask.append(attention_mask)

         encod_phrase=encod_phrase+list(np.repeat(1,index))
         encod_article.append(encod_phrase)
         
         encod_clss.append(clss_)
         output_.append(out)

         return encod_article,encod_mask,encod_segs,encod_clss,output_#,len(encod_article)
      else:
         raise ValueError("Attention ! La dimension de l'article et de l'ouput sont différentes !")
   except:
      return encod_article,encod_mask,encod_segs,encod_clss,output_#,len(encod_article)


def encod_articles_inference(article,tokenizer,dim=512):
   #tokenizer=CamembertTokenizer(tokenizer)
   encod_article=[]
   encod_mask=[]
   encod_phrase=[]

   segs=[]
   encod_segs=[]

   #clss_=[0]
   #encod_clss=[]

   try:
        # On prend chaque phrase dans l'article considéré
         for phrase in article:
               #On encode la phrase en dimension libre (nbr de tokens)
               encod=tokenizer(phrase)
               encod=encod['input_ids'] #On prend le vecteur des ids

               #Tant qu'on peut additionner les phrases sans dépasser 512 on fait ça :
               if (len(encod_phrase)+len(encod))<dim:        
                  encod_phrase=encod_phrase+encod #On ajoute la nouvelle phrase aux précédentes
                  #Pour avoir les phrases les unes après les autres séparées par les bons tokens
                  #On crée le vecteur de segments
                  if (article.index(phrase)%2==0): #Si la phrase est d'index paire
                     seg=list(np.repeat(0,len(encod))) #On lui associe nombre de tokens fois des zéros
                  else:
                     seg=list(np.repeat(1,len(encod))) #Sinon des 1 
                  segs=segs+seg #On ajoute pour que le vecteur de segment suive le vecteur des tokens
                  #clss=len(encod) 
                  #if article.index(phrase)!=(len(article)-1):
                   #  clss_=clss_+[clss_[-1]+clss] #Via ce vecteur, on veut garder la trace des premiers tokens de chaque phrase
                  #Du coup on prend le token 0, puis le premier token de chaque phrase, donc pour cela
                  #on ajoute la longueur des nouvelles phrases (en tokens)

               else: #Si la dimension dépasse 512, on s'arrête là pour le moment
                  index=dim-len(encod_phrase) #On prend la dim qui sépare de 512
                  
                  segs=segs+list(np.repeat(abs(segs[-1]-1),index)) #On rajoute les segments manquants pour avoir 512 du chiffre opposé du dernier 
                  encod_segs.append(segs) #On stock le segment des phrases considérées
                  segs=list(np.repeat(0,len(encod)))
                  #Pour l'attention_mask on met des 1 pour le nombre de vrais tokens, 0 sinon 
                  attention_mask=list(np.repeat(1,len(encod_phrase)))+list(np.repeat(0,index))
                  encod_mask.append(attention_mask) #Idem on stock
                  #On rajoute des 1 sur les places manquantes pour avoir dim=512
                  #1 étant le token de remplissage associé à rien, qui va disparaitre via l'attention_mask de toute façon
                  encod_phrase=encod_phrase+list(np.repeat(1,index))
                  encod_article.append(encod_phrase)
                  encod_phrase=encod #On a stocké le vecteur qui allait être trop grand (>512), donc maintenant
                  #On peut repartir avec la nouvelle phrase (encod donc)

                  #encod_clss.append(clss_)
                  #clss_=[0] #On réinitialise 

         #Ensuite une fois qu'on a terminé de passer en revue toutes les phrases de l'article
         #on va stocker les derniers vecteurs, donc
         #le seul si on a jamais dépassé dim 512
         # le dernier si on a déjà dû en stocker quelqu'uns
         index=dim-len(encod_phrase)
         #print(index)
         # try:
         segs=segs+list(np.repeat(abs(segs[-1]-1),index))
         encod_segs.append(segs)
         # except:
         #     segs=segs+list(np.repeat(0,index))
         #     encod_segs.append(segs)
               
         attention_mask=list(np.repeat(1,len(encod_phrase)))+list(np.repeat(0,index))
         encod_mask.append(attention_mask)

         encod_phrase=encod_phrase+list(np.repeat(1,index))
         encod_article.append(encod_phrase)
         
         #encod_clss.append(clss_)

         return encod_article,encod_mask,encod_segs#,encod_clss#,len(encod_article)
   except:
      print("Unexpected error:", sys.exc_info())
      return encod_article,encod_mask,encod_segs#,encod_clss#,len(encod_article)



class Make_Extractive():
    def __init__(self,cpu=None,path=os.getcwd(),fenetre=None,minimum=None,d=None,epochs=None,cosim=torch.nn.CosineSimilarity(-1)):
        super(Make_Extractive,self).__init__
        self.fenetre=fenetre
        self.minimum=minimum
        self.dim=d
        self.epochs=epochs
        self.cosim=cosim
        self.path=path
        self.document_encoding=encod_articles
        self.encoding_inference=encod_articles_inference
        self.cpu=cpu
    
    @staticmethod
    def make_w2v_sentences(documents):
        '''
        @documents : une liste de documents, qui sont une liste de phrases
        '''
        sentences=[]
        #for i in documents:
        for z in documents:
            if len(z.split())>0:
               sentences.append(z.split())
        return sentences
    
    @staticmethod
    def make_splitting(sequence,vocab=None):
       if vocab==None:
         sequence=[s.split() for s in sequence]
       else:
         sequence=[s.split() for s in sequence]
         sequence=[[i for i in s if i in vocab] for s in sequence]
       return sequence
    
    def make_W2V(self,docs):
      sentence=Parallel(n_jobs=self.cpu)(delayed(self.make_w2v_sentences)(d) for d in docs)
      import gensim
      try:
         W2V=gensim.models.Word2Vec(size=self.dim,window=self.fenetre,min_count=self.minimum)
      except:
         W2V=gensim.models.Word2Vec(vector_size=self.dim,window=self.fenetre,min_count=self.minimum)
      W2V.build_vocab(sentence)
      print("Démarrage de l'entraînement du modèle Word2Vec.")
      start=time()
      W2V.train(sentence,total_examples=W2V.corpus_count,epochs=self.epochs)
      end=time()
      print("Le modèle W2V est désormais entraîné et cela a pris :",round((end-start)/60,2),"minutes.")
      return W2V


    def remove_empty(text):
       while '' in text:
          text.remove('')
       return text

    def make_output(self,text,summary,W2V=None,verbose=0):
       assert len(text)==len(summary)
       if W2V==None:
          return_W2V=True
          W2V=self.make_W2V(text)
          try:
            vocab=list(W2V.wv.vocab.keys())
          except:
            vocab=list(set(W2V.wv.key_to_index))
          start=time()
          text=Parallel(n_jobs=self.cpu)(delayed(self.make_splitting)(s) for s in text)
          #text=[[i.split() for i in s] for s in docs]

          summary=self.make_splitting(summary,vocab)
          #  summary=[[i for i in s.split() if i in vocab] for s in summary]
          end=time()
          print("Le découpage des phrases a pris:",round((end-start)/60,2),"minutes.")
       else:
          return_W2V=False

       dimension=[len(text),len(text[0]),len(text[0][0]),len(text[0][0][0]),len(text[0][0][0][0])]
       print(dimension)
       score=[]
       erreur=[]
       import gensim
       if dimension[3]>1:
         if gensim.__version__<'4.0.0':
            for sent in tqdm(text):
               score_=[]
               for s in sent:
                  try:
                     score_.append(torch.stack(
                           [self.cosim(torch.as_tensor(W2V[i]),torch.as_tensor(W2V[summary[text.index(sent)]]))
                           for i in s]).mean())
                  except:
                     erreur.append([text.index(sent),sent.index(s)])
                     if verbose==1:
                        print("Attention, l'élément",erreur[-1],"n'a pas pu être encodé.")
                     continue
               try:
                  score.append(torch.stack(score_))
               except:
                  score.append(torch.Tensor())

         else:
            for sent in tqdm(text):
               score_=[]
               for s in sent:
                  try:
                     score_.append(torch.stack(
                           [self.cosim(torch.as_tensor(W2V.wv[i]),torch.as_tensor(W2V.wv[summary[text.index(sent)]]))
                           for i in s]).mean())
                  except:
                     erreur.append([text.index(sent),sent.index(s)])
                     # print("Attention, l'élément",erreur[-1],"n'a pas pu être encodé.")
                     continue
               try:
                  score.append(torch.stack(score_))
               except:
                  score.append(torch.Tensor())
       else:
         if gensim.__version__<'4.0.0':
            for sent in tqdm(text):
               score_=[]
               for s in sent:
                  try:
                     score_.append(
                           self.cosim(torch.as_tensor(W2V[s]),
                           torch.as_tensor(W2V[summary[text.index(sent)]])).mean())
                  except:
                     erreur.append([text.index(sent),sent.index(s)])
                     if verbose==1:
                        print("Attention, l'élément",erreur[-1],"n'a pas pu être encodé.")
                     continue
               try:
                  score.append(torch.stack(score_))
               except:
                  score.append(torch.Tensor())

         else:
            for sent in tqdm(text):
               score_=[]
               for s in sent:
                  try:
                     score_.append(
                           self.cosim(torch.as_tensor(W2V.wv[s]),
                              torch.as_tensor(W2V.wv[summary[text.index(sent)]])).mean())
                  except:
                     erreur.append([text.index(sent),sent.index(s)])
                     # print("Attention, l'élément",erreur[-1],"n'a pas pu être encodé.")
                     continue
               try:
                  score.append(torch.stack(score_))
               except:
                  score.append(torch.Tensor())
       if return_W2V:
         return score,text,summary,erreur,W2V
       else:
         return score,text,summary,erreur

    @staticmethod
    def make_input_tokenizer(text,path,name):
       text=[' '.join([i for i in t]) for t in text]
       if '/' in path:
          path=path+'/'+name+'.txt'
       elif '\\' in path:
          path=path+'\\'+name+'.txt'

       b=text[0]+'\n'
       for i in tqdm(range(len(text)-1)):
         b+=text[i+1]+'\n'
       with open(path,'w') as f:
         f.write(b)
       print("\nL'input du tokenizer a été sauvegardé ici:",path)
       return path

    
    def make_tokenizer(self,text,voc_size,prefix,mtype='bpe',name="tokenizer_input"):
         '''
         @path : chemin vers le fichier txt, de format un doc par ligne
         @voc_size : la taille souhaitée sur vocabulaire
         @prefix : le nom que l'on veut donner au modèle
         @mtype : le type de modèle, par exemple 'bpe'
         '''
         os.chdir(self.path)
         path=self.make_input_tokenizer(text,self.path,name)

         if '/' in path:
            chemin='/'.join([i for i in path.split('/')[:-1]])
            chemin=chemin+'/'
         elif '\\' in path:
            chemin='\\'.join([i for i in path.split('\\')[:-1]])
            chemin=chemin+"\\"
         else:
            raise ValueError('La fonction ne parvient pas à trouver le chemin pour enregistrer le tokenizer, vérifier le chemin fourni, la variable path')


         import sentencepiece as spm 
         #L'input doit être un fichier .txt
         FUES=spm.SentencePieceTrainer.train(
            input=path, #chemin vers le fichier txt, un doc par ligne
            vocab_size=voc_size, #taille du vocab, peut être augmenté, ne doit pas être trop grand par rapport aux mots des documents
            model_prefix=prefix, #nom du modèle, French Unsupervised Exctractive Summarizer
            model_type=mtype)
         
         from transformers import CamembertTokenizer
         tokenizer=CamembertTokenizer(chemin+prefix+'.model')
         return tokenizer
    
    @staticmethod
    def make_mask_cls(train_clss,dim=512):
         mask_cls_1=torch.zeros(dim)
         for i in train_clss:
            mask_cls_1[i]=1
         return mask_cls_1

    @staticmethod
    def make_tensor_clss(clss,dim=512):
         index=[dim-len(i) for i in clss]
         vect=[clss[i]+list(np.zeros(index[i])) for i in range(len(index))]
         clss=torch.as_tensor(vect)
         return clss

    @staticmethod
    def transform_target(mask_cls,output,dim=512,k=3):
      _,ind_mask=mask_cls.topk(k=int(mask_cls.sum()))
      out=torch.zeros(dim)
      values,ind=torch.topk(output,k)
      ind=ind[[values!=0]]
      ind_output=ind_mask.sort()[0][ind]
      out[ind_output]=1
      return out

    def make_encoding(self,text,output=None,tokenizer=None,voc_size=None,prefix="essai",name="Tokenizer_input",split=0.8,dim=512,training=True):
       if training:
         if tokenizer==None:
            assert voc_size!=None
            tokenizer=self.make_tokenizer(text,voc_size,prefix=prefix,name=name)
         tokenizer=CamembertTokenizer(tokenizer)
         doc_encod=partial(self.document_encoding,tokenizer=tokenizer,dim=dim)   
         encoding=Parallel(n_jobs=self.cpu)(delayed(doc_encod)(i,j) for i,j in zip(text,output))
         from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

         split_border=int(len(encoding)*split)
         train=encoding[:split_border]

         train_input_ids=[train[i][0][k] for i in range(len(train)) for k in range(len(train[i][0]))]
         train_mask=[train[i][1][k] for i in range(len(train)) for k in range(len(train[i][1]))]
         train_segs=[train[i][2][k] for i in range(len(train)) for k in range(len(train[i][2]))]
         #train_clss=[train[i][3][k] for i in range(len(train)) for k in range(len(train[i][3]))]
         #clss_index_train=[len(i) for i in train_clss]
         #train_mask_cls=torch.as_tensor([list(self.make_mask_cls(t)) for t in train_clss])
         #train_clss=self.make_tensor_clss(train_clss)
         train_output=self.make_tensor_clss([train[i][-1][k] for i in range(len(train)) for k in range(len(train[i][4]))])
         trace_train=[len(train[i][0]) for i in range(len(train))]

         dico_train={
            'input':train_input_ids,
            'mask':train_mask,
            'segs':train_segs,
            #'clss':train_clss,
            #'clss_index':clss_index_train,
            'output':train_output,
            #'mask_cls':train_mask_cls,
            'trace':trace_train
         }
         dico_train['mask_cls']=(torch.Tensor(dico_train['input'])==torch.tensor(5)).int()

         #  pickle.dump(dico_train,open(self.path+'/dico_train.pickle','wb'))
         
         if split<1:
            test=encoding[split_border:]
            test_input_ids=[test[i][0][k] for i in range(len(test)) for k in range(len(test[i][0]))]
            test_mask=[test[i][1][k] for i in range(len(test)) for k in range(len(test[i][1]))]
            test_segs=[test[i][2][k] for i in range(len(test)) for k in range(len(test[i][2]))]
            #test_clss=[test[i][3][k] for i in range(len(test)) for k in range(len(test[i][3]))]
            #clss_index_test=[len(i) for i in test_clss]
            #test_mask_cls=torch.as_tensor([list(self.make_mask_cls(t)) for t in test_clss])
            #test_clss=self.make_tensor_clss(test_clss)
            test_output=self.make_tensor_clss([test[i][-1][k] for i in range(len(test)) for k in range(len(test[i][-1]))])
            trace_test=[len(test[i][0]) for i in range(len(test))]

            dico_test={
               'input':test_input_ids,
               'mask':test_mask,
               'segs':test_segs,
               #'clss':test_clss,
               #'clss_index':clss_index_test,
               'output':test_output,
            #    'mask_cls':test_mask_cls,
               'trace' : trace_test
            }
            # pickle.dump(dico_test,open(self.path+'/dico_test.pickle','wb'))
            dico_test['mask_cls']=(torch.Tensor(dico_test['input'])==torch.tensor(5)).int()
            return dico_train,dico_test

         return dico_train

       else:
            tokenizer=CamembertTokenizer(tokenizer)
            doc_encod=partial(self.encoding_inference,tokenizer=tokenizer,dim=dim)
            
            if type(text[0])==list:
                train=Parallel(n_jobs=self.cpu)(delayed(doc_encod)(i) for i in text)
                train_input_ids=[train[i][0][k] for i in range(len(train)) for k in range(len(train[i][0]))]
                train_mask=[train[i][1][k] for i in range(len(train)) for k in range(len(train[i][1]))]
                train_segs=[train[i][2][k] for i in range(len(train)) for k in range(len(train[i][2]))]
                #train_clss=[train[i][3][k] for i in range(len(train)) for k in range(len(train[i][3]))]
                #clss_index_train=[len(i) for i in train_clss]
                #train_mask_cls=torch.as_tensor([list(self.make_mask_cls(t)) for t in train_clss])
                #train_clss=self.make_tensor_clss(train_clss)
                trace_train=[len(train[i][0]) for i in range(len(train))]
                
            elif type(text[0])==str:
                train=doc_encod(text)
                train_input_ids=[train[0][k] for k in range(len(train[0]))]
                train_mask=[train[1][i] for i in range(len(train[1]))]
                train_segs=[train[2][i] for i in range(len(train[2]))]
                #train_clss=[train[3][i] for i in range(len(train[3]))]
                #clss_index_train=[len(i) for i in train_clss]
                #train_mask_cls=torch.as_tensor([list(self.make_mask_cls(t)) for t in train_clss])
                #train_clss=self.make_tensor_clss(train_clss)
                trace_train=[len(train[i][0]) for i in range(len(train))]
                
            dico_train={
               'input':train_input_ids,
               'mask':train_mask,
               'segs':train_segs,
               #'clss':train_clss,
               #'clss_index':clss_index_train,
               #'mask_cls':train_mask_cls,
               'trace':trace_train
            }
            dico_train['mask_cls']=(torch.Tensor(dico_train['input'])==torch.tensor(5)).int()
            return dico_train


class Make_Embedding():
    def __init__(self,tok=None,cpu=psutil.cpu_count()) -> None:
        super(Make_Embedding,self).__init__
        self.tokenizer=tok
        self.cpu=cpu
        
    @staticmethod
    def complete_tokens(input_id,att_mask,dim=512):
        index=dim-len(input_id)
        input_id=input_id+list(np.repeat(1,index))
        att_mask=att_mask+list(np.repeat(0,index))
        return input_id,att_mask
        
    def make_token(self,sequence,cpu):
        tokens=self.tokenizer(sequence)
        input_ids=tokens['input_ids']
        att_mask=tokens['attention_mask']
        dico=Parallel(cpu)(delayed(self.complete_tokens)(i,j) for i,j in zip(input_ids,att_mask))
        input_ids=[dico[i][0] for i in range(len(dico))]
        att_mask=[dico[i][1] for i in range(len(dico))]
        return input_ids,att_mask

    def make_tokens(self,sequence,cpu):
        mt=partial(self.make_token,cpu=cpu)
        if type(sequence[0])==list:
           tokens=Parallel(n_jobs=self.cpu)(delayed(mt)(z) for z in sequence)
      #   elif type(sequence[0])==str:
      #      tokens=mt(sequence)
        dico={}
        dico['input_ids']=[tokens[i][0] for i in range(len(tokens))]
        dico['attention_mask']=[tokens[i][1] for i in range(len(tokens))]
        return dico

    @staticmethod
    def emb_phrase(input_id,att_mask,cam):
      embeddings=[]
      for i,a in zip(input_id,att_mask):
         try:
            embedding=cam(torch.tensor(i).squeeze(1),torch.tensor(a).squeeze(1))
            embeddings.append(embedding.last_hidden_state.mean(dim=1))
         except:
            embedding=cam(torch.tensor(i).squeeze(0),torch.tensor(a).squeeze(0))
            embeddings.append(embedding.last_hidden_state.mean(dim=1))
            #embeddings.append(embedding[0].mean(dim=0).squeeze(0))
      return embeddings
    
    def emb_phrases(self,input_ids,att_masks,cam):
        embedding=[]
        for input_id,att_mask in zip(input_ids,att_masks):
            embeddings=self.emb_phrase(input_id,att_mask,cam)
            embedding.append(embeddings)
        return embedding

class TextRank():
   def __init__(self,tok_path=None,cpu=psutil.cpu_count()):
      '''
      @tok_path: chemin vers le tokenizer à utiliser
      @cpu: nombre de cpu à utiliser
      '''
      super(TextRank,self).__init__
      if tok_path!=None:
         self.bert_embedding=Make_Embedding(tok=CamembertTokenizer(tok_path),cpu=cpu)
      self.camem=CamembertModel(CamembertConfig())
      self.cpu=cpu
        
   def make_embedding_bert(self,articles,camem,seuil=70):
      if type(articles[0])==str:
         dico={}
        #  start=time()
         input_ids,att_mask=self.bert_embedding.make_token(articles,self.cpu)
        #  end_1=time()
        #  print("Les tokens ont pris :",round((end_1-start)/60,2),"minutes.")
         if len(input_ids)>seuil: #Au-delà de 70 phrases, camembert plante. On vérifie que le paragraphe en contient moins puis
            #on va découper le paragraphe pour que chaque bout fasse moins de 70
            # d'abord il convient de trouver le chiffre tq len(input_ids)/chiffre<70
            for i in range(2,100):
                if (len(input_ids)/i)<seuil:
                    h=i
                    break
                else:
                    continue
            #une fois qu'on a ce chiffre h on y va :
            embeddings=[]
            for i in range(h):
                x_1=int(len(input_ids)*(i/h))
                x_2=int(len(input_ids)*((i+1)/h))
                embeddings.append(camem(torch.tensor(input_ids[x_1:x_2]),
                   torch.tensor(att_mask[x_1:x_2])).last_hidden_state.detach())
            embeddings=torch.cat(embeddings).mean(dim=1)
         
         else:
             embeddings=camem(torch.tensor(input_ids),
                       torch.tensor(att_mask))
             embeddings=embeddings.last_hidden_state.detach().mean(dim=1)
         
        #  end_2=time()
        #  print("L'embedding a pris :",round((end_2-end_1)/60,2),"minutes.")
         return embeddings,dico
      else:
        #  start=time()
         dico=self.bert_embedding.make_tokens(articles,self.cpu)
         input_ids=dico['input_ids']
         att_mask=dico['attention_mask']
        #  end_1=time()
        #  print("Les tokens ont pris :",round((end_1-start)/60,2),"minutes.")
         
         embeddings=self.bert_embedding.emb_phrase(input_ids,att_mask,camem)
        #  end_2=time()
        #  print("L'embedding a pris :",round((end_2-end_1)/60,2),"minutes.")
         return embeddings,dico
   
   @staticmethod
   def mat_sim(emb_2,cos_sim=torch.nn.CosineSimilarity(dim=0)):
      ouais=[[cos_sim(emb,y) for y in emb_2] for emb in emb_2]
      return torch.as_tensor(ouais)


   @staticmethod
   def get_emb_sentence(art,modele,verbose,di=0):
      vocab=list(set(modele.wv.vocab))
      art_2=art.copy()
      art_2.reverse()
      for w in art_2:
         if w in vocab:
            continue
         else:
            if w[:-1] in vocab:
               art[art.index(w)]=w[:-1]
            else:
               art.remove(w)
               if verbose==1:
                  print("Le mot :",w,"ne semble pas faire partie du vocabulaire, nous l'enlevons donc de la séquence traitée.")
         
      word=[modele[w] for w in art]
      word=torch.as_tensor(word).mean(dim=di)
      return word

   def get_matrix_section(self,art,W2V,verbose):
      mat=[self.get_emb_sentence(art[i],W2V,verbose) for i in range(len(art))]
      mat=[torch.as_tensor(np.nan_to_num(i)) if np.isnan(i).sum()>0 else i for i in mat]
      return mat

   def make_embedding_W2V(self,article,W2V,verbose):
      article_=[article[i].split() for i in range(len(article))]
      mat=self.get_matrix_section(article_,W2V,verbose)
      return mat
   @staticmethod
   def scores(matrice_similarite,nx=nx,k=3,weights=True,alpha=0.1,frac=0.25):
        graph=nx.from_numpy_array(np.array(matrice_similarite))
        scores=nx.pagerank_numpy(graph)
        if weights:
            l=matrice_similarite.size()[0]
            w=torch.ones(l)
            x1=int(frac*l)
            x2=int((1-frac)*l)
            w[:x1]=torch.mul(w[:x1],(1+alpha))
            w[x2:]=torch.mul(w[x2:],(1+alpha))

            scores=torch.tensor(list(scores.values()))
            scores_new=torch.mul(scores,w)
            rank=scores_new.topk(k)[1]
            return rank
        else:
            rank=sorted(scores.items(),key=lambda v:(v[1],v[0]),reverse=True)[:k]
            rank=[s[0] for s in rank]
            return rank
   
   def make_resume(self,article,type,modele,k=3,verbose=1,get_score=False,get_score_only=False,s=70,weights=True,alpha=0.1,frac=0.25):
      if type=='bert':
         b,d=self.make_embedding_bert(article,modele,seuil=s)
         mb=self.mat_sim(b)#[self.mat_sim(h) for h in b]
         sb=self.scores(mb,k=k,weights=weights,alpha=alpha,frac=frac)#[self.scores(m,k=k) for m in mb]
         if get_score:
            resume=[article[i] for i in sb]#[[article[k][i] for i in sb[k]] for k in range(len(sb))]
            if len(resume)==1:
               return resume[0],sb[0]
            else:
               return resume, sb
        
         elif get_score_only:
            return sb
        
         else:
            resume=[article[i] for i in sb]#[[article[k][i] for i in sb[k]] for k in range(len(sb))]
            if len(resume)==1:
               return resume[0]
            else:
               return resume
      
      elif type=='word2vec':
         assert modele!=None
         w=self.make_embedding_W2V(article,modele,verbose)
         mw=self.mat_sim(w)#[self.mat_sim(k) for k in w]
         sw=self.scores(mw,k=k,weights=weights,alpha=alpha)#[self.scores(m,k=k) for m in mw]
         resume=[article[i] for i in sw]#[[article[k][i] for i in sw[k]] for k in range(len(sw))]
         if get_score:
            if len(resume)==1:
               return resume[0],sw[0]
            else:
               return resume, sw
         elif get_score_only:
            return sw
         else:
            if len(resume)==1:
               return resume[0]
            else:
               return resume
      else:
         raise ValueError("Attention, vous devez spécifier le type d'embedding que vous voulez utiliser, soit 'bert' soit 'word2vec'.")

class BERTScore():
      def __init__(self,tok,camem=CamembertModel.from_pretrained("camembert-base"),#CamembertModel(CamembertConfig()),
      cosim=torch.nn.CosineSimilarity(dim=-1),cpu=1) -> None:
         super(BERTScore,self).__init__
         self.make_embedding=TextRank(tok,cpu=cpu).make_embedding_bert
         self.camem=camem
         self.cosim=cosim

      def make_score(self,article,k=3,s=70):
        #  start=time()
         b,_=self.make_embedding(article,self.camem,seuil=s)
        #  end_1=time()
        #  print("L'embedding a pris :",round((end_1-start)/60,2))
         #b=torch.stack(b)
         VSA=b.mean(dim=0)
        #  end_2=time()
        #  print("Le calcul a pris :",round((end_1-end_2)/60,2))
         score=self.cosim(VSA,b)
        #  end_2=time()
        #  print("Le calcul de similarité a pris :",round((end_2-end_1)/60,2))
         try:
             score=score.topk(k=k)[1]
         except:
            try:
                score=score.topk(k=k-1)[1]
            except:
                score=score.topk(k=k-2)[1]
        #  end_3=time()
        #  print("La récupération des scores a pris :",round((end_3-end_2)/60,2))
         return score
      
      def make_summary(self,article,k=3):
         score=self.make_score(article)
         score=score.topk(k=k)[1]
         resume=[article[i] for i in score]
         return resume


def Random_summary(section,k=2,get_index=False,get_index_only=False):
   x1=np.random.randint(low=0,high=len(section),size=k)
   resume=[]
   for i in x1:
      resume.append(section[i])
   if get_index:
      return resume,x1
   elif get_index_only:
      return x1
   else:
      return resume

def Lead_3(sections,k=3):
   resume=sections[:k]
   return resume


class Simple_Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Simple_Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.relu=nn.LeakyReLU(negative_slope= 0.01)

    def forward(self, x):
        x.requires_grad_(True)
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.relu(h)
        return sent_scores.squeeze(-1)

class Multi_Linear_Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Multi_Linear_Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.linear2 = nn.Linear(int(hidden_size/2),int(hidden_size/6))
        self.linear3 = nn.Linear(int(hidden_size/6),1)
        self.Lrelu=nn.LeakyReLU(negative_slope= 0.01)
        self.softmax=nn.Softmax(dim=-1)


    def forward(self, x):#, mask_cls):
        x.requires_grad_(True)
        h = self.linear1(x).squeeze(-1)
        h = self.softmax(h)#self.Lrelu(h) #* mask_cls.float()
        h = self.linear2(h)
        h = self.softmax(h)#self.Lrelu(h)
        h = self.linear3(h)
        #h = self.softmax(h)#self.Lrelu(h)
        return h.squeeze(-1)
    

class SMHA_classifier(nn.Module):
    def __init__(self, size,nhead):
        super(SMHA_classifier, self).__init__()
        self.MHA = nn.MultiheadAttention(size[1], nhead)
        self.LReLu=nn.LeakyReLU(negative_slope= 0.01)
        self.sigmoid = nn.Sigmoid()
        self.LN=nn.LayerNorm(size)

    def forward(self, x):
        x.requires_grad_(True)
        h,weights = self.MHA(x,x,x)
        normalized_h=self.LN(h)
        sent_scores = self.LReLu(normalized_h) #* mask_cls.float()
        return sent_scores.mean(dim=2)
    
class SMHA_Linear_classifier(nn.Module):
    def __init__(self, size,nhead,hidden_size):
        super(SMHA_Linear_classifier, self).__init__()
        self.MHA = nn.MultiheadAttention(size[1], nhead)
        self.LReLu=nn.LeakyReLU(negative_slope= 0.01)
        self.sigmoid = nn.Sigmoid()
        self.LN=nn.LayerNorm(size)
        self.linear1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.linear2 = nn.Linear(int(hidden_size/2),int(hidden_size/6))

    def forward(self, x):
        x.requires_grad_(True)
        h,weights = self.MHA(x,x,x)
        h=self.LN(h)
        h=self.linear1(h)
        h=self.LReLu(h)
        h=self.linear2(h)
        sent_scores = self.LReLu(h) #* mask_cls.float()
        return sent_scores.mean(dim=2) 
    

class Net(nn.Module):
    def __init__(self,k1,k2,k3,s1,s2,s3):
        super().__init__()
        self.conv1 = nn.Conv1d(512, 512, kernel_size=k1,stride=s1)
        self.pool = nn.MaxPool1d(k2, s2)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=k3,stride=s3)
        self.dim=int((768-k1)/s1)+1
        self.dim=int((self.dim-(k2-1)-1)/s2+1)
        self.dim=int((self.dim-k3)/s3)+1
        self.fc1 = nn.Linear(self.dim, int(self.dim/2))
        self.fc2 = nn.Linear(int(self.dim/2), int(self.dim/8))
        self.fc3 = nn.Linear(int(self.dim/8), 1)
        self.LReLu=nn.LeakyReLU(negative_slope= 0.01)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self, x):
        x.requires_grad_(True)
        x = self.pool(self.LReLu(self.conv1(x)))
        x =self.LReLu(self.conv2(x))
        #x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.LReLu(self.fc1(x))
        x = self.LReLu(self.fc2(x))
        x = self.fc3(x)
        #x=self.softmax(x)
        return x.flatten(1)

class F1_score:
    """
    Class for f1 calculation in Pytorch.
    """

    def __init__(self):#, average: str = 'weighted'):
        """
        Init.

        Args:
            average: averaging method
        """

        #self.average = average
        #if average not in [None, 'micro', 'macro', 'weighted']:
         #   raise ValueError('Wrong value of average parameter')
    @staticmethod
    def true_positive_mean(x,y) -> torch.tensor:
        '''
        Caclul le nombre moyen de vrai positif de la prediction x par rapport aux labels y (binaires).
        '''
        tp=torch.mul(x,y).sum()
        k=y.sum()
        tpm=torch.div(tp,k)#y.shape[0])
        return tpm
    @staticmethod
    def false_positive_mean(x,y) -> torch.tensor:
        '''
        Caclul le nombre moyen de faux négatif de la prediction x par rapport aux labels y (binaires).
        '''
        device=y.device
        fp=torch.sub(x,y)
        fp=torch.max(fp,torch.tensor([0.]).to(device))
        fp=fp.sum().float()
        numneg=y.shape[0]-y.sum()
        fpm=torch.div(fp,numneg)#y.shape[0])
        if numneg==0: # Pour éviter division par 0
            if fp==0: # Si fp==0 et numneg==0, alors on est bon
                return torch.tensor(0)
            if fp!=0: # alors fp==1, et on n'est pas bon
                return torch.tensor(0)
        return fpm
    @staticmethod
    def false_negative_mean(x,y) -> torch.tensor:
        '''
        Caclul le nombre moyen de faux négatif de la prediction x par rapport aux labels y (binaires).
        '''
        fn=torch.sub(y,x)
        device=y.device
        fn=torch.max(fn,torch.tensor([0.]).to(device))
        fn=fn.sum().float()
        numpos=y.sum()
        fnm=torch.div(fn,numpos)#y.shape[0])
        return fnm
    #@staticmethod
    def precision(self,x,y) -> torch.tensor:
        device=y.device
        tp=torch.mul(x,y).sum()
        fp=torch.sub(x,y)
        fp=torch.max(fp,torch.tensor([0.]).to(device))
        fp=fp.sum().float()
        return torch.div(tp,tp+fp)
        # tp=self.true_positive_mean(x,y)
        # fp=self.false_positive_mean(x,y)
        # if (tp+fp)!=0:
        #     prec=torch.div(tp,(tp+fp))
        #     return prec
        # else:
        #     return torch.tensor(0.).to(device)

    def recall(self,x,y) -> torch.tensor:
        tp=torch.mul(x,y).sum()
        fn=torch.sub(y,x)
        device=y.device
        fn=torch.max(fn,torch.tensor([0.]).to(device))
        fn=fn.sum().float()
        #self.true_positive_mean(x,y)
        # fn=self.false_negative_mean(x,y)
        rec=torch.div(tp,(tp+fn))
        return rec

    def __call__(self,x,y) -> torch.tensor:
        device=y.device
        rec=self.recall(x,y)
        prec=self.precision(x,y)
        f1=torch.mul(rec,prec)
        f1=torch.mul(2,f1)
        f1=torch.div(f1,prec+rec)
        if (prec+rec)!=0:
            return f1#prec,rec,
        else:
            return torch.tensor(0.).to(device)#prec,rec,

        
class Weighted_Loss:
    '''
    Fonction permettant de calculer la fonction de perte Mean Absolute Error mais pondérée par des poids.
    '''
    def __init__(self,weight,loss_type='L1',binary=True):
        '''
        On initialise notre fonction de perte :
        @weight : les poids que vous voulez pour chaque classe (dim=nombre de classe)
        '''
        self.weights=weight
        self.loss_type=loss_type
        self.binary=binary
        
    def Weighted_L1(self,y_hat,y) -> torch.Tensor:
        '''
        On calcule la fonction :
        @y_hat : les prédictions du modèle
        @y : les vraies valeurs
        
        Attention, dim(y_hat)==dim(y)
        '''
        if y_hat.shape!=y.shape:
            raise ValueError("Attention, les deux inputs n'ont pas la même dimension !")
        #On met les deux tensors sur le même service (ici GPU)
        device_yhat=y_hat.device
        device_y=y.device
        if device_yhat!=device_y:
            y.to(device_yhat)
        
        w=torch.repeat_interleave(self.weights[0].clone().detach(),y.shape[1])
        w=w.repeat(y.shape[0],1)
                
        if self.binary:
            w[torch.arange(y.shape[0],dtype=torch.long).unsqueeze(1),torch.topk(y,3)[1]]=self.weights[1]
        
        else: #On surpondère les indices qui représentent les phrases, puisque c'est cela que le modèle doit prédire
            x=torch.nonzero(y!=torch.tensor(0))#.nonzero()
            x_2=torch.index_select(x,1,torch.tensor(1).to(device_yhat)).reshape(-1).to(device_yhat)
            x_1=torch.nonzero(x_2==0).to(device_yhat)#.nonzero()
            sha=torch.arange(y.shape[0],dtype=torch.long).unsqueeze(1).to(device_yhat)
            
            for k in range(len(x_1)):
                if k<(len(x_1)-1):
                    w[sha[k],x_2[x_1[k]:x_1[k+1]]]=self.weights[1]
                else:
                    w[sha[k],x_2[x_1[k]:]]=self.weights[1]
               
        sum_weights=w.sum()
        w=w.to(device_yhat)
        sum_weights=sum_weights.to(device_yhat)
        errors=torch.sub(y,y_hat)
        errors=torch.abs(errors)
        weighted_errors=torch.mul(w,errors)
        sum_weighted_errors=weighted_errors.sum()
        WMAE=torch.div(sum_weighted_errors,sum_weights)
        #WMAE.requires_grad=True
        return Variable(WMAE,requires_grad=True)#,sum_weighted_errors,sum_weights
    
    def Weighted_Sum(self,y_hat,y) -> torch.Tensor:
        '''
        Calcule la somme pondérée de la différence de la prédiction du modèle et du vecteur cible.
        '''
        if y_hat.shape!=y.shape:
            raise ValueError("Attention, les deux inputs n'ont pas la même dimension !")
        
        #On met les deux tensors sur le même service (ici GPU)
        device_yhat=y_hat.device
        device_y=y.device
        if device_yhat!=device_y:
            y.to(device_yhat)
        
        w=torch.repeat_interleave(self.weights[0].clone().detach(),y.shape[1])
        w=w.repeat(y.shape[0],1)
        
        if self.binary:
            w[torch.arange(y.shape[0],dtype=torch.long).unsqueeze(1),torch.topk(y,3)[1]]=self.weights[1]
        
        else: #On surpondère les indices qui représentent les phrases, puisque c'est cela que le modèle doit prédire
            x=torch.nonzero(y!=torch.tensor(0))#.nonzero()
            x_2=torch.index_select(x,1,torch.tensor(1).to(device_yhat)).reshape(-1).to(device_yhat)
            x_1=torch.nonzero(x_2==0).to(device_yhat)#.nonzero()
            sha=torch.arange(y.shape[0],dtype=torch.long).unsqueeze(1).to(device_yhat)
            
            for k in range(len(x_1)):
                if k<(len(x_1)-1):
                    w[sha[k],x_2[x_1[k]:x_1[k+1]]]=self.weights[1]
                else:
                    w[sha[k],x_2[x_1[k]:]]=self.weights[1]
                    
        w=w.to(device_yhat)
        y_diff=torch.abs(torch.sub(y,y_hat))
        y_diff_pond=torch.mul(y_diff,w)
        sum_y_diff_pon=torch.div(torch.sum(y_diff_pond),y_hat.shape[0])
        return Variable(sum_y_diff_pon,requires_grad=True)
    
    def __call__(self,y_hat,y) -> torch.Tensor:
        if self.loss_type=='L1':
            loss=self.Weighted_L1(y_hat,y)
            return loss
        elif self.loss_type=='sum':
            loss=self.Weighted_Sum(y_hat,y)
            return loss
        else:
            raise ValueError("Attention, veuillez bien spécifier un type de perte.\nSeules les valeurs 'L1' ou 'sum' sont acceptées.")




def training_loop_gpu(model,optimizer,data,score,loss,epochs,camem2,
   device=torch.device('cuda') if torch.cuda.is_available else torch.device('cpu'),
   suppress_after=True):
    '''
    Cette fonction a pour but d'entraîner des modèles de deep learning via torch.
    On suppose que votre modèle est déjà sur le GPU.
    @model : le modèle que l'on veut entraîner, écrit en torch.
    @optimizer : l'optimiseur que l'on a choisi pour entraîner le modèle, de la famille torch.optim.
    @data : les données, de type torch.DataLoader.
    @score : une fonction de score, par exemple F1.
    @loss : une liste de différentes fonctions de pertes. L'optimisation est faite par rapport à la première fonction de perte de la liste.
    @epochs : le nombre d'épochs.
    @camem : le modèle RoBERTa pour l'embedding.
    @device : sur quel processeur on entraîne.
    ''' 
    
    training_stats = []
    #score_stat=[]
    # Boucle d'entrainement
    model.train()
   #  model.to(device)
    model.zero_grad()
    print("Entraînement du modèle :",str(model).split('(')[0])
    len_loss=len(loss)
    start=time()

    for epoch in range(0, epochs):

        # On initialise la loss pour cette epoque
        total_train_loss = 0
        if len_loss>1:
            total_train_loss_2 = 0
            if len_loss>2:
                total_train_loss_3 = 0
        f1_score=0
        prec_score=0

        # On met le modele en mode 'training'
        # Dans ce mode certaines couches du modele agissent differement

        # Pour chaque batch
        for step, batch in enumerate(tqdm(data)):

            # On recupere les donnees du batch
            input_id = batch[0]#.to(device)
            mask = batch[1]#.to(device)
            #clss = batch[2].float().to(device)
            #mask_cls=batch[3]#.to(device)
            output=batch[4].float().to(device)

            param1=list(model.parameters())[0].clone()

            # On met le gradient a 0
            optimizer.zero_grad()#summa_parallel.zero_grad()        

            # On passe la donnee au model et on recupere la loss et le logits (sortie avant fonction d'activation)
            topvec=camem2(input_id,mask)
            topvec=topvec.last_hidden_state.to(device)
            #topvec=topvec.mul(mask_cls.unsqueeze(2)).to(device)

            sortie=model(topvec)

            #On calcule et garde le score pour information, mais le détache pour éviter de faire exploser la mémoire
            f1_score+=score(sortie,output).detach().item()
            prec_score+=score.precision(sortie,output).detach().item()

            #output2=make_output_topk(output,k=1).long().to(device)
            loss_train=loss[0](sortie,output)#.detach().item() # on commente detach sur la loss par rapport à laquelle on veut optimiser
            if len_loss>1:
                loss_train_2=loss[1](sortie,output).detach().item()
                if len_loss>2:
                    loss_train_3=loss[2](sortie,output).detach().item()

            # Backpropagtion
            loss_train.backward()
            # On actualise les paramètres grace a l'optimizer
            optimizer.step()

            # Checks if the weights did update, if not, informs at which step
            param2=list(model.parameters())[0].clone()
            check=bool(1-torch.equal(param1.data,param2.data))
            if check==False:
                print("The weights did not update at batch",step,"epoch",epoch)        

            # Keep all the predictions
            #pred.append(sortie.detach())

            # .item() donne la valeur numerique de la loss
            total_train_loss += loss_train.detach().item() 
            if len_loss>1:
                total_train_loss_2 += loss_train_2#.detach().item() 
                if len_loss>2:
                    total_train_loss_3 += loss_train_3#.detach().item() 

        # On calcule les statistiques et les pertes moyennes sur toute l'epoque
        f1_stat=f1_score/len(data)
        prec_stat=prec_score/len(data)
        avg_train_loss = total_train_loss / len(data)
        if len_loss>1:
            avg_train_loss_2 = total_train_loss_2 / len(data)   
            if len_loss>2:
                avg_train_loss_3 = total_train_loss_3 / len(data)   

                print("\nAverage training loss MSE: {0:.4f}".format(avg_train_loss),
                      "\nAverage training loss L1: {0:.4f}".format(avg_train_loss_2),
                      "\nAverage training loss sum: {0:.4f}".format(avg_train_loss_3),
                      "\nAverage f1 score: {0:.4f}".format(f1_stat),
                      "\nAverage precision score: {0:.4f}".format(prec_stat))  

                # Enregistrement des stats de l'epoque
                training_stats.append(
                    {'epoch': epoch + 1,
                    'Training Loss MSE': avg_train_loss,
                    'Training Loss L1': avg_train_loss_2,
                    'Training Loss sum': avg_train_loss_3,
                    'Training f1 score': f1_stat,
                    'Training precision score':prec_stat})
            else:
                
                print("\nAverage training loss MSE: {0:.4f}".format(avg_train_loss),
                      "\nAverage training loss L1: {0:.4f}".format(avg_train_loss_2),
                      "\nAverage f1 score: {0:.4f}".format(f1_stat),
                      "\nAverage precision score: {0:.4f}".format(prec_stat))  

                # Enregistrement des stats de l'epoque
                training_stats.append(
                    {'epoch': epoch + 1,
                    'Training Loss MSE': avg_train_loss,
                    'Training Loss L1': avg_train_loss_2,
                    'Training f1 score': f1_stat,
                    'Training precision score':prec_stat})
        else:
            
                print("\nAverage training loss MSE: {0:.4f}".format(avg_train_loss),
                      "\nAverage f1 score: {0:.4f}".format(f1_stat),
                      "\nAverage precision score: {0:.4f}".format(prec_stat))  

                # Enregistrement des stats de l'epoque
                training_stats.append(
                    {'epoch': epoch + 1,
                    'Training Loss MSE': avg_train_loss,
                    'Training f1 score': f1_stat,
                    'Training precision score':prec_stat})

    end=time()
    print("L'entraînement a duré :",round((end-start)/60,2),"minutes.")
   #  model.to('cpu')
    if suppress_after:
       del loss_train
       torch.cuda.empty_cache()

    return model, optimizer, training_stats


def correct_mask_cls(input_ids):
    vec=(torch.as_tensor(input_ids)==torch.tensor(5)).nonzero()
    mask=torch.zeros(torch.as_tensor(input_ids).size())
    mask[vec]=1
    return mask


def make_dataloader(dico_train,prefix,cpu_max=None,batch_size=64):
    '''
    A partir du dico fourni par la classe, crée un dataloader utilisable.
    '''
    train_input_ids=dico_train['input']
    train_mask=dico_train['mask']
    clss=dico_train['clss']
    train_mask_cls=dico_train['mask_cls']
    train_output=dico_train['output']
    trace=dico_train['trace']

    ouais=torch.as_tensor([(train_output[i]!=torch.tensor(0)).nonzero().size()[0] for i in range(len(train_output))])
    v=((train_mask_cls.sum(dim=1)>ouais)==True).nonzero()

    train_mask_cls_2=Parallel(cpu_max)(delayed(correct_mask_cls)(train_input_ids[i]) for i in range(len(train_input_ids)))
    train_mask_cls_2=torch.stack(train_mask_cls_2)

    v=((train_mask_cls_2.sum(dim=1)>ouais)==True).nonzero()

    np.sum([int(train_mask_cls_2[i].sum())==(train_output[i]!=torch.tensor(0)).nonzero().size(0) for i in range(len(train_output))])/len(train_output)

    out=torch.zeros(train_mask_cls_2.shape,dtype=torch.float64)

    x=(train_output!=torch.tensor(0)).nonzero()
    dim_1=torch.unique(torch.stack([x[i][0] for i in range(len(x))]))

    x_2=torch.index_select(x,1,torch.tensor(1)).reshape(-1)
    x_1=(x_2==0).nonzero()

    dim_2=[]
    for k in tqdm(range(len(x_1))):
        if k<(len(x_1)-1):
            dim_2.append(x_2[x_1[k]:x_1[k+1]])
        else:
            dim_2.append(x_2[x_1[k]:])

    for k in tqdm(range(len(dim_1))):
        out[k,(train_mask_cls_2[k]!=torch.tensor(0)).nonzero().squeeze(1)]=train_output[dim_1[k],dim_2[k]]

    train_dataset = TensorDataset(
        torch.tensor(train_input_ids),
        torch.tensor(train_mask),
        clss,
        train_mask_cls_2,
        out)

    pickle.dump(train_dataset,open(prefix+'_dataset.pickle','wb'))

    K=len(train_dataset)
    train_2=TensorDataset(torch.stack([train_dataset[i][0] for i in range(K)]),
                          torch.stack([train_dataset[i][1] for i in range(K)]),
                          torch.stack([train_dataset[i][2] for i in range(K)]),
                          torch.stack([train_dataset[i][3] for i in range(K)]),
                          torch.stack([train_dataset[i][4] for i in range(K)]))

    print(batch_size)

    dataloader = DataLoader(
                train_2,
                sampler = RandomSampler(train_2),
                batch_size = batch_size)

    pickle.dump(dataloader,open(prefix+'_dataloader.pickle','wb'))
    return dataloader,train_2

def make_text(texte,j=5,s=True,t=True,seuil=2,lem=False,sc=3):
    '''
    Fonction permettant le nettoyage du texte et sa préparation pour l'utilisation dans les modèles de résumé.
    @texte : le texte à nettoyer, une liste de phrases.
    @j : n_jobs, nombre de cpu, attention, plutôt ne pas en mettre beaucoup (2 par exemple) à cause de la mémoire.
    @s : dummy pour indiquer si on coupe les paragraphes, fixé à True, doit l'être sir vous ne mettez qu'un paragraphe à la fois.
    @t : dummy pour indiquer l'activation de la sélection des mots suffisamment grand, fixé à True.
    @seuil : seuil pour la taille des mots, fixé à 2.
    @lem : activation de la lemmatisation, fixé à False
    @sc : seuil pour le nombre de caractères dans la phrase, fixé à 3.
    '''
    WC=Word_Cleaning(n_jobs=j,#Pas trop, attention à la mémoire attribuée à chaque worker !
                sentence=s, #est-ce qu'on coupe ce qu'il y a dans la liste pour en faire des phrases ? Oui
                threshold=t, #On active la sélection de mots suffisamment grand
                seuil=seuil, #seuil pour la taille des mots
                lemma=lem, #est-ce qu'on lemmatise ?
                seuil_carac=sc) #nombre de caractères dans la phrase
    text=WC.make_documents(texte)
    empty=[]
    for i in range(len(text)):
        if (len(text[i])==0) or text[i][0]=='':
            empty.append(i)
    text=WC.remove_empty(text)
    if s:
        text=functools.reduce(operator.iconcat,text,[])
        
#     return text,empty
#     if len(text[0])>0:
    #text=WC.remove_empty(text)
#     try:
#         text=[i for i in text if ]
    return text,empty
#     else:
#         return text,empty

def make_DL_resume(texte,cpu,choose_model,k=3,camem=None,vs=12000,sp=1,tok='MLSUM_tokenizer.model',tr=False,get_score_only=False,x=3,time_=False):
    '''
    Fonction permettant d'utiliser les modèles de Deep Learning en torch pour produire des résumés automatiques.
    @texte : une liste de phrases ou une liste de listes de phrases.
    @cpu : nombre de cpu à utiliser, je conseille peu de cpu pour garder de la mémoire.
    @choose_model : le nom du modèle à aller chercher. si vous ne les connaissez pas, ne mettez rien et l'erreur affichera les noms des modèles.
    @k : nombre de phrases à retenir.
    @camem : modèle camembert à utiliser, sinon le modèle from_pretrained('camembert_base') sera utilisé.
    @vs : taille du vocabulaire pour l'encoding, paramètre pour le tokenizer. Fixé à 12000 par convenance.
    @sp : split, fixé à 1 car cette fonction n'est pas pour l'entraînement.
    @tok : le chemin vers le tokenizer.
    @tr : dummy pour l'entraînement, fixé à False donc.
    @get_score_only : dummy pour ne récupérer que les index des phrases et non les phrases elles-mêmes.    
    '''
    s1=time()
    try :
        assert choose_model in ['SMHA','Simple','Net','Multi']
    except:
        raise ValueError("Attention, le nom de votre modèle n'est pas correcte, il doit faire partie des noms suivants :\nSMHA (pour Self Multi Head Attention),\nSimple (pour Simple Linear Classifier),\nNet (pour Convolutional Network),\nMulti (pour Multi Linear)")
    
    try:
        assert np.sum([1 if choose_model in i else 0 for i in os.listdir()])>0
    except:
        raise ValueError("Attention, ce que vous avez entré pour la variable choose_model ne semble pas convenir. Vous ne possédez probablement pas le modèle dans votre dossier ou l'avez mal rebaptisé. Les noms corrects sont les suivants :\nSMHA_Linear_classifier.pt,\nSimple_Classifier.pt,\nNet.pt,\nMulti_Linear_Classifier.pt")
        
    if camem==None:
        from transformers import CamembertModel
        camem=CamembertModel.from_pretrained("camembert-base")

    if time_:
        s2=time()   
        print("checks :",round((s2-s1)/60,2),"minutes") 

    Text=Parallel(n_jobs=cpu)(delayed(make_text)(t) for t in texte)
    
    text=[Text[i][0] for i in range(len(Text))]
    empty=[Text[i][1] for i in range(len(Text))]
    
    dico=Make_Extractive(cpu).make_encoding(text,voc_size=vs,split=sp,tokenizer=tok,training=tr)

    if time_:
        s3=time()
        print("nettoyage :",round((s3-s2)/60,2),"minutes") 

    if choose_model=='SMHA':
        model=SMHA_Linear_classifier(torch.Size([512,768]),8,768)
        model.load_state_dict(torch.load('SMHA_Linear_classifier.pt',map_location=torch.device('cpu')))
        model.eval()
    
    elif choose_model=='Simple':
        model=Simple_Classifier(camem.config.hidden_size)
        model.load_state_dict(torch.load('Simple_Classifier.pt',map_location=torch.device('cpu')))
        model.eval()
    
    elif choose_model=='Net':
        model=Net(2**8,2**6,2,2,2,2)
        model.load_state_dict(torch.load('Net.pt',map_location=torch.device('cpu')))
        model.eval()    
        
    elif choose_model=='Multi':
        model=Multi_Linear_Classifier(camem.config.hidden_size)
        model.load_state_dict(torch.load('Multi_Linear_Classifier.pt',map_location=torch.device('cpu')))
        model.eval()    
    
    if time_:
        s4=time()
        print("sélection modèle :",round((s4-s3)/60,2),"minutes") 
    
    try:
        y=model(camem(torch.Tensor(dico['input']).long(),torch.Tensor(dico['mask']).long()).last_hidden_state.detach())
    except:
        y=[]
        l=len(dico['input'])
        for i in range(x):
            y.append(
                model(
                    camem(torch.Tensor(dico['input']).long()[int((i/2)*l):int(((i+1)/2)*l)],
                          torch.Tensor(dico['mask']).long()[int((i/2)*l):int(((i+1)/2)*l)]).last_hidden_state.detach()))
        y=torch.cat(y)
    
    if time_:
        s5=time()
        print("Camembert :",round((s5-s4)/60,2),"minutes")
    
    del model
    gc.collect() 
    
    t=torch.mul(dico['mask_cls'],y).topk(3)
    vec=[(dico['mask_cls'][i]==torch.tensor(1)).nonzero() for i in range(len(dico['mask_cls']))]
    values=t[0]
    indice=t[1]
    Ve=[]
    Va=[]
    I=[]
    ind=0
    for i in range(len(dico['trace'])):
        Ve.append(vec[ind:ind+dico['trace'][i]])
        Va.append(values[ind:ind+dico['trace'][i]])
        I.append(indice[ind:ind+dico['trace'][i]])
        ind+=dico['trace'][i]
        
    Va=[v.reshape(-1) for v in Va]
    I=[v.reshape(-1) for v in I]
    
    I2=[torch.cat([I[k][i*3:(i+1)*3]+512*i for i in range(int(len(I[k])/3))]) for k in range(len(I))]
    Ve2=[torch.cat([Ve[k][i]+512*i for i in range(len(Ve[k]))]).squeeze(1) for k in range(len(Ve))]

    index=[torch.div(Va[i],Va[i].sum()).topk(k=3)[1] for i in range(len(Va))]
    vrai_index=[I2[i][index[i]] for i in range(len(texte))]

    rindex=[torch.cat([(Ve2[h]==vrai_index[h][i]).nonzero().squeeze(1) for i in range(k)]) for h in range(len(Ve2))]
    
    a=[[i for i in range(len(texte[k]))] for k in range(len(texte))]
    
    if time_:
        s6=time()
        print("Récupération index :",round((s6-s5)/60,2),"minutes") 
    
    for i in range(len(texte)):
        for k in empty[i]:
            a[i].remove(k)
    Index_final=[[a[k][i] for i in rindex[k]] for k in range(len(rindex))]
    if get_score_only:
        return Index_final#,vrai_index,Ve2,I2
    else:
        resu=[[texte[i][k] for k in Index_final[i]] for i in range(len(text))]
        return resu


def make_U_resume(sequence,type_,k,cpu=2,modele=None,tok_path=None,get_score_only=False,seuil=70,weights=True,alpha=0.2,frac=0.25):
    '''
    Fonction permettant de produire des résumés sans deep learning.
    @sequence : liste de phrases.
    @type_ : type de modèles que vous allez utiliser, si vous ne savez pas, ne mettez rien et l'erreur affichera les options.
    @k : nombre de phrases désirées.
    @cpu : nombre de cpu pour la parallélisation, fixé à 2 pour garder de la mémoire disponible.
    @modele : le modele BERT ou W2V à utiliser pour l'embedding.
    @tok_path : le chemin vers le tokenizer.    
    '''
    
    if type_=='TextRankBert':
        try:
            TR=TextRank(tok_path,cpu=cpu)
            fnct=partial(TR.make_resume,
                           type='bert',
                           modele=modele,
                           k=k,
                           get_score_only=get_score_only,
                           s=seuil,
                           weights=weights,
                           alpha=alpha,frac=frac)
        except:
            print("Êtes-vous certain d'avoir mis le chemin du tokenizer en ayant spécifié que vous vouliez un embedding BERT ?\n Êtes-vous certain d'avoir spécifié un tokenizer correct ?")
    
    elif type_=='TextRankWord2Vec':
        TR=TextRank(cpu=cpu)
        fnct=partial(TR.make_resume,
                        type='word2vec',
                        modele=modele,
                        k=k,
                        get_score_only=get_score_only,
                        s=seuil,
                        weights=weights,
                        alpha=alpha,frac=frac)
    
    elif type_=='BertScore':
        if modele!=None:
            BS=BERTScore(tok_path,camem=modele,cpu=cpu)
        else:
            BS=BERTScore(tok_path,cpu=cpu)
        
        if get_score_only:
            fnct=BS.make_score      
        else:
            fnct=partial(BS.make_summary,
                              k=k)
        
    elif type_=='Lead3':
        fnct=partial(Lead_3,k=k)
    
    elif type_=='Random':
        fnct=Random_summary
    
    else:
        raise ValueError("Il semblerait que vous n'ayez pas spécifié de type pour le résumé.\nVous pouvez spécifier les types suivants : TextRankBert,TextRankWord2Vec,BertScore,Lead3,Random")
      
    res=fnct(sequence)
    return res

def Resume(texte,DL,cpu=2,type_=None,modele=None,choose_model=None,k=3,vs=12000,sp=1,tok='MLSUM_tokenizer.model',tr=False,get_score_only=False,s=True,t=True,seuil=2,lem=False,sc=3,weights=True,alpha=0.2,frac=0.25): #,camem=None
    '''
    Fonction produisant le résumé. 
    @texte : liste de listes de phrases. Autrement dit, vous avez une liste de paragraphes, vous les tronquez à chaque point, et vous obtenez une liste de liste de phrases.
    @DL : dummy pour indiquer si vous désirez utiliser des méthodes de Deep Learning.
    @cpu : nombre de CPU à utiliser, fixé à 2 pour garder de la mémoire disponible (l'embedding est gourmand).
    @type_ : nom du modèle si vous ne choisissez pas DL.
    @choose_model : nom du modèle si vous choisissez DL.
    @k : nombre de phrases, fixé à 3.
    @camem : le modèle BERT pour le DL.
    @vs : taille du vocabulaire pour le tokenizer, fixé à 12000.
    @sp : split, pour le split train test si entraînement. Fixé à 1 car on n'entraîne pas.
    @tok : chemin vers le tokenizer.
    @tr : dummy pour savoir si l'on entraine, ici non.    
    '''
    
    if type(texte[0])==str:
        if type(texte)!=list:
            texte=[i.split('.') for i in texte.tolist()]
        else:
            texte=[i.split('.') for i in texte]
    
    
    
    if DL:
        res=make_DL_resume(texte,cpu=cpu,choose_model=choose_model,k=k,camem=modele,vs=vs,sp=sp,tok=tok,tr=tr,get_score_only=get_score_only)
        return res
    
    
    
    else:
        start_time=time()
        mk=partial(make_text,j=cpu,s=s,t=t,seuil=seuil,lem=lem,sc=sc)
        Text=Parallel(n_jobs=cpu)(delayed(mk)(t) for t in texte)
    
        text=[Text[i][0] for i in range(len(Text))]
        WC=Word_Cleaning(n_jobs=cpu,#Pas trop, attention à la mémoire attribuée à chaque worker !
                sentence=s, #est-ce qu'on coupe ce qu'il y a dans la liste pour en faire des phrases ? Oui
                threshold=t, #On active la sélection de mots suffisamment grand
                seuil=seuil, #seuil pour la taille des mots
                lemma=lem, #est-ce qu'on lemmatise ?
                seuil_carac=sc)
        text=WC.remove_empty(text)
        empty=[Text[i][1] for i in range(len(Text))]
        end_time=time()
        print("Le processing du text a pris :",round((end_time-start_time)/60,2),"minutes")
        mur=partial(make_U_resume,type_=type_,k=k,cpu=cpu,modele=modele,tok_path=tok,
                            get_score_only=get_score_only,
                            weights=weights,alpha=alpha,frac=frac)
        
        if get_score_only:
            res=[mur(t) for t in text]
            end_2=time()
            print("La production des résumés a pris :",round((end_2-end_time)/60,2),"minutes")
            return res,text
        
        else:
            res=[mur(text[i]) for i in range(len(text))]
            end_2=time()
            print("La production des résumés a pris :",round((end_2-end_time)/60,2),"minutes")
            return res,text

def make_new_paragraphes(tcp,trace):
   paragraphe=[]
   paragraphe.append(tcp[:trace[-1][0]])
   if len(trace[-1])>1:
      for i in range(1,len(trace[-1])):
         paragraphe.append(tcp[trace[-1][i-1]:(np.sum(trace[-1][:i])+trace[-1][i])])
   return paragraphe

def make_new_sortie(sortie,index=None,k=2):
    '''
    Fonction pour transformer les sorties de dimension x en dimension k.
    '''
    ouais=torch.zeros(sortie.size())
    if index==None:
        if len(sortie)>=2:
            ouais[sortie.topk(k)[1]]=1
        else:
            ouais[sortie.topk(1)[1]]=1
    else:
        ouais[index[:k]]=1
    return ouais


def comparaison(simple3,score_vrai):
    assert len(simple3)==len(score_vrai)
        
    F1=F1_score()
    # tp=[]
    # fp=[]
    # fn=[]
    # p=[]
    # r=[]
    # f=[]
    
    # for i in tqdm(range(len(simple3))):
        # tp.append(F1.true_positive_mean(simple3[i],score_vrai[i]))
        # fp.append(F1.false_positive_mean(simple3[i],score_vrai[i]))
        # fn.append(F1.false_negative_mean(simple3[i],score_vrai[i]))
        # p.append(F1.precision(simple3[i],score_vrai[i]))
        # r.append(F1.recall(simple3[i],score_vrai[i]))
        # f.append(F1(simple3[i],score_vrai[i]))
    tp=F1.true_positive_mean(simple3,score_vrai)
    fp=F1.false_positive_mean(simple3,score_vrai)
    fn=F1.false_negative_mean(simple3,score_vrai)
    p=F1.precision(simple3,score_vrai)
    r=F1.recall(simple3,score_vrai)
    f=F1(simple3,score_vrai)
        
    # mtp=torch.mean(torch.tensor(tp))
    # mfp=torch.mean(torch.tensor(fp))
    # mfn=torch.mean(torch.tensor(fn))
    # mp=torch.mean(torch.tensor(p))
    # mr=torch.mean(torch.tensor(r))
    # mf=torch.mean(torch.tensor(f))
    # resultat=[mtp,mfp,mfn,mp,mr,mf]
    
    return [tp,fp,fn,p,r,f]

def make_compa(n,s):
    sortie_multi=pickle.load(open(n,'rb'))

    if type(sortie_multi[0])!=torch.Tensor:
        sortie_multi=[torch.tensor(i).to(torch.long) for i in sortie_multi]

    S=[]
    erreur=[]
    if len(sortie_multi)==len(s):
        for i in range(len(s)):
            try:
                S.append(make_new_sortie(s[i],sortie_multi[i]))
            except:
                print("Unexpected error:", sys.exc_info())
                break
                print(i)
                erreur.append(i)

        try:
            if erreur[0]==201:
                s_prime=s[:201]+s[202:]
                Sta=comparaison(torch.cat(S),torch.cat(s_prime))
                return Sta
        except:
            Sta=comparaison(torch.cat(S),torch.cat(s))
            return Sta
    else:
        print(n)