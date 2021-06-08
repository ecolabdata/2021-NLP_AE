import numpy as np
import os  
from unidecode import unidecode
import re
from joblib import Parallel,delayed
from functools import partial
from tqdm import tqdm
import torch
import pickle
import warnings

class Word_Cleaning():
      def __init__(self,n_jobs,sentence=False,threshold=False,seuil=None,lemma=False,seuil_carac=None):
        super(Word_Cleaning,self).__init__
        self.cpu=n_jobs
        self.seuil=seuil
        self.lemma=lemma
        self.threshold=threshold
        self.seuil_carac=seuil_carac
        self.sentence=sentence
    
    
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
    if len(article)==len(output):
        encod_article=[]
        encod_mask=[]
        encod_phrase=[]

        segs=[]
        encod_segs=[]

        clss_=[0]
        encod_clss=[]

        out=[]
        output_=[]

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

        return encod_article,encod_mask,encod_segs,encod_clss,output_
    else:
        raise ValueError("Attention ! La dimension de l'article et de l'ouput sont différentes !")



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
        self.cpu=cpu
    
    @staticmethod
    def make_w2v_sentences(documents):
        '''
        @documents : une liste de documents, qui sont une liste de phrases
        '''
        sentences=[]
        for i in documents:
            for z in i:
                if len(z.split())>0:
                    sentences.append(z.split())
        return sentences

    def make_output(self,docs,summary,W2V=None):
       assert len(docs)==len(summary)

       sentence=self.make_w2v_sentences(docs)
       
       if W2V==None:
         import gensim
         try:
            W2V=gensim.models.Word2Vec(size=self.dim,window=self.fenetre,min_count=self.minimum)
         except:
            W2V=gensim.models.Word2Vec(vector_size=self.dim,window=self.fenetre,min_count=self.minimum)
         W2V.build_vocab(sentence)
         W2V.train(sentence,total_examples=W2V.corpus_count,epochs=self.epochs)
       
       text=[[i.split() for i in s] for s in docs]
       try:
          vocab=list(W2V.wv.vocab.keys())
       except:
          vocab=list(set(W2V.wv.key_to_index))

       summary=[[i for i in s.split() if i in vocab] for s in summary]
       score=[]
       erreur=[]
       for sent in tqdm(text):
          score_=[]
          for s in sent:
             try:
                score_.append(torch.stack(
                     [self.cosim(torch.as_tensor(W2V[i]),torch.as_tensor(W2V[summary[text.index(sent)]]))
                     for i in s]).mean())
             except:
                erreur.append([text.index(sent),sent.index(s)])
                print("Attention, l'élément",erreur[-1],"n'a pas pu être encodé.")
                continue

          score.append(torch.stack(score_))
       return score,text,summary,W2V,erreur

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

    def make_encoding(self,text,output,tokenizer=None,voc_size=None,prefix="essai",name="Tokenizer_input",split=0.8,dim=512):
       if tokenizer==None:
          assert voc_size!=None
          tokenizer=self.make_tokenizer(text,voc_size,prefix=prefix,name=name)
       doc_encod=partial(self.document_encoding,tokenizer=tokenizer,dim=dim)   
       encoding=Parallel(n_jobs=self.cpu)(delayed(doc_encod)(i,j) for i,j in zip(text,output))
       from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

       split_border=int(len(encoding)*split)
       train=encoding[:split_border]

       train_input_ids=[train[i][0][k] for i in range(len(train)) for k in range(len(train[i][0]))]
       train_mask=[train[i][1][k] for i in range(len(train)) for k in range(len(train[i][1]))]
       train_segs=[train[i][2][k] for i in range(len(train)) for k in range(len(train[i][2]))]
       train_clss=[train[i][3][k] for i in range(len(train)) for k in range(len(train[i][3]))]
       clss_index_train=[len(i) for i in train_clss]
       train_mask_cls=torch.as_tensor([list(self.make_mask_cls(t)) for t in train_clss])
       train_clss=self.make_tensor_clss(train_clss)
       train_output=self.make_tensor_clss([train[i][4][k] for i in range(len(train)) for k in range(len(train[i][4]))])

       dico_train={
         'input':train_input_ids,
         'mask':train_mask,
         'segs':train_segs,
         'clss':train_clss,
         'clss_index':clss_index_train,
         'output':train_output,
         'mask_cls':train_mask_cls
       }

       pickle.dump(dico_train,open(self.path+'/dico_train.pickle','wb'))
       
       if split<1:
         test=encoding[split_border:]
         test_input_ids=[test[i][0][k] for i in range(len(test)) for k in range(len(test[i][0]))]
         test_mask=[test[i][1][k] for i in range(len(test)) for k in range(len(test[i][1]))]
         test_segs=[test[i][2][k] for i in range(len(test)) for k in range(len(test[i][2]))]
         test_clss=[test[i][3][k] for i in range(len(test)) for k in range(len(test[i][3]))]
         clss_index_test=[len(i) for i in test_clss]
         test_mask_cls=torch.as_tensor([list(self.make_mask_cls(t)) for t in test_clss])
         test_clss=self.make_tensor_clss(test_clss)
         test_output=self.make_tensor_clss([test[i][4][k] for i in range(len(test)) for k in range(len(test[i][4]))])

         dico_test={
            'input':test_input_ids,
            'mask':test_mask,
            'segs':test_segs,
            'clss':test_clss,
            'clss_index':clss_index_test,
            'output':test_output,
            'mask_cls':test_mask_cls
         }
         pickle.dump(dico_test,open(self.path+'/dico_test.pickle','wb'))
         return dico_train,dico_test

       return dico_train