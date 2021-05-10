###################################################################################################"
######### Création séquence pour BERT
####################################################################################################"
#%% 
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
import psutil
from joblib import Parallel,delayed
from functools import partial
import time
import torch
import torch.nn as nn
#%%
##### On crée le fichier ligne par ligne article+résumé pour l'entraînement du tokenizer
###################
texte_entier=pickle.load(open(chemin_d+"OrangeSum_texte_entier_clean.pickle",'rb'))
b=texte_entier[0]+'\n'
for i in tqdm(range(len(texte_entier)-1)):
    b+=unidecode(texte_entier[i+1])+'\n'
with open(chemin_d+'OrangeSum_texte_entier_clean.txt','w') as f:
    f.write(b)

#%%
##### On entraîne le tokenizer
###################
import sentencepiece as spm 
#L'input doit être un fichier .txt
FUES=spm.SentencePieceTrainer.train(
    input=chemin_d+'OrangeSum_texte_entier_clean.txt', #chemin vers le fichier txt, un doc par ligne
    vocab_size=12000, #taille du vocab, peut être augmenté, ne doit pas être trop grand par rapport aux mots des documents
    model_prefix='FUES', #nom du modèle, French Unsupervised Exctractive Summarizer
    model_type='bpe') #Type de modèle Byte-Pair Encoding (Sennrich et al 2016)
#%%
##### On rentre notre SentencePiece model dans le tokenizer Camembert
######################################################################

from transformers import CamembertTokenizer
tokenizer=CamembertTokenizer(chemin_d+'FUES.model')
#%%
#### On charge les articles pour les avoir en format différent
##############################################################
articles=pickle.load(open(chemin_d+'OrangeSum_articles_phrases_clean.pickle','rb'))
headings=pickle.load(open(chemin_d+'OrangeSum_headings_phrases_clean.pickle','rb'))


##### On charge l'output qu'on a créé pour OrangeSum Extractive
##################################################################

output=pickle.load(open(chemin_d+"Output_OrangeSum.pickle",'rb'))
output=[i for i in output if len(i)>0]
print(len(output))

##### On va modifier (brutalement) la labellisation pour la mettre sur {0,1}
#############################################################################

output=[
    np.nan_to_num(i) if np.isnan(i).sum()>0 else i for i in output]

######## On crée un output de label 0,1
output_=[] 
for i in output:
    c=np.zeros(len(i))
    c[[list(i).index(k) for k in sorted(i)[-3:]]]=1
    output_.append(c)
output_
#%%
#### Mask cls : un exemple
############################
mask_cls_1=torch.zeros(512)
for i in train_clss[0]:
    mask_cls_1[i]=1
print(mask_cls_1)


#%%
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
                out=[]

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

def make_mask_cls(train_clss,dim=512):
    mask_cls_1=torch.zeros(dim)
    for i in train_clss:
        mask_cls_1[i]=1
    return mask_cls_1

def make_tensor_clss(clss):
    index=[512-len(i) for i in clss]
    vect=[clss[i]+list(np.zeros(index[i])) for i in range(len(index))]
    clss=torch.as_tensor(vect)
    return clss

#%%
cpu=psutil.cpu_count()
print("Nombre de coeur utilisé :",cpu)


start=time.time()
encod_articles_=partial(encod_articles,tokenizer=tokenizer)
articles_encodees=Parallel(n_jobs=cpu)(delayed(encod_articles_)(arti,out) for arti,out in zip(articles,output_))
end=time.time()
print("Durée de la parallélisation :",round((end-start)/60,3),"minutes")
#%%
print("Il y a ",len(articles_encodees),"articles associés à un output")
print("Pour chaque article, il y a ",len(articles_encodees[0]),"vecteurs")
print("Chaque article est découpé en un ou plusieurs bouts, ici par exemple en",len(articles_encodees[0][0]),"\ncela dépend de la taille du paragraphe pour être sûr d'avoir bien toutes les phrases.")
print("Chaque vecteur est de dimension",len(articles_encodees[0][0][0]),"comme l'exige bien le format d'input de BERT")
#%%
pickle.dump(articles_encodees,open(chemin_d+"OrangeSum_articles_ALL.pickle",'wb'))
#%%
##### On crée nos tensors et la data loader pour la boucle d'entraînement
###################

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

split_border=int(len(articles_encodees)*0.8)
train=articles_encodees[:split_border]

train_input_ids=[train[i][0][k] for i in range(len(train)) for k in range(len(train[i][0]))]
train_mask=[train[i][1][k] for i in range(len(train)) for k in range(len(train[i][1]))]
train_segs=[train[i][2][k] for i in range(len(train)) for k in range(len(train[i][2]))]
train_clss=[train[i][3][k] for i in range(len(train)) for k in range(len(train[i][3]))]
clss_index_train=[len(i) for i in train_clss]
train_mask_cls=torch.as_tensor([list(make_mask_cls(t)) for t in train_clss])
train_clss=make_tensor_clss(train_clss)
train_output=make_tensor_clss([train[i][4][k] for i in range(len(train)) for k in range(len(train[i][4]))])

dico_train={
    'input':train_input_ids,
    'mask':train_mask,
    'segs':train_segs,
    'clss':train_clss,
    'clss_index':clss_index_train,
    'output':train_output,
    'mask_cls':train_mask_cls
}
pickle.dump(dico_train,open(chemin_d+'dico_train.pickle','wb'))

test=articles_encodees[split_border:]
test_input_ids=[test[i][0][k] for i in range(len(test)) for k in range(len(test[i][0]))]
test_mask=[test[i][1][k] for i in range(len(test)) for k in range(len(test[i][1]))]
test_segs=[test[i][2][k] for i in range(len(test)) for k in range(len(test[i][2]))]
test_clss=[test[i][3][k] for i in range(len(test)) for k in range(len(test[i][3]))]
clss_index_test=[len(i) for i in test_clss]
test_mask_cls=torch.as_tensor([list(make_mask_cls(t)) for t in test_clss])
test_clss=make_tensor_clss(test_clss)
test_output=make_tensor_clss([test[i][4][k] for i in range(len(test)) for k in range(len(test[i][4]))])

dico_test={
    'input':test_input_ids,
    'mask':test_mask,
    'segs':test_segs,
    'clss':test_clss,
    'clss_index':clss_index_test,
    'output':test_output,
    'mask_cls':test_mask_cls
}
pickle.dump(dico_test,open(chemin_d+'dico_test.pickle','wb'))
#%%
dico_train=pickle.load(open(chemin_d+'dico_train.pickle','rb'))
#%%
train_input_ids=dico_train['input']
train_mask=dico_train['mask']
clss=make_tensor_clss(dico_train['clss'])
train_mask_cls=dico_train['mask_cls']
train_output=dico_train['output']


#%%
from transformers import CamembertModel,BertModel,RobertaModel
camem=CamembertModel.from_pretrained("camembert-base")
camem.config.hidden_size
#camem=BertModel.from_pretrained('bert-base-uncased')
# camem=RobertaModel.from_pretrained('roberta-base')# ROBERTA PROBLEME
# T=camem(torch.tensor(train_input_ids[0:3]),attention_mask=torch.tensor(train_mask[0:3]))#,token_type_ids=torch.tensor(train_segs[0:3]))
# T
#%%
print(T.last_hidden_state.shape) # nbre de docs * dim input (512) * 768
# il faudrait réduire 512 -> nbr de phrases du doc
#comme ça on aurait dim_sortie_finale = nbr de docs * nbr de phrases * 1 (ou 2)
T.pooler_output.shape # nbre de docs * 768
#%%
import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()

    def forward(self, x):#, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.relu(h) #* mask_cls.float()
        return sent_scores

def select_sent(phrase,clss,k=3):
    index_phrase=torch.topk(phrase,k)[1]
    pred_phrase=torch.zeros(clss.shape)
    index_1=[[clss[k].tolist().index(int(index_phrase[k][i])) for i in range(len(index_phrase[k]))] for k in range(clss.shape[0])]
    index_2=[[i] for i in range(clss.shape[0])]
    pred_phrase[index_2,index_1]=torch.ones(index_phrase.shape)
    return pred_phrase

from sklearn.metrics import confusion_matrix
def confusion_output(sent,output,clss_index):
    a=[confusion_matrix(output[i],sent[i]) for i in range(sent.shape[0])]
    c=[a[i][0][0]+a[i][1][1] for i in range(sent.shape[0])]
    score_total=[round((c[i]-(512-clss_index[i]))/(clss_index[i]),3) for i in range(sent.shape[0])]
    score_1=[(a[i][1][1])/3 for i in range(sent.shape[0])]
    return score_total,score_1


class Summarizer(nn.Module):
    def __init__(self, device):#args, , load_pretrained_bert = False, bert_config = None):
        super(Summarizer, self).__init__()
        self.device = device
        self.bert =CamembertModel.from_pretrained("camembert-base")
        #BertModel.from_pretrained('bert-base-uncased')
        #Bert(args.temp_dir, load_pretrained_bert, bert_config)
        self.encoder = Classifier(self.bert.config.hidden_size)
        self.select_sent=select_sent
        # self.score=confusion_output
        self.to(device)

    def forward(self,x,mask, mask_cls,clss,output,k=3):#,segs):#, sentence_range=None): #segs, 
        #x input_ids
        #Segs = Segment pour phrases (0 ou 1), marche pas dans un RoBERTa
        #clss index du début des phrases 
        #mask_cls vecteur pour passer de l'embedding au cls, en gros sélectionne le bon index des vecteurs de l'embedding qu'on va utiliser pour faire la classif
        top_vec= self.bert(x, mask)#, segs)
        # sents_vec=self.sent_vec(last,clss)

        # sents_vec = top_vec[0][torch.arange(top_vec[0].size(0)).unsqueeze(1), clss]
        # sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(top_vec.last_hidden_state).squeeze(-1)#, mask_cls
        sent_scores_masked = torch.mul(sent_scores,mask_cls)
        sent_scores_masked = self.select_sent(sent_scores_masked,clss,k=k)
        # score=self.score(sent_scores,output)
        return sent_scores_masked,sent_scores,top_vec.last_hidden_state#,score#, mask_cls
#%%
summa=Summarizer(device='cpu')
#%%
topvec=summa(x=torch.tensor(train_input_ids[0:3]),
            mask=torch.tensor(train_mask[0:3]),
            clss=clss[0:3],
            mask_cls=torch.as_tensor([list(train_mask_cls[i]) for i in range(3)]),
            output=train_output[:3])
#%%

multihead_attn = nn.MultiheadAttention(768, 8)
query=topvec[2]
key=topvec[2]
value=topvec[2]
attn_output, attn_output_weights = multihead_attn(query, key, value)


#%%

tens=torch.as_tensor([list(train_mask_cls[i]) for i in range(3)])
phrase=torch.mul(topvec[0],tens)

# torch.max(phrase).item()
#phrase.argmax()
index_phrase=torch.topk(phrase,3)[1]
pred_phrase=torch.zeros(clss[:3].shape)
#%%
############################################
###### Problème important (10/05/21) : que faire quand moins de trois phrases sont prédites ?
###### Pourquoi ne pas laisser le score et choisir en fonction de ça ?
############################################
#En fait c'est à ce moment qu'on aurait besoin d'un segment mais bon...
#En gros le problème c'est : si le réseau associe un score élevé à un tokens qui n'est pas un début de phrase, il sera invisibilisé par la multiplication du mask
#Donc, comment on fait ? Pour le moment, on perd purement et simplement l'info
#Je propose d'être plus général, et de laisser le réseau prédire ce qu'il veut en récupérant différemment l'info
#Par exemple, en prenant simplement les 3 max scores sur 512 et repérer la phrase à laquelle appartient le token
#Ensuite simplement récupérer la phrase, et la suite reste la même 


index_1=[[clss[k].tolist().index(int(index_phrase[k][i])) for i in range(len(index_phrase[k]))] for k in range(3)]
#%%
index_2=[[i] for i in range(3)] #clss.shape[0]
pred_phrase[index_2,index_1]=torch.ones(index_phrase.shape)
pred_phrase
# %%
a=[confusion_matrix(train_output[i],pred_phrase[i]) for i in range(len(pred_phrase))]
c=[a[i][0][0]+a[i][1][1] for i in range(len(pred_phrase))]
score_total=[round((c[i]-(512-clss_index_train[i]))/(clss_index_train[i]),3) for i in range(len(pred_phrase))]
print(score_total)
score_1=[(a[i][1][1])/3 for i in range(len(pred_phrase))]
print(score_1)
print("Au total il y a ",round(np.mean(score_total),2),"bonnes prédictions,\n",np.sum(score_1),"bonnes phrases sont prédites")
#%%
tes
#%%
train_dataset = TensorDataset(
    torch.tensor(train_input_ids),
    torch.tensor(train_mask),
    train_clss,
    torch.as_tensor([list(train_mask_cls[i]) for i in range(len(train_mask_cls))]),
    train_output)

validation_dataset = TensorDataset(
    torch.tensor(test_input_ids),
    torch.tensor(test_mask),
    test_clss,
    torch.as_tensor([list(test_mask_cls[i]) for i in range(len(test_mask_cls))]),
    test_output)

batch_size=32

dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size)
#%%
pickle.dump(dataloader,open(chemin_d+'dataloader.pickle','wb'))
pickle.dump(train_dataset,open(chemin_d+'train.pickle','wb'))
pickle.dump(validation_dataset,open(chemin_d+'validation.pickle','wb'))

#%%
device = torch.device("cpu")
training_stats = []
epochs=3
summa=Summarizer(device='cpu')

# topvec=summa(x=torch.tensor(train_input_ids[0:3]),
#             mask=torch.tensor(train_mask[0:3]),
#             clss=clss[0:3],
#             mask_cls=torch.as_tensor([list(train_mask_cls[i]) for i in range(3)]),
#             output=train_output[:3])

# Boucle d'entrainement
for epoch in range(0, epochs):
     
    print("")
    print(f'########## Epoch {epoch+1} / {epochs} ##########')
    print('Training...')
 
 
    # On initialise la loss pour cette epoque
    total_train_loss = 0
 
    # On met le modele en mode 'training'
    # Dans ce mode certaines couches du modele agissent differement
    summa.train()
 
    # Pour chaque batch
    for step, batch in enumerate(dataloader):
 
        # On fait un print chaque 40 batchs
        if step % 40 == 0 and not step == 0:
            print(f'  Batch {step}  of {len(train_dataloader)}.')
         
        # On recupere les donnees du batch
        input_id = batch[0].to(device)
        mask = batch[1].to(device)
        clss = batch[2].to(device)
        mask_cls=batch[3].to(device)
        output=batch[4].to(device)
 
        # On met le gradient a 0
        summa.zero_grad()        
 
        # On passe la donnee au model et on recupere la loss et le logits (sortie avant fonction d'activation)
        loss, logits = summa(input_id,
        mask,clss,mask_cls,output)
 
        # On incremente la loss totale
        # .item() donne la valeur numerique de la loss
        total_train_loss += loss.item()
 
        # Backpropagtion
        loss.backward()
 
        # On actualise les parametrer grace a l'optimizer
        optimizer.step()
 
    # On calcule la  loss moyenne sur toute l'epoque
    avg_train_loss = total_train_loss / len(train_dataloader)   
 
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))  
     
    # Enregistrement des stats de l'epoque
    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
        }
    )
#%%
print("Model saved!")
torch.save(summa.state_dict(), chemin_d+"model_essai.pt")












#%%
# filee=open(chemin_d+"cnndm.test.0.bert.pt",'r').read()
filee=torch.load(chemin_d+"cnndm.train.0.bert.pt")
#%%
print(len(filee),len(filee[0]))
print(filee[0].keys())
print(len(filee[0]['src_txt']))
print(len(filee[0]['src']))
print(len(filee[0]['labels']))
nphrase=[]
ntok=[]
ncls=[]
nlab=[]
for i in range(len(filee)):
    nphrase.append(len(filee[i]['src_txt']))
    ntok.append(len(filee[i]['src']))
    ncls.append(len(filee[i]['clss']))
    nlab.append(len(filee[i]['labels']))

print("Phrases :",np.mean(nphrase),np.std(nphrase),np.median(nphrase),max(nphrase))
print("Tokens :",np.mean(ntok),np.std(ntok),np.median(ntok),max(ntok))
print("cls :",np.mean(ncls),np.std(ncls),np.median(ncls),max(ncls))
print("Labels :",np.mean(nlab),np.std(nlab),np.median(nlab),max(nlab))

# Donc le nombre de phrases par article varie 
#Le nombre de tokens varie aussi beaucoup
# Par contre on a bien dim(label)=nombre de phrases 
# Mais cette dim varie à chaque fois puisque le nb de phrases varie !!
#%%
import matplotlib.pyplot as plt
f,a=plt.subplots(1,figsize=(14,10))
a.hist(ncls,density=True)
a.hist(ntok,density=True)
a.hist(nphrase,density=True)

# a.set(xlabel="nombre de mots",ylabel='quantité de phrases',
#       title='Distribution du nombre de tokens')
# plt.legend(['Tokens','mots'])
#%%
###############################
##### CREATION DU MODELE (TÊTE)
###############################

#Il est possible de modifier  la dimension d'entrée de BERT, via les commandes config
# par exemple CamembertConfig()

####### MultiHeadAttention Layer's Inputs
#embed_dim – total dimension of the model.
#num_heads – parallel attention heads.

#query: (L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
#key: (S,N,E) , where S is the source sequence length, N is the batch size, E is the embedding dimension.
#value: (S,N,E) where S is the source sequence length, N is the batch size, E is the embedding dimension.

import torch.nn as nn

class CustomCamembert(nn.Module):
    def __init__(self):#,num_labels=2): Nous on a pas de labels, donc pas besoin
        super(CustomCamembert,self).__init__()
        self.camembert=CamembertModel.from_pretrained("camembert-base")
        self.dropout = nn.Dropout(.05)
        self.classifier = nn.Linear(768, )
        self.MHA=nn.MultiheadAttention(embed_dim=768, num_heads=8)
        # nn.TransformerDecoderLayer(d_model=768, nhead=8)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        last_hidden_state , pooled_output = self.camembert(input_ids, token_type_ids, attention_mask)
        vec = self.MHA(last_hidden_state)      
        output=self.classifier(vec)
        return output

#### Initialize the model
camembert_new= CustomCamembert()
#%%
import torch.nn as nn

decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8)
memory = torch.rand(10,512,768)
tgt = torch.rand(1,512,768)# ça c'est vraiment la séquence des ids de l'output (dans un cas de translation)
#Donc nous on peut pas faire de TransformerDecoderLayer puisque ce n'est pas ça que nous faisons
out = decoder_layer(tgt, memory)
out
#%%
import torch
from models.model_builder import ExtSummarizer
from ext_sum import summarize
from newspaper import Article
url = "https://www.cnn.com/2020/05/29/tech/facebook-violence-trump/index.html" #@param {type: 'string'}
article = Article(url)
article.download()
article.parse()
print(wrapper.fill(article.text))

# Save input text into `raw_data/input.txt`
with open(chemin_d+'article_exemple.txt', 'w') as f:
    f.write(article.text)
# Load model
model_type = 'mobilebert' #@param ['bertbase', 'distilbert', 'mobilebert']
checkpoint = torch.load(f'checkpoints/{model_type}_ext.pt', map_location='cpu')
model = ExtSummarizer(checkpoint=checkpoint, bert_type=model_type, device='cpu')

# Run summarization
input_fp = chemin_d+'article_exemple.txt'
result_fp = chemin_d+'article_exemple_summary.txt'
summary = summarize(input_fp, result_fp, model, max_length=3)
print(summary)







#%%
####################
########## BARThez
####################

text_sentence = "Citant les préoccupations de ses clients dénonçant des cas de censure après la suppression du compte de Trump, un fournisseur d'accès Internet de l'État de l'Idaho a décidé de bloquer Facebook et Twitter. La mesure ne concernera cependant que les clients mécontents de la politique de ces réseaux sociaux."

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

barthez_tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez")
barthez_model = AutoModelForSeq2SeqLM.from_pretrained("moussaKam/barthez-orangesum-abstract")
# MBartForConditionalGeneration(
#   (model): MBartModel(
#     (shared): Embedding(50002, 768, padding_idx=1)
#     (encoder): MBartEncoder(
#       (embed_tokens): Embedding(50002, 768, padding_idx=1)
#       (embed_positions): MBartLearnedPositionalEmbedding(1026, 768)
input_ids = torch.tensor(
    [barthez_tokenizer.encode(text_sentence, add_special_tokens=True)]
)

barthez_model.eval()
predict = barthez_model.generate(input_ids, max_length=100)[0]


barthez_tokenizer.decode(predict, skip_special_tokens=True)
# %%
################# BERTSUM
get_corona_summary=open(chemin_d+'corona.txt','r').read()
from summarizer import Summarizer
model = Summarizer()
result = model(get_corona_summary, min_length=20)
summary = "".join(result)
print(summary)
# %%
