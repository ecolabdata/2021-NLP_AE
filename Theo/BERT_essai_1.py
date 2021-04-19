#%%
import torch
from fairseq.models.roberta import CamembertModel
camembert_path='C:/Users/theo.roudil-valentin/.cache/torch/hub/camembert-base/'
camembert = CamembertModel.from_pretrained(camembert_path)
# %%
masked_line = 'Le camembert est <mask> :)'
camembert.fill_mask(masked_line, topk=3)
# %%
line = "J'aime le camembert !"
tokens = camembert.encode(line)
last_layer_features = camembert.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 10, 768])
#%%
# Extract all layer's features (layer 0 is the embedding layer)
all_layers = camembert.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 13
assert torch.all(all_layers[-1] == last_layer_features)
#%%
########################################################################################################
#############   Prise en main      ###########################################################################################
##################################################################################################################
chemin_donnees="C:/Users/theo.roudil-valentin/Documents/Donnees/EI_txt/"
chemin_modele="C:/Users/theo.roudil-valentin/Documents/Donnees/Modele_Transformer/"

import torch
from fairseq.models.roberta import CamembertModel
camembert_path='C:/Users/theo.roudil-valentin/.cache/torch/hub/camembert-base/'
camembert = CamembertModel.from_pretrained(camembert_path)
# camembert.train()
#%%
#On import CamembertModel : 
#The bare CamemBERT Model transformer outputting raw hidden-states 
# without any specific head on top. 
#C'est exactement ce qu'on veut !
from transformers import CamembertModel
camembert=CamembertModel.from_pretrained("camembert-base")
#Autrement dit, il n'y a pas de "dernière couche" avec un objectif :
#C'est à nous de créer le(s) dernier(s) layer(s !!)
#%%
from transformers import AdamW
optimizer = AdamW(camembert.parameters(), lr=1e-5)

from transformers import CamembertTokenizer,CamembertForSequenceClassification
tokenizer=CamembertTokenizer.from_pretrained(camembert_path)

#%%
from pathlib import Path
text_batch = []
for src_file in Path(chemin_donnees).glob("**/*.txt"):
            print("🔥", src_file)
            text_batch +=  src_file.read_text(encoding="utf-8").splitlines()
            
#%%
ml=512
N=10
z=9
encoding = tokenizer(text_batch[z], return_tensors='pt',max_length=ml, padding='max_length', truncation=True)
# input_ids = encoding['input_ids']
# attention_mask = encoding['attention_mask']
#%%
################ COMPRENDRE LES ETAPES D'ENCODAGE
#Prenons l'exemple de la phrase 9 de notre ensemble de texte :
print(text_batch[z],len(text_batch[z]))
print(tokenizer.tokenize(text_batch[z]),len(tokenizer.tokenize(text_batch[z])))
print(encoding['input_ids'][0])
print(encoding['attention_mask'][0])
#%%
##################### Faire marcher camembert sur une ligne d'exemple
outputs=camembert(**encoding)
#%%
last_hidden_states = outputs.last_hidden_state
last_hidden_states
#%%
forward=camembert.forward(input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask'],
        encoder_hidden_states=last_hidden_states)

#%%
################### Config modèle (test)
# set special tokens
camembert.config.decoder_start_token_id = tokenizer.bos_token_id                                             
camembert.config.eos_token_id = tokenizer.eos_token_id

# sensible parameters for beam search
# set decoding params                               
camembert.config.max_length = 512
camembert.config.early_stopping = True
# camembert.config.no_repeat_ngram_size = 3
# camembert.config.length_penalty = 2.0
# camembert.config.num_beams = 4
camembert.config.vocab_size = camembert.config.encoder.vocab_size


#%%
# labels = torch.tensor([1,0]).unsqueeze(0)
# camembert.train(mode=True)
#%%
params = list(camembert.parameters())
print(len(params))
print(params[0].size())
#%%
camembert.eval()
out=camembert(input_ids[:10])
#%%
# model = CamembertForSequenceClassification.from_pretrained(
#     'camembert-base',
#     num_labels = 2)
loss = model.loss
loss.backward()
optimizer.step()
#%%
#####################
#### Première généralisation
#####################

import torch.nn as nn
################ Custom Camembert

class CustomCamembert(nn.Module):
    def __init__(self):#,num_labels=2): Nous on a pas de labels, donc pas besoin
        super(CustomCamembert,self).__init__()
        self.camembert=CamembertModel.from_pretrained("camembert-base")
        self.dropout = nn.Dropout(.05)
        self.classifier = nn.Linear(768, DIMENSION ???)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _ , pooled_output = self.roberta(input_ids, token_type_ids, attention_mask)
        logits = self.classifier(pooled_output)        
        return logits
#### Initialize the model
camembert_new= = CustomCamembert()

#%%
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
############ Exemples pour TensorDataset et DataLoader :
unit=50
dim=512
#Dans TensorDataset, tu peux mettre autant de tensor que tu veux en fait
#Il faudra juste bien se souvenir de leur rang pour bien les appeler une fois
#qu'on en aura besoin
ex_data=TensorDataset(torch.randn([unit,dim]),torch.randn([unit,dim]),
                        torch.randn([unit,dim]),torch.randn([unit,2]))
batch_size=32

#On définit un "chargeur de données", qui va sélectionner des groupes de taille batch_size
#de manière aléatoire dans les données qu'on lui donne, ici : ex_data
ex_dataloader = DataLoader(
            ex_data,
            sampler = RandomSampler(ex_data),
            batch_size = batch_size)

for step,batch in enumerate(ex_dataloader):
    print(batch[0])

#%%

################ Un exemple de training supervisé

train_dataset = TensorDataset(
    encoding['input_ids'][:split_border],
    encoding['attention_mask'][:split_border],
    sentiments[:split_border])
validation_dataset = TensorDataset(
    encoded_batch['input_ids'][split_border:],
    encoded_batch['attention_mask'][split_border:],
    sentiments[split_border:])



device = torch.device("cpu")
training_stats = []
epochs=3
 
# Boucle d'entrainement
for epoch in range(0, epochs):
     
    print("")
    print(f'########## Epoch {epoch+1} / {epochs} ##########')
    print('Training...')
 
 
    # On initialise la loss pour cette epoque
    total_train_loss = 0
 
    # On met le modele en mode 'training'
    # Dans ce mode certaines couches du modele agissent differement
    model.train()
 
    # Pour chaque batch
    for step, batch in enumerate(train_dataloader):
 
        # On fait un print chaque 40 batchs
        if step % 40 == 0 and not step == 0:
            print(f'  Batch {step}  of 
{len(train_dataloader)}.')
         
        # On recupere les donnees du batch
        input_id = batch[0].to(device)
        attention_mask = batch[1].to(device)
        sentiment = batch[2].to(device)
 
        # On met le gradient a 0
        model.zero_grad()        
 
        # On passe la donnee au model et on recupere la loss et le logits (sortie avant fonction d'activation)
        loss, logits = model(input_id, 
                             token_type_ids=None, 
                             attention_mask=attention_mask, 
                             labels=sentiment)
 
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
 
print("Model saved!")
torch.save(model.state_dict(), "./sentiments.pt")
#%%
########################################################################################################
#############    Distribution du nombre de tokens par phrase      ###########################################################################################
##################################################################################################################

chemin_donnees="C:/Users/theo.roudil-valentin/Documents/Donnees/EI_txt/"
chemin_modele="C:/Users/theo.roudil-valentin/Documents/Donnees/Modele_Transformer/"

import torch
camembert_path='C:/Users/theo.roudil-valentin/.cache/torch/hub/camembert-base/'

from transformers import CamembertTokenizer
tokenizer=CamembertTokenizer.from_pretrained(camembert_path)


from pathlib import Path
text_batch = []
for src_file in Path(chemin_donnees).glob("**/*.txt"):
            print("🔥", src_file)
            text_batch +=  src_file.read_text(encoding="utf-8").splitlines()
            
#%%
import time
ml=512
# N=10
# z=9
start=time.time()
encoding = tokenizer(text_batch, return_tensors='pt',max_length=ml, padding='max_length', truncation=True)
end=time.time()
print('La tokenization a durée :',round((end-start)/60,2),"minutes")
#%%
print(encoding['attention_mask'].shape)
distrib=[i.sum() for i in encoding['attention_mask']]


#%%
########################################################################################################
#############    Extractive summarization            ###########################################################################################
##################################################################################################################


######### 1.  Essai avec un SBERT
 
from rouge_score import rouge_scorer
keys=['rouge1', 'rougeL']
scorer = rouge_scorer.RougeScorer(keys, use_stemmer= True)

import spacy
nlp = spacy.load('fr_core_news_sm')
#%%
from sentence_transformers import SentenceTransformer
embedder=SentenceTransformer('C:/Users/theo.roudil-valentin/.cache/torch/sentence_transformers/sbert.net_models_distilbert-base-nli-mean-tokens_part')
# scorer.score(pred_summaries, gold_summaries)
# %%
import pandas as pd
import numpy as np
import sklearn
import pickle

chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"

df=pd.read_csv(chemin+'etudes_dataset_cleaned_prepared.csv',sep=";")
doc=df.docs[0]
# %%
texte=nlp(doc) #étape inutile sachant que le texte a déjà été nettoyé
sents = list(texte.sents) #liste des phrases dans le document
sents
# %%
min_len = 2 #nombre de mots minimum par phrases
#On ne garde que les phrases qui contiennent strictement plus de 2 mots
sents_clean = [sentence.text for sentence in sents if len(sentence)> min_len]
sents_clean = [sentence for sentence in sents_clean if len(sentence)!=0]
#%%
sents_embedding= np.array(embedder.encode(sents_clean, convert_to_tensor=True))
doc_embedding=sents_embedding.mean(axis=0)

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
def summary(sents_embedding,sents_clean):
    doc_embedding=sents_embedding.mean(axis=0)
    summa=[
        cos_sim(doc_embedding,sents_embedding[j]) 
        for j in range(sents_embedding.shape[0])
    ]
    idx=[summa.index(np.sort(summa)[-10:][z]) for z in range(10)]
    sentences=[sents_clean[k] for k in idx]
    return sentences

sen=summary(sents_embedding,sents_clean)
sen
#%%
######### 2.  Essai avec CamemBERT
def sentences_embeddings(sentences):
    wrong_tok=[]
    embed={}
    for s in sentences:
        try:
            tokens=camembert.encode(s)
            if len(tokens)<513:
                all_layers=camembert.extract_features(tokens, return_all_hiddens=True)
                embed[sentences.index(s)]=all_layers[0]
            else:
                print('La phrase',sentences.index(s),'est trop grande')
                wrong_tok.append(sentences.index(s))
                continue
        except:
            print('Bouuuhhh ça marche pas...')
            continue

    return embed,wrong_tok

Embeddings,wrong_tok=sentences_embeddings(sents_clean)
# %%
#problème du nombre de tokens par prhases 
def summary_camembert(embeddings):
    


    return sentences