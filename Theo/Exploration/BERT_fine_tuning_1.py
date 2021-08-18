################################################################################################################################################################################
########### CamemBERT Fine-tuning ########################################################################################################################################################################
################################################################################################################################################################################

# Tentatives de fine-tuning de CamemBERT, la tâche étant trop compliqué, compte tenu
# du temps, de la puissance de calcul disponible, nous avons fait autrement.


#%%
import pandas as pd
import numpy as np
import sklearn
import pickle
import tensorflow as tf 
import keras
import torch
import gc
import psutil
from keras.layers import Input,Conv1DTranspose, Conv1D, MaxPooling1D, GlobalAveragePooling1D, UpSampling1D, Dense, Dropout, Activation, Lambda, Reshape, Flatten, Embedding, LSTM, TimeDistributed
from keras.models import Model,Sequential
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.metrics import RootMeanSquaredError
import time
import seaborn
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer, AdamW
#%%
chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"

df=pd.read_csv(chemin+'etudes_dataset_cleaned_prepared.csv',sep=";")
#%%
#####################
########### Essai 1 
#####################

TOKENIZER = CamembertTokenizer.from_pretrained(
    'camembert-base',
    do_lower_case=True)
MAX_LENGTH=512
encoded_batch = TOKENIZER.batch_encode_plus(df.docs,
                                            add_special_tokens=True,
                                            max_length=MAX_LENGTH,
                                            padding=True,
                                            truncation=True,
                                            return_attention_mask = True,
                                            return_tensors = 'pt')
# %%
pickle.dump(encoded_batch,open(chemin+'encoded_batch.pickle','wb'))
# %%
encoded_batch_=pickle.load(open(chemin+'encoded_batch.pickle','rb'))
print(encoded_batch_.keys())
print(encoded_batch_[list(set(encoded_batch_.keys()))[0]].shape)
encoded_batch_[list(set(encoded_batch_.keys()))[0]]
#%%
print(encoded_batch_.keys())
print(encoded_batch_[list(set(encoded_batch_.keys()))[1]].shape)
encoded_batch_[list(set(encoded_batch_.keys()))[1]]
#%%
print(df.shape)
doc1=df.docs[0]
print("Il y a ",len(doc1),"caractères dans le document.")
doc1_mots=df.docs[0].split(' ')
print("Il y a ",len(doc1_mots)," mots dans le document.")
doc1_phrases=[' '.join(df.docs[0].split(' ')[i:i+10]) for i in range(int(len(df.docs[0].split(' '))/10))]
print("Il y a ",len(doc1_phrases)," phrases dans le document.")
#%%
################################################################################
# On coupe et regroupe toutes les phrases de tous les documents 

documents=' '.join([i for i in df.docs.to_list()])
mots=documents.split(' ')
phrases=[' '.join(mots[i:i+10]) for i in range(int(len(mots)/10))]
len(phrases)
#%%
#On les tokenize
TOKENIZER = CamembertTokenizer.from_pretrained(
    'camembert-base',
    do_lower_case=True)
MAX_LENGTH=512
print('Il y a ',len(phrases)," à encoder.")
import time
start=time.time()
encoded_batch = TOKENIZER.batch_encode_plus(phrases,
                                            add_special_tokens=True,
                                            max_length=MAX_LENGTH,
                                            padding=True,
                                            truncation=True,
                                            return_attention_mask = True,
                                            return_tensors = 'pt')
end=time.time()
print("L'encodage a durée :",round((end-start)/60,2)," minutes.")
pickle.dump(encoded_batch,open(chemin+'phrases_encode.pickle','wb'))
#%%
################################################################################
#On regarde un peu la forme
encoded_batch_=pickle.load(open(chemin+'phrases_encode.pickle','rb'))
print(encoded_batch_.keys())
print(encoded_batch_[list(set(encoded_batch_.keys()))[0]].shape)
encoded_batch_[list(set(encoded_batch_.keys()))[0]]

#%%
print(encoded_batch_.keys())
print(encoded_batch_[list(set(encoded_batch_.keys()))[1]].shape)
encoded_batch_[list(set(encoded_batch_.keys()))[1]]
#%%
split_border = int(encoded_batch_[list(set(encoded_batch_.keys()))[1]].shape[0]*0.8)

################################################################################
#On crée le dataset mais attention : ce n'est pas supervisé !

train_dataset = TensorDataset(
    encoded_batch_['input_ids'][:split_border],
    encoded_batch_['attention_mask'][:split_border])
validation_dataset = TensorDataset(
    encoded_batch_['input_ids'][split_border:],
    encoded_batch_['attention_mask'][split_border:])

batch_size = 32

train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size)
 
validation_dataloader = DataLoader(
            validation_dataset,
            sampler = SequentialSampler(validation_dataset),
            batch_size = batch_size)

#%%
model = CamembertForSequenceClassification.from_pretrained(
    'camembert-base',
    num_labels = 2)
#%%
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # Learning Rate, plus petit pour éviter le 
                            # catastroph forgetting
                  eps = 1e-8) # Epsilon
epochs = 3

# On va stocker nos tensors sur mon cpu : je n'ai pas mieux
device = torch.device("cpu")
 
# Pour enregistrer les stats a chaque epoque
training_stats = []
#%%
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
 
        # Backpropagtions
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
torch.save(model.state_dict(), chemin+"BERT_essai_1.pt")
# %%
#####################
########### Essai 2 
#####################

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('camembert-base')
# %%
text_embeddings = model.encode(df.docs, 
        batch_size = 8,
        show_progress_bar = True)
# %%
similarities = cosine_similarity(text_embeddings)
print('pairwise dense output:\n {}\n'.format(similarities))
# %%
#####################
########### Essai 3 
#####################

import torch
camembert = torch.hub.load('pytorch/fairseq', 'camembert')
camembert.eval()
# %%
#####################
########### Essai 4 
#####################

from pathlib import Path
import pandas as pd
import numpy as np
import sklearn
import pickle
import tensorflow as tf 
import keras
import torch
import gc
import psutil
# from keras.layers import Input,Conv1DTranspose, Conv1D, MaxPooling1D, GlobalAveragePooling1D, UpSampling1D, Dense, Dropout, Activation, Lambda, Reshape, Flatten, Embedding, LSTM, TimeDistributed
# from keras.models import Model,Sequential
# from keras import backend as K
# from keras.callbacks import EarlyStopping
# from keras.regularizers import l2
# from keras.metrics import RootMeanSquaredError
import time
# import seaborn
from sklearn import metrics
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from transformers import CamembertForSequenceClassification, CamembertTokenizer, AdamW
import fastai 
import fast_bert
chemin="C:/Users/theo.roudil-valentin/Documents/Donnees/"
df=pd.read_csv(chemin+'etudes_dataset_cleaned_prepared.csv',sep=";")
train=df[:int(len(df)*0.8)].docs.to_list()
test=df[int(len(df)*0.8):].docs.to_list()
# %%
