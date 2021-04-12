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

from transformers import AdamW
optimizer = AdamW(camembert.parameters(), lr=1e-5)

from transformers import CamembertTokenizer
tokenizer=CamembertTokenizer.from_pretrained(camembert_path)

#%%
text_batch = []
for src_file in Path(chemin_donnees).glob("**/*.txt"):
            print("🔥", src_file)
            text_batch +=  src_file.read_text(encoding="utf-8").splitlines()
            
#%%
encoding = tokenizer(text_batch, return_tensors='pt', padding='max_length', truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
#%%
labels = torch.tensor([1,0]).unsqueeze(0)
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
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