#%%
import pandas as pd
import numpy as np
import pickle
import time

chemin_donnees="C:/Users/theo.roudil-valentin/Documents/Donnees/EI_txt/"
chemin_modele="C:/Users/theo.roudil-valentin/Documents/Donnees/Modele_Transformer/"

# df=pd.read_csv(chemin+'etudes_dataset_cleaned_prepared.csv',sep=";")

import fairseq
from fairseq.models.transformer import TransformerEncoderLayer
import torch
import tensorflow as tf
import sklearn
# %%

# %%
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path(chemin_donnees).glob("**/*.txt")]
#%%
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()
start=time.time()
# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
end=time.time()
print("L'entraînement a pris :",round((end-start)/60,2),' minutes.')
# Save files to disk
tokenizer.save_model(chemin_modele, "Token_BERT_MTE")
# %%
print(
    tokenizer.encode("environnement"),tokenizer.encode("environnement")
)
# %%
tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
tokenizer.enable_truncation(max_length=512)
#%%
examples = []
for src_file in Path(chemin_donnees).glob("**/*.txt"):
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            examples += [x.ids for x in tokenizer.encode_batch(lines)]

# %%
