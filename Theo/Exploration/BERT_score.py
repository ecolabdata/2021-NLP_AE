#%%
import time
import pickle
from pathlib import Path
import gensim
import pandas as pd
import numpy as np
import sklearn
import re
import pickle
from unidecode import unidecode
import functools
import operator
import psutil
from joblib import Parallel,delayed
from functools import partial
import time
import torch
import torch.nn as nn
from tqdm import tqdm
chemin_d="C:/Users/theo.roudil-valentin/Documents/OrangeSum/"
#%%
embedding_section_W2V=pickle.load(open(chemin_d+'embedding_section_W2V.pickle','rb'))
embedding_section_bert=pickle.load(open(chemin_d+'embedding_section_bert.pickle','rb'))
#%%
type(embedding_section_bert),type(embedding_section_bert[0]),type(embedding_section_bert[0][0])
#%%
len(embedding_section_bert),len(embedding_section_bert[0]),len(embedding_section_bert[0][0])
# %%
print("Il y a ",len(embedding_section_bert),"articles\n")
section_mean=round(np.mean([len(i) for i in embedding_section_bert]),2)
print("avec en moyenne",section_mean,"sections")
phrase_mean=
print("avec en moyenne",phrase_mean,"phrases")

# %%
