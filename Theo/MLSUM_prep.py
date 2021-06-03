#%%
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
from tqdm import tqdm
import os
os.chdir("C:/Users/theo.roudil-valentin/Documents")
from datasets import load_dataset
dataset = load_dataset(
   'mlsum', 'de')
#%%
import json
with open('C:/Users/theo.roudil-valentin/Documents/Donnees/fr_train.jsonl') as f:
   json.load(f)
# %%
