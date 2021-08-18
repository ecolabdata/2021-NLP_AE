# Code permettant de faire rapidement le DataLoader pour l'entraînement DeepLearning

#%%
from joblib.parallel import cpu_count
import pandas as pd
import numpy as np
import torch
import pickle
import time
import functools
import operator
from joblib import Parallel,delayed
from functools import partial
import sentencepiece as spm 
import psutil
from tqdm import tqdm
import pickle
import os
import re
from unidecode import unidecode
from bs4 import BeautifulSoup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
#%%
num=2
#%%
os.chdir('C:/Users/theo.roudil-valentin/Documents/Resume/MLSUM')

dico_train=pickle.load(open('dico_train_'+str(num)+'.pickle','rb'))

train_input_ids=dico_train['input']
train_mask=dico_train['mask']
clss=dico_train['clss']
train_mask_cls=dico_train['mask_cls']
train_output=dico_train['output']
#clss_index_train=[len(i) for i in dico_train['clss']]

ouais=torch.as_tensor([(train_output[i]!=torch.tensor(0)).nonzero().size()[0] for i in range(len(train_output))])
v=((train_mask_cls.sum(dim=1)>ouais)==True).nonzero()

def correct_mask_cls(input_ids):
    vec=(torch.as_tensor(input_ids)==torch.tensor(5)).nonzero()
    mask=torch.zeros(torch.as_tensor(input_ids).size())
    mask[vec]=1
    return mask

cpu_max=psutil.cpu_count()
train_mask_cls_2=Parallel(cpu_max)(delayed(correct_mask_cls)(train_input_ids[i]) for i in range(len(train_input_ids)))
train_mask_cls_2=torch.stack(train_mask_cls_2)

v=((train_mask_cls_2.sum(dim=1)>ouais)==True).nonzero()

import numpy as np
np.sum([int(train_mask_cls_2[i].sum())==(train_output[i]!=torch.tensor(0)).nonzero().size(0) for i in range(len(train_output))])/len(train_output)

out=torch.zeros(train_mask_cls_2.shape,dtype=torch.float64)

x=(train_output!=torch.tensor(0)).nonzero()
dim_1=torch.unique(torch.stack([x[i][0] for i in range(len(x))]))

x_2=torch.index_select(x,1,torch.tensor(1)).reshape(-1)
x_1=(x_2==0).nonzero()

dim_2=[]
from tqdm import tqdm
for k in tqdm(range(len(x_1))):
    if k<(len(x_1)-1):
        dim_2.append(x_2[x_1[k]:x_1[k+1]])
    else:
        dim_2.append(x_2[x_1[k]:])
    
for k in tqdm(range(len(dim_1))):
    out[k,(train_mask_cls_2[k]!=torch.tensor(0)).nonzero().squeeze(1)]=train_output[dim_1[k],dim_2[k]]
#%%
train_dataset = TensorDataset(
    torch.tensor(train_input_ids),
    torch.tensor(train_mask),
    clss,
    train_mask_cls_2,
    out)

pickle.dump(train_dataset,open('train_dataset_'+str(num)+'.pickle','wb'))

# %%
train_dataset=pickle.load(open('train_dataset_'+str(num)+'.pickle','rb'))
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

K=len(train_dataset)
train_2=TensorDataset(torch.stack([train_dataset[i][0] for i in range(K)]),
                      torch.stack([train_dataset[i][1] for i in range(K)]),
                      torch.stack([train_dataset[i][2] for i in range(K)]),
                      torch.stack([train_dataset[i][3] for i in range(K)]),
                      torch.stack([train_dataset[i][4] for i in range(K)]))

batch_size=int(1024/8/2)
print(batch_size)

dataloader = DataLoader(
            train_2,
            sampler = RandomSampler(train_2),
            batch_size = batch_size)

pickle.dump(dataloader,open('C:/Users/theo.roudil-valentin/Documents/Resume/MLSUM/train_loader_'+str(num)+'.pickle','wb'))

# # %%
# MLSUM=pickle.load(open('C:/Users/theo.roudil-valentin/Documents/Resume/MLSUM/MLSUM_fr_train_1.pickle','rb'))
# MLSUM
# %%
