#%%
import torch.nn as nn
import sklearn
import torch
import pickle
import os
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import torch.optim as optim

#%%

#Contrairement a la métrique précédente, calcule un score F1 sur tous les enjeux a la fois
#On ne calcule pas le F1 de chaque enjeu séparément
class F1_loss:
    """
    Class for f1 calculation in Pytorch.
    x : prediction
    y : labels
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
        tpm=torch.div(tp,y.shape[0])
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
        fpm=torch.div(fp,y.shape[0])
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
        fnm=torch.div(fn,y.shape[0])
        return fnm
    #@staticmethod
    def precision(self,x,y) -> torch.tensor:
        device=y.device
        tp=self.true_positive_mean(x,y)
        fp=self.false_positive_mean(x,y)
        if (tp+fp)!=0:
            prec=torch.div(tp,(tp+fp))
            return prec
        else:
            return torch.tensor(0.).to(device)

    def recall(self,x,y) -> torch.tensor:
        tp=self.true_positive_mean(x,y)
        fn=self.false_negative_mean(x,y)
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
            return torch.tensor(1-f1.data,requires_grad=True)#prec,rec,
        else:
            return torch.tensor(1.,requires_grad=True).to(device)#prec,rec,

#%%

class Hamming_loss:
    """
    Class for f1 calculation in Pytorch.
    x : prediction
    y : labels
    """
    def __init__(self):#, average: str = 'weighted'):
        """
        Init.

        Args:
            average: averaging method
        """
        
    def __call__(self,x,y) -> torch.tensor:
            device=y.device
            rec=self.recall(x,y)
            prec=self.precision(x,y)
            f1=torch.mul(rec,prec)
            f1=torch.mul(2,f1)
            f1=torch.div(f1,prec+rec)
            if (prec+rec)!=0:
                return torch.tensor(1-f1.data,requires_grad=True)#prec,rec,
            else:
                return torch.tensor(1.,requires_grad=True).to(device)#prec,rec,

#%%
a = [1,2,3]
for k,b in enumerate(a):
    print(k,b)
# %%
