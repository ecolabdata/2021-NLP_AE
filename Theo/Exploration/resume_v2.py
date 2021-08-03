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

from fats import BERTScore,TextRank,Random_summary,Lead_3,SMHA_Linear_classifier
from fats import Make_Extractive,Word_Cleaning