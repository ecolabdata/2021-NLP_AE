#%%
from PyPDF2 import PdfFileReader
import os
import numpy as np
from numpy.core.defchararray import title
os.chdir('C:\\Users\\theo.roudil-valentin\\Documents\\Donnees\\PDF_EI')
fichiers=os.listdir()
fichiers=[i for i in fichiers if i[-3:]=='pdf']
# fichiers
print("Nombre de pdf :",len(np.unique(fichiers)))
# %%
from joblib import Parallel,delayed

def count_page(f):
    with open(f, "rb") as pdf_file:
        try:
            pdf_reader = PdfFileReader(pdf_file)
            return pdf_reader.numPages
        except:
            return np.nan

count_page(fichiers[0])
#%%
import psutil
cpu=psutil.cpu_count()
num_pages=Parallel(n_jobs=cpu)(delayed(count_page)(f) for f in fichiers)
# %%
import matplotlib.pyplot as plt

fig,ax=plt.subplots(figsize=(18,12))
ax.hist(num_pages)
ax.set_title('Distribution du nombre de pages par documents')
# %%
print("Nombre de pages ")
print("Moyenne :",round(np.nanmean(num_pages),2))
print("Ecart-type :",round(np.nanstd(num_pages),2))
print("Médiane :",round(np.nanmedian(num_pages),2))
# %%
# 