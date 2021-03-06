
# coding: utf-8

# In[98]:



import pandas as pd, numpy as np
import os
# Read recipe inputs

path = "Data\Enjeux\Thesaurus\\"
versions = os.listdir(path)
version = 0
versions[version]


# In[99]:


enjeux = pd.read_csv(open(path+versions[version],encoding = 'utf-8'),delimiter=';')



# In[101]:


final = pd.DataFrame(enjeux['Enjeu environnemental'])
enjeux.replace(np.nan,0)
def itsplit(row,delimiter = ', '):
    try:
        return (row.Dictionnaire.split(delimiter)+row.Correction.split(delimiter)+row.Extension.split(delimiter))
    except:
        try:
            return (row.Dictionnaire.split(delimiter)+row.Correction.split(delimiter))
        except:
            try:
                return (row.Dictionnaire.split(delimiter))
            except:
                print(row)
final['Dictionnaire'] = enjeux.apply(itsplit,axis=1)


# In[102]:


final.Dictionnaire


# In[103]:


import pickle
pickle.dump(final,open(path+'\Thesaurus'+str(version)+'.pickle','wb'))


# %%
