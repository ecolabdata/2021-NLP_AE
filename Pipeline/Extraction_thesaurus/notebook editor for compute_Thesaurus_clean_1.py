
# coding: utf-8

# In[98]:



import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
thesaurus = dataiku.Folder("el0cX6TX")
thesaurus_info = thesaurus.get_info()

path = thesaurus.get_path()
versions = thesaurus.list_paths_in_partition()
version = 1
versions[version]


# In[99]:


enjeux = pd.read_csv(open(path+versions[version]),encoding = 'utf-8',delimiter=';')


# In[100]:


type(enjeux.Extension[0])


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


final.Dictionnaire[0]


# In[103]:


import pickle
path_clean = dataiku.Folder('ywCWAV6b').get_path()
pickle.dump(enjeux,open(path_clean+'/Thesaurus'+str(version)+'.pickle','wb'))

