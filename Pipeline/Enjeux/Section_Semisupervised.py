# %%
import pickle
from Pipeline.Enjeux.bagging import CorExBoosted
docs_df = pickle.load(open('Data/Workinprogress/df_section_clean.pickle','rb'))
docs_df.rename(columns = {'num_etude':'id_AAE'},inplace = True)
Thesaurus = pickle.load(open('Data\Thesaurus_csv\Thesaurus1_clean.pickle','rb'))
instance = CorExBoosted(docs_df,Thesaurus)
# %%
instance.encode()
#%%

instance.fit(n_classif=100,stratify=False)
# %%
import pandas as pd

preds = instance.predict(instance.X)
g = pd.DataFrame(preds,columns=instance.enjeux_list)
d = docs_df.drop('section_html',axis = 1)
d = d.reset_index()
# %%


final = pd.concat([d,g],axis=1)
final.drop(['level_0','index'],inplace = True, axis =1)
# %%


final.to_csv('Data\Workinprogress\section_test.csv')
# %%
