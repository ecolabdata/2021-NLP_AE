# %%
import pickle
from Pipeline.Enjeux.topicmodeling_pipe import CorExBoosted

#On charge notre texte et notre thésaurus et on nettoie un peu/formate les données et colonnes
docs_df = pickle.load(open("Data\Bagging_model\df_sections.pickle",'rb'))
docs_df.dropna(inplace = True)
docs_df.rename(columns = {'num_etude':'id_AAE'},inplace = True)
Thesaurus = pickle.load(open("Data\Thesaurus_csv\Thesaurus1.pickle",'rb'))
#%%
from Pipeline.Enjeux.processing_encoding import processing_thesaurus,processing

#Le préprocessing permet de lemmatiser les mots du thésaurus de la même manière que les mots du texte vont l'être (sinon ils ne seront pas reconnus)
Thesaurus = processing_thesaurus(Thesaurus)

#Ces études sont celles les mieux océrisées et dont le sommaire et les sections sont les mieux extraites
correct = [100689,100707,102316,106168,110277,114799,118071,120638]
docs_df_correct = docs_df[docs_df.id_AAE.isin(correct)]
docs_df_correct = docs_df_correct[docs_df_correct.section_clean_1.str.len()>50]
docs_df_correct = docs_df_correct[docs_df_correct.section_clean_1.str.len()<20000]

#Petite visualisation de la longueur des études
lenEtude =docs_df_correct.section_clean_1.str.len().tolist()
lenEtude.sort()
import seaborn as sns
sns.histplot(lenEtude)


#On initialise la classe CorExBoosted sur les documents et le thésaurus
instance = CorExBoosted(docs_df_correct,Thesaurus)

#Préprocessing du texte. 
instance.preprocess('section_clean_1')

#On dispose d'outils de diagnostic du vocabulaire du thésaurus si nécessaire pour visualiser la couverture du vocabulaire (nombre de mots du dictionnaire réellement présents dans le vocabulaire
# du vectoriseur)
diagnostics = instance.diagnostic()

#On encode avec une instance CountVectoriser. Les paramètres du vectoriseur sont préréglés mais il est possible de les modifier.
instance.encode()

#On peux accéder a des informations de diagnostic ici encore grace a la méthode encore qui génère des attributs, respectivement:
#Le mapping word-id, la fréquence d'apparition des mots, le vocabulaire trié, les mots du thésaurus qui ne sont pas dans le vocabulaire...
instance.word2id,instance.words_freq,instance.vocab_sort,instance.notinvoc

#On fit les classifieurs. On peux en mettre plus ou moins. La stratification et l'augmentation ne sont possible que si on dispose de données labellisées !
instance.fit(n_classif=100,stratify=False,augment=False)

pickle.dump(instance,open('Data/enjeux_section.pickle','wb'))
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
