#%%
import spacy
from keyboost.keyBoost import *

def extract_keywords_doc(Paragraphes,maxlen = 1000,language='fr',
                                n_top=10,
                                keyphrases_ngram_max=2,
                                consensus='statistical',
                                models=['keybert','yake','textrank']):
    keywords_paragraphe = []
    for section in Paragraphes:
        if len(section.split(' ')) <=maxlen:
            nlp = spacy.load('fr_core_news_sm')
            stopwords = nlp.Defaults.stop_words
            keyboost = KeyBoost('paraphrase-MiniLM-L6-v2')
            keywords = keyboost.extract_keywords(text=section,
                                language=language,
                                n_top=n_top,
                                keyphrases_ngram_max=keyphrases_ngram_max,
                                stopwords=stopwords,
                                consensus=consensus,
                                models=models)
            keywords_paragraphe.append(keywords)
        else:
            keywords_paragraphe.append('trop court')
# %%
