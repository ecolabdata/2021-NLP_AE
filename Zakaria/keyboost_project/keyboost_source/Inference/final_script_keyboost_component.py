import spacy
from keyboost.keyBoost import *

if len(section.split(' ')) <=1000:


    nlp = spacy.load('fr_core_news_sm')

    stopwords = nlp.Defaults.stop_words


    keyboost = KeyBoost('paraphrase-MiniLM-L6-v2')


    keywords = keyboost.extract_keywords(text=section,
                           language='fr',
                           n_top=10,
                           keyphrases_ngram_max=2,
                           stopwords=stopwords,
                           consensus='statistical',
                           models=['keybert','yake','textrank'])
