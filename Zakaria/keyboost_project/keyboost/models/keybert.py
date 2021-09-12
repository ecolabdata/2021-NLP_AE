import numpy as np
import pandas as pd
from keybert import KeyBERT
from sklearn.metrics.pairwise import cosine_similarity
from keybert.backend._utils import select_backend
from keyboost.consensus.utils import score_transformation

def keybert_extraction(text,
                       keyphrases_ngram_max,
                       n_top,
                       stopwords=None,
                       transformers_model='distilbert-base-nli-mean-tokens',
                       transformation='normalize'):

          kb = KeyBERT(transformers_model)

          kb_keywords = kb.extract_keywords(text,
                                            keyphrase_ngram_range = (1,keyphrases_ngram_max),
                                            top_n = n_top,
                                            stop_words=stopwords)

          kb_rank = pd.DataFrame(kb_keywords,columns=['Term','Score'])

          kb_rank['Score'] = score_transformation(score=kb_rank['Score'],kind=transformation)

          return  kb_rank
