import numpy as np
import pandas as pd
from keyboost.models.yake import yake_extraction
from keyboost.models.keybert import keybert_extraction
from keyboost.models.textrank import textrank_extraction
from keyboost.consensus.statistical import *
from keyboost.consensus.ranking import rank_consensus

class KeyBoost:

    def __init__(self,
    transformers_model
    ):
        self.transformers_model = transformers_model
        self.statistical_consensus_scores = None
        self.is_statistical_consensus_completed = None


    def extract_keywords(self,
                        text,
                        language,
                        n_top,
                        keyphrases_ngram_max,
                        consensus,
                        models,
                        stopwords=None):




        key_extractions = list()

        stat_sample = 100
        # YAKE extraction

        if 'yake' in models:
            yk_rank = yake_extraction(text=text,
                                language=language,
                                keyphrases_ngram_max=keyphrases_ngram_max,
                                n_top=stat_sample,
                                stopwords=stopwords)
            key_extractions.append(yk_rank)

        # KeyBERT extraction
        if 'keybert' in models:

            kb_rank = keybert_extraction(text=text,
                                  keyphrases_ngram_max=keyphrases_ngram_max,
                                  n_top=stat_sample,
                                  stopwords=stopwords,
                                  transformers_model=self.transformers_model)
            key_extractions.append(kb_rank)

        if 'textrank' in models:

            tr_rank =textrank_extraction(text=text, n_top=stat_sample)
            key_extractions.append(tr_rank)

        # Extract scores

        if  consensus   == 'statistical':

            result = statistical_consensus(key_extractions=key_extractions,
                                  n_top=n_top)

            if type(result) == list:
                 keywords = result
                 self.is_statistical_consensus_completed = False
            else:
                keywords = result['Keyword']

                self.statistical_consensus_scores = result
                self.is_statistical_consensus_completed = True


        elif consensus == 'rank':

            keywords =  rank_consensus(key_extractions=key_extractions,n_top=n_top)


        return keywords
