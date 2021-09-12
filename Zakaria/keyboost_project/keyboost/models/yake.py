import numpy as np
import pandas as pd
import yake
from keyboost.consensus.utils import score_transformation

def yake_extraction(text,
                      language,
                      keyphrases_ngram_max,
                      n_top,
                      stopwords=None,
                      transformation = 'inverse_normalize'):

      yk = yake.KeywordExtractor(lan=language,
                                  n=keyphrases_ngram_max,
                                  top=n_top,
                                  stopwords=stopwords)

      yk_keywords = yk.extract_keywords(text)

      yk_rank = pd.DataFrame(yk_keywords,columns=['Term','Score'])

      yk_rank['Score'] = score_transformation(score=yk_rank['Score'],kind=transformation)


      return  yk_rank
