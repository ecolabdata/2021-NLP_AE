import numpy as np
import pandas as pd
from gensim.summarization import keywords
from keyboost.consensus.utils import score_transformation

def textrank_extraction(text,
                       n_top,
                       transformation='normalize'):


          tr_keywords =  keywords(text,
                                  scores=True,
                                  words=n_top)

          tr_rank = pd.DataFrame(tr_keywords,columns=['Term','Score'])

          tr_rank['Score'] = score_transformation(score=tr_rank['Score'],kind=transformation)

          return  tr_rank
