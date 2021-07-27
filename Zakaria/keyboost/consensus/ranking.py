import numpy as np
import pandas as pd


def rank_consensus(key_extractions,n_top):

    n = round(n_top/len(key_extractions))
    keywords = list()

    for key_extraction in key_extractions:
        for i in range(n):
            keywords.append(key_extraction.values[i][0])

    return  keywords[:n_top]
