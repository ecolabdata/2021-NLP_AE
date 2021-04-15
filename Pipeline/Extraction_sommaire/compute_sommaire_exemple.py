# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
base_classif_RF = dataiku.Dataset("base_classif_RF")
base = base_classif_RF.get_dataframe(sampling="head",limit=10000)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from bs4 import BeautifulSoup
base['phrase_2']=[BeautifulSoup(i,"html.parser").text for i in base.phrase]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
base.num_etude[0]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for i in base[base.label_RF==0][base.num_etude==base.num_etude[0]].phrase_2.values:
    print(i)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# On remarque que beaucoup de choses ne sont pas des titres. Mais est-ce vraiment le cas ?
# 
# En fait l'algorithme semble classer également les premiers paragraphes des titres (puisque le titre est associé au premier paragraphe semble-t-il). Du coup, on a bien le sommaire au début, puis ensuite directement des bouts de sections avec le titre.
# 
# Pourquoi l'algo classifie-t-il les paragraphes avec les titres comme des titres ? Pour la simple et bonne raison que le titre présent apporte l'information pour être classifié comme titre.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
a=40
[i for i in base[base.label_RF==0][base.num_etude==base.num_etude[0]].phrase_2.values if len(i)<a]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

sommaire_exemple_df =  # For this sample code, simply copy input to output


# Write recipe outputs
sommaire_exemple = dataiku.Dataset("sommaire_exemple")
sommaire_exemple.write_with_schema(sommaire_exemple_df)