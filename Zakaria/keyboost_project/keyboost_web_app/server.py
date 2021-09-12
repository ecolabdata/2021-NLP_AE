import streamlit as stl
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from keyBoost.keyBoost import *
from PIL import Image
import spacy

@stl.cache(allow_output_mutation=True,ttl=600)
def load_keyboost():
    return KeyBoost(transformers_model='paraphrase-MiniLM-L6-v2')

@stl.cache(allow_output_mutation=True,ttl=600)
def load_stopwors(language):

    if language == 'en':

        nlp = spacy.load('en_core_web_sm')
        stopwords = nlp.Defaults.stop_words



    elif language == 'fr':

        nlp = spacy.load('fr_core_news_sm')
        stopwords = nlp.Defaults.stop_words

    return stopwords

@stl.cache(allow_output_mutation=True,ttl=600)
def key_extraction(keyboost,
                   language,
                   n_top,
                   stopwords,
                   selected_models,
                   selected_consensus):
    return keyboost.extract_keywords(text=txt,
                           language=language,
                           n_top=n_top,
                           keyphrases_ngram_max=2,
                           stopwords=stopwords,
                           models = selected_models,
                           consensus = selected_consensus)


image = Image.open('keyboost.png')


col1, col2, col3 = stl.columns([6,12,1])

with col1:
    stl.write("")

with col2:
    stl.image(image)

with col3:
    stl.write("")

initial_text = """
         L'apprentissage supervisé est la tâche d'apprentissage automatique consistant à apprendre une fonction qui
          mappe une entrée à une sortie sur la base d'exemples de paires entrée-sortie.[1] Il en déduit un
          fonction à partir de données d'apprentissage étiquetées consistant en un ensemble d'exemples d'apprentissage.[2]
          En apprentissage supervisé, chaque exemple est une paire constituée d'un objet d'entrée
          (généralement un vecteur) et une valeur de sortie souhaitée (également appelée signal de supervision).
          Un algorithme d'apprentissage supervisé analyse les données d'apprentissage et produit une fonction inférée,
          qui peut être utilisé pour mapper de nouveaux exemples. Un scénario optimal permettra à
          algorithme de déterminer correctement les étiquettes de classe pour les instances invisibles. Cela nécessite
          l'algorithme d'apprentissage pour généraliser à partir des données d'entraînement à des situations invisibles dans un
          manière « raisonnable » (voir biais inductif).
      """


keyboost = load_keyboost()

language = stl.selectbox(label='Quelle est la langue du texte ?',
                          options =['fr','en'])



selected_models = stl.multiselect(label="Quels sont les modèles d'extraction sous-jacents que vous souhaitez utiliser ?",
                          default=['yake','textrank','keybert'],
                          options =['yake','textrank','keybert'])


selected_consensus = stl.selectbox(label='Quel type de consensus voulez-vous effectuer ?',
                          options =['statistical','rank'])


txt = stl.text_area(label='Texte à analyser',
                   value=initial_text,
                   height=350)

n_top = stl.slider(label='Combien de mots-clés au maximum voulez-vous extraire ?',
                        min_value=1,max_value=10,value=5)

stopwords =  load_stopwors(language)



with stl.spinner(text='Veuillez patienter...'):
    if txt == '' :
        stl.info('Veuillez saisir du texte dans la zone dédiée')
    elif len(selected_models)==0 :
        stl.info('Veuillez selectionner au moins un sous-modèle')



    else:

        try:

            keywords = key_extraction(keyboost,
                               language,
                               n_top,
                               stopwords,
                               selected_models,
                               selected_consensus)


            if 'textrank' in selected_models:
               keywords = [k for k in keywords if k not in stopwords]

            css = '''text-transform: lowercase;
            	background: linear-gradient(to right, #acb4fc 0%, #6fd4fc 100%);
            	-webkit-text-fill-color: white;
            	display: inline-block;
                padding: 3px 3px;
                margin: 5px 5px;'''

            # maybe adding the score feature later
            # if keyboost.is_statistical_consensus_completed:
            #     mkds = ''
            #
            #
            #
            #
            #     css_confidence = '''text-transform: lowercase;
            #     	background: black
            #     	-webkit-text-fill-color: white;
            #     	display: inline-block;
            #         padding: 3px 3px;
            #         margin: 5px 5px;'''
            #
            #     max_score = keyboost.statistical_consensus_scores['Score'].max()
            #
            #     for k,s in keyboost.statistical_consensus_scores.values:
            #         print(k,s)
            #         mkds+='''<p style='{}'>{} (score:{})</p>'''.format(css,k,round(s/max_score*100,2))
            #     stl.markdown(mkds,unsafe_allow_html=True)
            # else:

            mkd = ''
            for k in keywords:
                mkd+='''<p style='{}'>{}</p>'''.format(css,k)

            stl.markdown(mkd,unsafe_allow_html=True)

        except Exception as e:

            stl.info('Veuillez réessayer')
