import functools
import operator
import psutil
import gensim
from transformers import CamembertTokenizer,CamembertModel
tok=CamembertTokenizer('MLSUM_tokenizer.model')
camem=CamembertModel.from_pretrained("camembert-base")

W2V=gensim.models.Word2Vec.load("W2V_all.model")

from fats import Word_Cleaning,Resume

Paragraphes=
#introduisez ici une suite de phrases non-nettoyées dans une liste
# Paragraphes = [ phrase1, phrase2, ..., phraseN ]
cpu_max=int(psutil.cpu_count()/6) #On utilise un sixième des cpu de l'ordi

WC=Word_Cleaning(n_jobs=cpu_max,
                sentence=True, #est-ce qu'on coupe ce qu'il y a dans la liste pour en faire des phrases ? Oui
                threshold=True, #On active la sélection de mots suffisamment grand
                seuil=2, #seuil pour la taille des mots
                lemma=False, #est-ce qu'on lemmatise ?
                seuil_carac=3) #nombre de caractères dans la phrase
text=functools.reduce(operator.iconcat,
        WC.remove_empty(
            WC.make_documents(Paragraphes)
            ),
             [])
nphrase=2
R=Resume(type_=, #choisissez le type de résumé que vous voulez
        k=nphrase, #le nombre de phrases
        modele=, #le modèle d'embedding à utiliser, si besoin donc ici camem ou W2V
        tok_path=) # le chemin vers le tokenizer, si besoin : 'MLSUM_tokenizer.model'
resume=R.resume(text)
resume