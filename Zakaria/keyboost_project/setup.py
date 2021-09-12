from setuptools import setup, find_packages

setup(
   name='keyboost',
   version='0.1',
   description='KeyBoost is simple and easy-to-use keyword extraction tool that moves away the hassle of selecting the best models for your specific use-case.',
   author='Zakaria Bekkar',
   author_email='zakaria.bekkar@ens-paris-saclay.fr',
   packages=find_packages("."),  #same as name
   install_requires=['wheel',
                    'torch',
                    'torchvision',
                    'statsmodels',
                    'sklearn',
                    'scipy',
                    'pandas',
                    'numpy',
                    'spacy',
                    'yake',
                    'keybert',
                    'gensim==3.8.3',
                    'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz',
                    'fr_core_news_sm @ https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.0.0/fr_core_news_sm-3.0.0.tar.gz',
                    ], #external packages as dependencies
                    )
