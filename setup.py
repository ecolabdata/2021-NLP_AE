#%%
import os
import win32api
import shutil
import tqdm

fini = False

while fini == False:

    #On s'assure qu'on est a la bonne racine dans la console
    CURR_DIR = os.getcwd().split('\\')[-1]
    if CURR_DIR != '2021-NLP_AE':
        print("ERREUR : La racine courante n'est pas 2021-NLP_AE")
        break
    print('\n')
    print('Création des dossiers locaux...')
    #On crée les dossiers si ils ne sont pas la
    CURR_DIR = os.getcwd()
    dirs = os.listdir(CURR_DIR)
    if 'Data' not in dirs:
        os.mkdir('Data')
    CURR_DIR += '\Data\\'
    final_paths = ['Avis_txt','Bagging_model','Etude_html_csv','Thesaurus_csv','Workinprogress']
    for p in final_paths:
        try:
            os.mkdir(CURR_DIR+p)
        except:
            print(p,'existe déjà')

    print('\n')
    print('Téléchargement des fichiers...')
    #On compare les fichiers contenus dans Data et on copie si il y'a des différences
    drives = win32api.GetLogicalDriveStrings().split('\000')[:-1]
    if "K:\\" not in drives:
        print("ERREUR : L'ordinateur n'est pas connecté au disque partagé du SRI sous le nom K")
        break
    datapath = "K:\\03 ECOLAB\\2 - POC DATA et IA\Données NLP\Data\\"
    files = os.listdir(datapath)
    for file in files:
        dataDistant = os.listdir(datapath+file)
        dataLocal = os.listdir(CURR_DIR+file)
        if dataDistant ==  dataLocal:
            print(file,"déjà téléchargé")
        else:
            dataDistant = set(dataDistant)
            dataLocal = set(dataLocal)
            dataRestant = list(dataDistant-dataLocal)
            print("Downloading ",file)
            for data in tqdm.tqdm(dataRestant):
                shutil.copyfile(datapath+file+'\\'+data,CURR_DIR+file+'\\'+data)

    fini = True
   

# %%
