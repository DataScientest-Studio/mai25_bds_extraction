# streamlit du projet

Ce dossier permet de générer une application streamlit utilisée lors de la soutenance du projet.
L'arborescence est modulaire afin de faciliter l'édition individuelle des différents fichiers.
L'application s'inspire du style de ce site: https://state-of-llm.streamlit.app

## Arborescence

mai25_bds_extraction/
├── src/
│   └── streamlit/
│       ├── .streamlit/
│       │   └── config.toml         <- éléments de configuration générale (background color, taille textes, ...)
│       ├── assets/
│       │   ├── __init__.py         <- définition de PATHS (pourra servir pour d'autres "variables globales" au besoin)
│       │   ├── style.py            <- définition de variables de style
│       │   └── images/             <- les fichiers images utilisés pour la présentation 
│       │      ├── xxx.jpg             (/!\ /!\ /!\ différent de "raw/..." ). Par exemple: schema des notebooks, ...
│       │      └── ...
│       ├── app.py                  <- point d'entrée de l'applicaiton, menus, headers, ...
│       ├── app_pages/              <- les différentes pages de l'application (menu de niveau 1)
│       │   ├── presentation.py     
│       │   ├── texte.py
│       │   ├── images.py
│       │   ├── multimodal.py
│       │   ├── enseignements.py
│       │   └── demo.py
│       └── README.md               <- vous êtes ici




## Pour lancer le projet:

‘‘‘streamlit run app.py‘‘‘ 


## couleurs extraites du site de référence;

nom,code hexa,usage
noir1,#111111,background page principale
gris1,#31333E,background bandeau navigation+background menus déroulants
gris2,#818494,texte adouci(titre bandeau nav)+icone suppression non actif dans multiselects
gris3,#A4A8B7,
blanc1,#FAFAFA,corps de texte
jaune1,#ECB959,texte mis en évidence
bleu1,#56BDED,lignes de séparation entre les parties
bleu2,#5098F8,texte emphase légère
rose1,#E24768,surbrillance en sélection

graphes (par ordre d'utilisation):
nom,code hexa
bleu3,#93C7FA
bleu4,#2B66C2
rose2,#F3AFAD
rouge,#EA4339
vert1,#9AECA8
vert2,#57AD9D
jaune2,#F8D37A