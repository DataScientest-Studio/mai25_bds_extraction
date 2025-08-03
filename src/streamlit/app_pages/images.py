import streamlit as st
import os 
import pandas as pd
from assets import style
from assets import PATHS

sections = ["Machine Learning",
            "Deep Learning"]

section_counter = 0
def next_section():
    global section_counter
    st.markdown(f'<a name="{sections[section_counter]}"></a>', unsafe_allow_html=True)
    st.markdown(f"<hr style='{style.divider}'>",unsafe_allow_html=True)
    # st.markdown("""<div style="padding-top: 240px; margin-top: -240px;"></div>""",unsafe_allow_html=True)
    st.header(sections[section_counter])
    section_counter += 1


def show():
    st.title("Classification des Images scannées")

    next_section()
    st.subheader("Préparation des données pour le Machine Learning")
    st.markdown(f"""Le machine learning s'appuie sur: 
- les caractéristiques extraites des images, normalisées (après séparation en sets de train, test et validation)
- les valeurs des pixels des images.\n
Cependant, chaque vecteur représentant une image doit avoir le même nombre de colonnes:
- les marges des documents scannés a été retirées pour se concentrer sur le contenu
- les images ont été redimensionnées en 100x100
- pour éviter la déformation, des zones blanches ont parfois été rajoutées.\n
Voici le processus en image :
    """)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.image(os.path.join(PATHS.streamlit, "assets", "images","img_init1.png"), width=300, caption="Image initiale")
    with col2:
        st.image(os.path.join(PATHS.streamlit, "assets", "images","img_cropped1.png"), width=300, caption="Image rognée")
    with col3:
        st.image(os.path.join(PATHS.streamlit, "assets", "images","img_canvas1.png"), width=300, caption="Image redimensionnée sans déformation")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.image(os.path.join(PATHS.streamlit, "assets", "images","img_init2.png"), width=300, caption="Image initiale")
    with col2:
        st.image(os.path.join(PATHS.streamlit, "assets", "images","img_cropped2.png"), width=300, caption="Image rognée")
    with col3:
        st.image(os.path.join(PATHS.streamlit, "assets", "images","img_canvas2.png"), width=300, caption="Image redimensionnée sans déformation")

    st.subheader("Sélection du meilleur modèle et des meilleurs paramètres")
    st.markdown(f""" Les étapes: 
    - Echantillon de 10 000 images (sur 400 000)
    - PCA pour réduire le nombre de dimensions: on passe de 10 012 à 2000
    - Lazy Classifier pour tester, avec les paramètres par défaut, un grand nombre de classifiers
    - Grid Search sur 3 classifiers
    - LGBM : le meilleur classifier pour nos images
                """)
    
    st.image(os.path.join(PATHS.streamlit, "assets", "images","PCA.png"), caption="PCA sur un echantillon de 10 000 images. 2000 dimensions sont conservées")
    st.markdown(f""" Resultats du Lazy Classifier:""")
    data = pd.DataFrame({
        'Model': ['LGBMClassifier', 'XGBClassifier', 'RandomForestClassifier', 'BaggingClassifier', 'LinearDiscriminantAnalysis'],
        'Accuracy': [0.35, 0.35, 0.27, 0.27, 0.25],
        'F1 Score': [0.34, 0.34, 0.35, 0.25, 0.26]
    })
    st.table(data)
    
    st.markdown(f"""Pour la suite, ont été retenus:
- LGBMClassifier
- XGBClassifier
- SGDClassifier (non testé par le lazy classifier mais recommandé dans le cours dans le cas de gros échantillons)""")
    
    st.subheader("Résultats du machine learning sur notre échantillon de 10 000 images")
    
    with st.expander("Resultats du SGD Classifier"):
        st.markdown(f"""Avec les meilleurs paramètres (Grid Search): Accuracy 0,17""")
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","sgd_spider.png"), caption="Représentation de l'accuracy du XGBoost")
        with col2:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","sgd_cm.png"), caption="Matrice de confusion pour le XGBoost")

    with st.expander("Resultats du XGBoost Classifier"):
        st.markdown(f"""Avec les meilleurs paramètres (Grid Search): Accuracy 0,37""")
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","xgb_spider.png"), caption="Représentation de l'accuracy du XGBoost")
        with col2:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","xgb_cm.png"), caption="Matrice de confusion pour le XGBoost")

    with st.expander("Resultats du LGBM Classifier"):
        st.markdown(f"""Avec les meilleurs paramètres (Grid Search): Accuracy 0,42""")
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","lgbm_spider.png"), caption="Représentation de l'accuracy du LGBM")

        with col2:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","lgbm_cm.png"), caption="Matrice de confusion pour le LGBM")

    
    st.subheader("LGBM lancé sur le dataset entier")
    st.markdown(f"""Pour le LGBM, avec les meilleurs paramètres, cette fois :
- entrainé sur 320 000 images 
- testé sur 4 000 images\n
Accuracy 0,55""")
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.image(os.path.join(PATHS.streamlit, "assets", "images","lgbm_spider_all_images.png"), caption="Représentation de l'accuracy du LGBM")

    with col2:
        st.image(os.path.join(PATHS.streamlit, "assets", "images","lgbm_cm_all_images.png"), caption="Matrice de confusion pour le LGBM")

    
        

    st.subheader("Conclusions sur le machine learning appliqué aux images")
    data2 = pd.DataFrame({
        'Model': ['LGBMClassifier', 'XGBClassifier', 'SGDCLassifier'],
        'Durée de Bayes Search (min)': [72, 323, 3],
        'Accuracy sur le set de valisation': [0.42, 0.37, 0.17]
    })
    st.table(data2)
    
    st.markdown(f""" Le LGBM et XGBoost ont des performances similaires mais le LGBM est bien plus rapide, c’est donc lui qu’il faudrait utiliser pour travailler sur l’ensemble des images. 
Cependant, il n’est pas pertinent de continuer à utiliser des algorithmes de machine learning simples pour classifier nos images car cette approche n'est pas réellement adaptée: 
- Perte du contexte spatial : nos algorithmes traitent chaque pixel de manière indépendante, sans tenir compte des relations spatiales entre les pixels voisins, ce qui est essentiel pour comprendre des structures visuelles.
- Trop de dimensions : nos images contiennent un grand nombre de pixels. Il a donc fallu perdre en résolution en changeant leur taille, et effectuer une PCA pour être capable de travailler dessus. Cela entraîne une grande perte d’information.
""")
    


    next_section()

    st.subheader("Préparation des images pour le DeepLearning")
    st.markdown(f"""Deux réseaux de neurones classiques ont été étudiés pour cette partie de DeepLearning: VGG16 et ResNet50. Dans les deux cas, sont attendues des images au format JPEG avec 3 canaux de couleur, nos images ont donc été converties sous ce nouveau format.\n
Aussi, aucune augmentation n'a été effectuée:
- D'une part les images scannées, représentant du texte ne s'y prêtent pas: pas de rotation aléatoire, ni de symértries
- D'autre part, les quelques augmentations testées (zoom et légères translations) ont fortement fait chuter les performances des modèles.
""")
    
    st.markdown(f"""Les premiers modèles de deepLearning ont été enregistrés... mais pas leur history, et il aurait été trop long de les relancer juste pour les avoir sur fond noir et frustrant de ne pas en profiter pour les améliorer... Donc voici ce qui a été obtenu.""")



    st.subheader("Résultats ResNet50")
    with st.expander("ResNet50 - 10 couches de fine-tuning"):
        st.markdown(f"""Le premier ResNet avec des résultats intéressants a subi un peu de fine-tuning: les 10 dernières couches de convolution ont été dégelées\n
Pour limiter le temps d'execution, l'ensemble a été fait sur un échantillon de 40 000 images (1/10 de l'ensemble des images)
L'accuracy finale obtenue sur le set de validation de 4000 images: 0,74
    """)
        st.image(os.path.join(PATHS.streamlit, "assets", "images","ResNet_1_history.png"), caption="Evaluation de la fonction de perte et de l'accuracy au fil des epoch")
        st.markdown(f"""50 epochs avaient été programmés mais l'apprentissage s'est arrêté à 17 pour absence d'amélioration""")

        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","ResNet_1_spider.png"), caption="Représentation de l'accuracy - ResNet50 - 10 couches")

        with col2:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","ResNet_1_cm.png"), caption="Matrice de confusion - ResNet50 - 10 couches")

    with st.expander("ResNet50 - Dégel progressif manuel des couches"):
        st.markdown(f"""Cette fois, pour essayer d’apprendre mieux les particularités de nos images:
- Etape 1:  on commence sans aucune couche dégelée,  puis on lance 10 epochs (en réalité, l'entraînement s’est arrêté à 7 car il n’apprenait plus)
- Etape 2: on garde gelées 140 couches (sur 177) donc 37 couches sont dégelées et on relance 10 epochs
- Etape3: on garde 120 couches gelées, donc 57 couches dégelées au total et on relance 10 epochs

Les “history” des 3 entrainements ont été concaténées et voici leur résultat: """)

        
        
        st.image(os.path.join(PATHS.streamlit, "assets", "images","ResNet_2_history.png"), caption="Evaluation de la fonction de perte et de l'accuracy au fil des epoch")
        st.markdown(f"""Là encore, pour limiter le temps d'execution, l'ensemble a été fait sur un échantillon de 40 000 images (1/10 de l'ensemble des images)
L'accuracy finale obtenue sur le set de validation de 4000 images: 0,78
    """)
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","ResNet_2_spider.png"), caption="Représentation de l'accuracy - ResNet50 - 10 couches")

        with col2:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","ResNet_2_cm.png"), caption="Matrice de confusion - ResNet50 - 10 couches")

        st.markdown(f"""Le dégel manuel des couches rend l'apprentissage un peu chaotique, avec des variations fortes lors du dégel de nouvelles couches. Il faudrait rendre ce processus automatique (ça a été fait pour VGG16 qui donne de meilleurs résultats)
    """)
        
    st.subheader("Résultats VGG16")
    with st.expander("VGG16 - 4 couches de fine-tuning"):
        st.markdown(f"""Le premier VGG16 avec des résultats intéressants a subi un peu de fine-tuning: les 4 dernières couches de convolution ont été dégelées\n
Pour limiter le temps d'execution, l'ensemble a été fait sur un échantillon de 40 000 images (1/10 de l'ensemble des images)
L'accuracy finale obtenue sur le set de validation de 4000 images: 0,79
    """)
        st.image(os.path.join(PATHS.streamlit, "assets", "images","VGG16_1_history.png"), caption="Evaluation de la fonction de perte et de l'accuracy au fil des epoch")
        st.markdown(f"""50 epochs avaient été programmés mais l'apprentissage s'est arrêté à 25 pour absence d'amélioration""")

        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","vgg16_1_spider.png"), caption="Représentation de l'accuracy - VGG16 - 4 couches")

        with col2:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","vgg16_1_cm.png"), caption="Matrice de confusion - VGG16 - 4 couches")

    with st.expander("VGG16 -  Degel progressif automatique des couches"):
        st.markdown(f"""Cette fois, un nouveau callback a été ajouté, qui dégèle 5 nouvelles couches à fine-tuner quand la fonction de perte ,n'est pas meilleure que la meilleure précédemment enregistrée pendant 5 epochs (et on part d'un VGG16 avec 4 couches dégelées dès le départ)\n
Pour limiter le temps d'execution, l'ensemble a été fait sur un échantillon de 40 000 images (1/10 de l'ensemble des images)
L'accuracy finale obtenue sur le set de validation de 4000 images: 0.81
    """)
        st.image(os.path.join(PATHS.streamlit, "assets", "images","VGG16_2_history.png"), caption="Evaluation de la fonction de perte et de l'accuracy au fil des epoch")
        st.markdown(f"""50 epochs avaient été programmés mais l'apprentissage s'est arrêté à """)

        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","vgg16_2_spider.png"), caption="Représentation de l'accuracy - VGG16 - 4 couches")

        with col2:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","vgg16_2_cm.png"), caption="Matrice de confusion - VGG16 - 4 couches")


    st.subheader("Conclusion sur le DeepLearning")
    data3 = pd.DataFrame({
    'Model': ['ResNet50 - 10 couches', 'ResNet50 - Degel progressif manuel', 'VGG16 - 4 couches', 'VGG16 - Degel progressif automatique'],
    "Epochs avant arrêt": ["18/50", "Etape 1: 9/10 <br>Etape 2: 6/10<br>Etape 3: 6/10", "25/50", "29/50"],
    "Durée d'entrainement": ["2h45", "Etape 1: 1h25 <br>Etape 2: 1h11<br>Etape 3: 1h20", "9h", "18h"],
    'Accuracy sur le set de validation': [0.74, 0.78, 0.79, 0.81]
    })

    # Création du tableau HTML
    table_html = data3.to_html(escape=False, index=False) #pour que les passages à la ligne fonctionnent

    # Affichage dans Streamlit
    st.markdown(table_html, unsafe_allow_html=True)
    