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
    - PCA pour réduire le nombre de dimensions: on passe de 100 012 à 2000
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
        st.markdown(f"""Avec les meilleurs paramètres (Grid Search): F1 Score 0,17""")
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","sgd_spider.png"), caption="Représentation de l'accuracy du XGBoost")
        with col2:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","sgd_cm.png"), caption="Matrice de confusion pour le XGBoost")

    with st.expander("Resultats du XGBoost Classifier"):
        st.markdown(f"""Avec les meilleurs paramètres (Grid Search): F1 Score 0,37""")
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","xgb_spider.png"), caption="Représentation de l'accuracy du XGBoost")
        with col2:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","xgb_cm.png"), caption="Matrice de confusion pour le XGBoost")

    with st.expander("Resultats du LGBM Classifier"):
        st.markdown(f"""Avec les meilleurs paramètres (Grid Search): F1 Score 0,42""")
        col1, col2 = st.columns([0.4, 0.6])
        with col1:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","lgbm_spider.png"), caption="Représentation de l'accuracy du LGBM")

        with col2:
            st.image(os.path.join(PATHS.streamlit, "assets", "images","lgbm_cm.png"), caption="Matrice de confusion pour le LGBM")

    
        
        

    st.subheader("Conclusions sur le machine learning appliqué aux images")
    data2 = pd.DataFrame({
        'Model': ['LGBMClassifier', 'XGBClassifier', 'SGDCLassifier'],
        'Durée de Bayes Search (min)': [72, 323, 3],
        'F1 Score sur le set de valisation': [0.42, 0.37, 0.17]
    })
    st.table(data2)
    
    st.markdown(f""" Le LGBM et XGBoost ont des performances similaires mais le LGBM est bien plus rapide, c’est donc lui qu’il faudrait utiliser pour travailler sur l’ensemble des images. 
Cependant, il n’est pas pertinent de continuer à utiliser des algorithmes de machine learning simples pour classifier nos images car cette approche n'est pas réellement adaptée: 
- Perte du contexte spatial : nos algorithmes traitent chaque pixel de manière indépendante, sans tenir compte des relations spatiales entre les pixels voisins, ce qui est essentiel pour comprendre des structures visuelles.
- Trop de dimensions : nos images contiennent un grand nombre de pixels. Il a donc fallu perdre en résolution en changeant leur taille, et effectuer une PCA pour être capable de travailler dessus. Cela entraîne une grande perte d’information.
""")
    
    st.markdown(f"""Pour le LGBM, avec les meilleurs paramètres, cette fois relancé sur l'ensemble des 400 000 images\nF1 Score 0,55""")
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.image(os.path.join(PATHS.streamlit, "assets", "images","lgbm_spider_all_images.png"), caption="Représentation de l'accuracy du LGBM")

    with col2:
        st.image(os.path.join(PATHS.streamlit, "assets", "images","lgbm_cm_all_images.png"), caption="Matrice de confusion pour le LGBM")

    

    next_section()

    st.subheader("Préparation des images pour le DeepLearning")
    st.markdown(f"""Deux réseaux de neurones classiques ont été étudiés pour cette partie de DeepLearning: VGG16 et ResNet50. Dans les deux cas, sont attendues des images au format JPEG avec 3 canaux de couleur, nos images ont donc été converties sous ce nouveau format.\n
Aussi, aucune augmentation n'a été effectuée:
- D'une part les images scannées, représentant du texte ne s'y prêtent pas: pas de rotation aléatoire, ni de symértries
- D'autre part, les quelques augmentations testées (zoom et légères translations) ont fortement fait chuter les performances des modèles.
""")
    

    st.subheader("Résultats VGG16")

    st.subheader("Résultats ResNet50")

    st.subheader("Conclusion sur le DeepLearning")

    