import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from assets import PATHS
from assets import style
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
# src/streamlit/app_pages/texte.py

sections = ["Preprocessing", "Machine Learning", "Deep Learning", "Evaluation"]  

section_counter = 0
def next_section():
    global section_counter
    st.markdown(f'<a name="{sections[section_counter]}"></a>', unsafe_allow_html=True)
    st.markdown(f"<hr style='{style.divider}'>",unsafe_allow_html=True)
    # st.markdown("""<div style="padding-top: 240px; margin-top: -240px;"></div>""",unsafe_allow_html=True)
    st.header(sections[section_counter])
    section_counter += 1

def show():

    st.title("📄 Analyse Textuelle des Documents OCR")

    st.header("🎯 Objectif du projet")
    st.markdown("""
    Prétraiter et analyser automatiquement un ensemble de documents OCR pour :
    - Nettoyer les métadonnées
    - Extraire du contenu textuel structurel
    - Identifier les entités clés (noms, dates, organisations…)
    - Classifier automatiquement les types de documents
    - Poser les bases pour des analyses NLP avancées
    """)

    # st.image(os.path.join(PATHS.streamlit, "assets", "images","pipeline.png"), caption="Objectif du pipeline", use_container_width=True)

    st.header("🧹 Prétraitement des données")
    st.markdown("""
    Les étapes principales du nettoyage ont été effectuées avec **Pandas** :
    - Suppression des doublons et lignes vides
    - Nettoyage des chaînes de caractères (`None`, espaces, casse)
    - Parsing des dates (`document_date`, `scan_date`)
    - Nettoyage des champs textuels (`title`, `ocr_text`, `author`)
    - Standardisation des types de documents (`dt`)
    - Conversion des champs numériques (`pages_amount`, `np`, etc.)
    """)
    st.image(os.path.join(PATHS.streamlit, "assets", "images","dfocr.png"), caption="DF Avant après", use_container_width=True)

    st.subheader("🔧 Problèmes rencontrés")
    st.markdown("""
    Certains jeux de données étaient très hétérogènes :
    - Valeurs manquantes fréquentes
    - Données bruyantes ou mal OCRisées
    - Incohérences dans les métadonnées
    """)
    st.image(os.path.join(PATHS.streamlit, "assets", "images","pb_rencontre.png"), caption="Ocr handwritten", use_container_width=True)
    st.image(os.path.join(PATHS.streamlit, "assets", "images","valeurs manquantes.png"), caption="Valeurs manquantes", use_container_width=True)
    st.header("🧠 Extraction automatique du contenu")
    st.markdown("""
    Les étapes d’analyse de texte comprennent :
    - **Tokenisation** avec POS-tagging et lemmatisation
    - **Segmentation en phrases** : pour préparation au résumé ou à la structuration
    - **Stop words plus suppression caractères trop reccurents (pgnbr,html, etc.)**
    """)

    st.header("🧪 Modélisation : Classification des types de documents")
    st.subheader("Pipeline ML (modèles classiques)")
    st.markdown("""
    1. **TF-IDF Vectorization**
    2. **Logistic Regression**
    3. **Évaluation (train/test split)**
    4. **Analyse des résultats**
    """)

    st.image(os.path.join(PATHS.streamlit, "assets", "images","vecteur.png"), caption="Ocr handwritten", use_container_width=True)

    st.subheader("📊 Résultats")
    st.markdown("Les performances varient selon la source des données. Deux jeux sur trois ont été retenus pour l’expérimentation.")
    st.markdown("Un pipeline complet a été conçu pour automatiser tout le processus :")
    st.markdown("- Nettoyage → Vectorisation → Modèle → Prédiction")

    st.header("📈 Modèles Classiques : Comparaison")

    ### Logistic Regression
    st.subheader("🔹 Régression Logistique")

    st.image(os.path.join(PATHS.streamlit, "assets", "images","REG LOG.png"), caption="Resultats Regression logistique", use_column_width=True)

    st.markdown("""
    - **Accuracy** : 82%
    - **F1 macro** : 82%
    - Classes bien identifiées : `2`, `7`, `14`
    - Classe difficile : `3` (handwritten)

    | Classe | Précision | Rappel | F1-Score | Commentaire |
    |--------|-----------|--------|----------|-------------|
    | 2      | 0.95      | 0.91   | 0.93     | Excellent   |
    | 3      | 0.62      | 0.76   | 0.68     | Ambiguë     |
    | 8      | 0.63      | 0.81   | 0.71     | Précision faible |
    """)


    ### Random Forest
    st.subheader("🔸 Random Forest")
    st.image(os.path.join(PATHS.streamlit, "assets", "images","Random Forest.png"), caption="Resultats Random Forest", use_column_width=True)

    st.markdown("""
    - **Accuracy** : 82%
    - **F1 macro** : 82%
    - Moins interprétable mais robuste
    - Résultats plus variés selon les classes

    | Classe | Précision | Rappel | F1-Score | Commentaire |
    |--------|-----------|--------|----------|-------------|
    | 14     | 0.98      | 0.97   | 0.98     | Quasi parfaite |
    | 3      | 0.56      | 0.75   | 0.64     | Confusion forte |
    """)

    ### Naive Bayes
    st.subheader("⚪ Naive Bayes")
    st.image(os.path.join(PATHS.streamlit, "assets", "images","result naive bayes.png"), caption="Resultats Naive Bayes", use_container_width=True)

    st.markdown("""
    - **Accuracy** : 68%
    - **F1 macro** : 67%
    - Bon sur classes claires, faible sur classes ambiguës

    | Classe | F1-Score | Commentaire |
    |--------|----------|-------------|
    | 2, 7, 14 | > 0.85 | Bonnes performances |
    | 3, 8    | < 0.60 | Très mal reconnues |
    """)
    st.image(os.path.join(PATHS.streamlit, "assets", "images","comparaison metriques graphique.png"), caption="Ocr handwritten", use_container_width=True)

    st.header("🧠 Deep Learning sur le texte (MLP)")

    st.markdown("""
    - Texte encodé avec **Tokenizer Keras** → Séquences entières
    - Padding à longueur fixe (300)
    - Architecture simple avec :
        - Embedding
        - GlobalAveragePooling1D
        - Dense (ReLU + Dropout)
        - Softmax final
    """)


    st.subheader("📉 Résultats")

    st.image(os.path.join(PATHS.streamlit, "assets", "images","resultat dl.jpg"), caption="Objectif du pipeline", use_container_width=True)
    st.image(os.path.join(PATHS.streamlit, "assets", "images","Matrice confusion mlp.png"), caption="Objectif du pipeline", use_container_width=True)

    st.markdown("""
    - **Accuracy** : 81%
    - Classes très bien apprises : `2`, `7`, `14`
    - Confusion sur : `3`, `8`, `15`
    """)

    st.markdown("""
    | Classe | Précision | Rappel | F1-score | Support | Commentaire |
    |--------|-----------|--------|----------|---------|-------------|
    | 2      | 0.92      | 0.92   | 0.92     | 2515    | Excellent |
    | 3      | 0.65      | 0.68   | 0.67     | 2230    | Ambiguë et bruitée |
    | 8      | 0.65      | 0.66   | 0.65     | 1298    | Peut-être à traiter via multimodal |
    """)

    st.image(os.path.join(PATHS.streamlit, "assets", "images","courbe de perte mlp.png"), caption="Resultats Regression logistique", use_container_width=True)

    st.header("🧾 Conclusion")
    st.markdown("""
    - Le **MLP** offre des performances solides (F1 macro = 81%).
    - Les modèles **classiques** (Logistic Regression, RF) sont performants, rapides et interprétables.
    - Le **Naive Bayes** reste utile pour des cas simples.
    - Des pistes d’amélioration sont possibles :
        - LSTM, BERT
        - Ajout de métadonnées
        - Approche **multimodale**
    """)