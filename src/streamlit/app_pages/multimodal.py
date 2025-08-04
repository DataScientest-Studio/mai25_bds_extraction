import os
import json

import streamlit as st
import pickle
import pandas as pd
from matplotlib import pyplot as plt

from assets import style
from assets import PATHS, LABELS
from assets.utils import get_rvl_image_path, get_random_image_ids, dumb_draw_spider_graph_dark

# from src.streamlit.utils import dumb_draw_spider_graph_dark

sections = [
    "Pourquoi du multimodal?",
    "Voteurs",
    "Machine Learning",
    "Adaptation CLIP",
    ]

section_counter = 0
def next_section():
    global section_counter
    st.markdown(f'<a name="{sections[section_counter]}"></a>', unsafe_allow_html=True)
    st.markdown(f"<hr style='{style.divider}'>",unsafe_allow_html=True)
    # st.markdown("""<div style="padding-top: 240px; margin-top: -240px;"></div>""",unsafe_allow_html=True)
    st.header(sections[section_counter])
    section_counter += 1


def show():
    st.title("Traitement multimodal")
    st.markdown("*Utiliser les features texte et image pour réaliser la prédiction*")

    # region Pourquoi le MMO?
    next_section() # Pourquoi du multimodal?
    st.markdown(f"""<h3>Intuition: informations unimodales insuffisantes</h3>""", unsafe_allow_html = True)
    col1, col2 = st.columns([1, 2])
    label_left, label_right = 0, 5
    with col1:
        image_id = get_random_image_ids(labels=8, random_state = 4)[0]
        image_path = get_rvl_image_path(image_id)
        st.image(image_path, use_container_width=True)
        st.caption(LABELS[8])
    with col2:
        subcol1, subcol2 = st.columns([1, 1])
        with subcol1:
            image_id = get_random_image_ids(labels=label_left, random_state = 4)[0]
            image_path = get_rvl_image_path(image_id)
            st.image(image_path, use_container_width=True)
            st.caption(LABELS[label_left])
        with subcol2:
            image_id = get_random_image_ids(labels=label_right, random_state = 15)[0]
            image_path = get_rvl_image_path(image_id)
            st.image(image_path, use_container_width=True)
            st.caption(LABELS[label_right])
    st.markdown(style.highlight('So what??'), unsafe_allow_html=True)
    st.text("- Quasiment aucune information textuelle n'est fournie pour un dossier")
    st.text("- Il est difficile sans aller lire le document de définir s'il s'agit d'une lettre ou d'un rapport scientifique.")

    st.markdown(f"""<h3>Confrontation aux résultats obtenus</h3>""", unsafe_allow_html = True)
    
    with open(PATHS.models / "performance_summaries.json") as f:
        performance_summaries = json.load(f)
    text_models = {name: data for name, data in performance_summaries.items() if name.startswith("Text")}
    best_text_model = max(text_models, key=lambda k: performance_summaries[k]["accuracy"])
    image_models = {name: data for name, data in performance_summaries.items() if name.startswith("Image")}
    best_image_model = max(image_models, key=lambda k: performance_summaries[k]["accuracy"])    
    col1, col2 = st.columns([1, 1])

    with col1:
        fig, axe = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 8))
        fig.patch.set_facecolor('black')
        axe.set_facecolor('black')
        dumb_draw_spider_graph_dark([best_text_model],[performance_summaries[best_text_model]["precisions"]], axe=axe)
        st.pyplot(fig)
        st.caption(f"spider graph du meilleur modèle texte ({best_text_model})")
    with col2:
        fig, axe = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 8))
        fig.patch.set_facecolor('black')
        axe.set_facecolor('black')
        dumb_draw_spider_graph_dark([best_image_model],[performance_summaries[best_image_model]["precisions"]], axe=axe)
        st.pyplot(fig)
        st.caption(f"spider graph du meilleur modèle image ({best_image_model})")
    st.markdown(f"""
    <h3>architectures étudiées</h3>

    - unimodaux + voteurs
    
    - modèles composites
    
    - base CLIP
    """, unsafe_allow_html=True)

    # region Voteurs
    next_section()
    st.markdown(f"""<h3>Architecture générale</h3>""",unsafe_allow_html=True)
    image_path = PATHS.streamlit_images / "multimodal" / "schema_voteurs.png"
    st.image(image_path, use_container_width=True)
    st.markdown(f"""Fonctions d'agrégation:

- simples: moyenne / maximum

- moyenne pondérée

- pondérations par classe
    """,unsafe_allow_html=True)
    
    st.markdown(f"<h3>Réultats obtenus</h3>", unsafe_allow_html=True)

    st.markdown(f"<h6>simples: moyenne / maximum</h6>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.latex(r"\hat{y}_{\text{vot}} = \frac{1}{2} \left( \hat{y}_{\text{img}} + \hat{y}_{\text{txt}} \right)")
        st.text("Résultat avec moyenne meilleurs qu'avec maximum")
        st.markdown(f"Score = {style.highlight('84.05%')}", unsafe_allow_html=True)
    with col2:

        fig, axe = plt.subplots(subplot_kw=dict(polar=True), figsize=(12, 8))
        fig.patch.set_facecolor('black')
        axe.set_facecolor('black')
        names = [
            "MMO-Voter Average on img-LGBM+txt-LogRed",
            "Image-based LGBM",
            "Text-based Logistic Regressor"
        ]
        precision_lists = [
            performance_summaries[name]["precisions"] for name in names
        ]
        dumb_draw_spider_graph_dark(names,precision_lists, axe=axe)
        st.pyplot(fig)
        st.caption(f"{names[0]}")

    st.markdown(f"<h6>moyenne pondérée</h6>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.latex(r"\hat{y}_{\text{vot}} = (1 - \alpha) \cdot \hat{y}_{\text{img}} + \alpha \cdot \hat{y}_{\text{txt}}")
        image_path = PATHS.streamlit_images / "multimodal" / "weighted_voter.png"
        st.image(image_path, use_container_width=True)
        st.markdown(f"Score = {style.highlight('84.06%')}", unsafe_allow_html=True)
    with col2:

        fig, axe = plt.subplots(subplot_kw=dict(polar=True), figsize=(12, 8))
        fig.patch.set_facecolor('black')
        axe.set_facecolor('black')
        names = [
            "MMO-Voter 0.49-weighted on img-LGBM+txt-LogRed",
            "Image-based LGBM",
            "Text-based Logistic Regressor"
        ]
        precision_lists = [
            performance_summaries[name]["precisions"] for name in names
        ]
        dumb_draw_spider_graph_dark(names,precision_lists, axe=axe)
        st.pyplot(fig)
        st.caption(f"{names[0]}")

    st.markdown(f"<h6>pondérations par classe</h6>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.latex(r"\hat{y}_{\text{vot}} = (\mathbf{I}_{16} - \mathbf{A}) \times \hat{y}_{\text{img}} + \mathbf{A} \times \hat{y}_{\text{txt}}")
        st.latex(r"""\mathbf{A} =
        \begin{bmatrix}
        \alpha_1 & 0        & \cdots & 0 \\
        0        & \alpha_2 & \cdots & 0 \\
        \vdots   & \vdots   & \ddots & \vdots \\
        0        & 0        & \cdots & \alpha_{16}
        \end{bmatrix}
        """)
        st.latex(r"\alpha_i = \frac{y_{\text{txt}, i}}{y_{\text{txt}, i} + y_{\text{img}, i}}")
        st.markdown(f"Score = {style.highlight('84.24%')}", unsafe_allow_html=True)
    with col2:
        fig, axe = plt.subplots(subplot_kw=dict(polar=True), figsize=(12, 8))
        fig.patch.set_facecolor('black')
        axe.set_facecolor('black')
        names = [
            "MMO-Voter class-weighted on img-LGBM+txt-LogRed",
            "Image-based LGBM",
            "Text-based Logistic Regressor"
        ]
        precision_lists = [
            performance_summaries[name]["precisions"] for name in names
        ]
        dumb_draw_spider_graph_dark(names,precision_lists, axe=axe)
        st.pyplot(fig)
        st.caption(f"{names[0]}")
    
    # region Composite
    next_section()

    st.markdown(f"""<h3>Architecture générale</h3>""",unsafe_allow_html=True)
    image_path = PATHS.streamlit_images / "multimodal" / "schema_composite.png"
    st.image(image_path, use_container_width=True)
    st.markdown(f"""modèle: 1 seul type exploré : LogisticRegression (mais architecture adaptée pour tout type de ML)
    """,unsafe_allow_html=True)
    st.markdown(f"""<h3>Résultats</h3>""",unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"Score = {style.highlight('84.44%')}", unsafe_allow_html=True)
    with col2:
        fig, axe = plt.subplots(subplot_kw=dict(polar=True), figsize=(12, 8))
        fig.patch.set_facecolor('black')
        axe.set_facecolor('black')
        names = [
            "MMO-Composite LogReg on img-LGBM+txt-LogReg",
            "Image-based LGBM",
            "Text-based Logistic Regressor"
        ]
        precision_lists = [
            performance_summaries[name]["precisions"] for name in names
        ]
        dumb_draw_spider_graph_dark(names,precision_lists, axe=axe)
        st.pyplot(fig)
        st.caption(f"{names[0]}")
    st.markdown(f"""Possibilité de multiplier les expériences:

- modèles texte et image utilisés en entrée

- type de modèle d'aggrégation (ML ou DL)
""")

    # region Composite
    next_section()
    st.markdown(f"""<h3>Architecture générale</h3>""",unsafe_allow_html=True)
    image_path = PATHS.streamlit_images / "multimodal" / "schema_clip.png"
    st.image(image_path, use_container_width=True)
    st.markdown(f"""modèle: 1 seul type exploré : LogisticRegression (mais architecture adaptée pour tout type de ML)
    """,unsafe_allow_html=True)
    st.markdown(f"""<h3>Résultats</h3>""",unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        image_path = PATHS.streamlit_images / "multimodal" / "tableau.png"
        st.image(image_path, use_container_width=True)
        st.markdown(f"Meilleur score = {style.highlight('90.16%')}", unsafe_allow_html=True)
    with col2:
        fig, axe = plt.subplots(subplot_kw=dict(polar=True), figsize=(12, 8))
        fig.patch.set_facecolor('black')
        axe.set_facecolor('black')
        names = [
            "MMO-CLIP-Based MLP1",
            "MMO-CLIP-Based MLP5",
            "MMO-CLIP-Based MLP7",
            "MMO-CLIP-Based MLP9",
            "MMO-CLIP-Based MLP12",
        ]
        precision_lists = [
            performance_summaries[name]["precisions"] for name in names
        ]
        dumb_draw_spider_graph_dark(names,precision_lists, axe=axe)
        st.pyplot(fig)
        st.caption(f"Exemples de {names[0][:-1]}")

