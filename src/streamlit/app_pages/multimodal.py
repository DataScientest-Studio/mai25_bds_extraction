import os

import streamlit as st
import pickle
import pandas as pd

from assets import style
from assets import PATHS, LABELS
from assets.utils import get_rvl_image_path, get_random_image_ids

sections = [
    "Pourquoi du multimodal?",
    "Voteurs",
    "Machine Learning",
    "Adaptation CLIP",
    "Bilan"
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

    next_section() # Pourquoi du multimodal?
    st.markdown(f"""<h3>Intuition: informations unimodales insuffisantes</h3>""", unsafe_allow_html = True)
    
    col1, col2 = st.columns([1, 2])
    label_left, label_right = 0, 5
    with col1:
        image_id = get_random_image_ids(labels=8, random_state = 4)[0]
        image_path = get_rvl_image_path(image_id)
        st.image(image_path, use_container_width=True)
    st.caption(LABELS[8])
    st.text("Quasiment aucune information textuelle n'est fournie pour un dossier")
    with col2:
        subcol1, subcol2 = st.columns([1, 1])
        with subcol1:
            image_id = get_random_image_ids(labels=label_left, random_state = 123)[0]
            image_path = get_rvl_image_path(image_id)
            st.image(image_path, use_container_width=True)
        st.caption(LABELS[label_left])
        with subcol2:
            image_id = get_random_image_ids(labels=label_right, random_state = 123)[0]
            image_path = get_rvl_image_path(image_id)
            st.image(image_path, use_container_width=True)
        st.caption(LABELS[label_right])
        st.text("Il est difficile sans aller lire le document de définir s'il s'agit d'une lettre ou d'un rapport scientifique.")

    st.markdown(f"""<h3>Confrontation aux résultats obtenus</h3>""", unsafe_allow_html = True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.caption("spider graph du meilleur modèle texte")
    with col2:
        st.caption("spider graph du meilleur modèle image")
    st.markdown(f"""
    <h3>en pratique</h3>
    comparatif arraignée Alexis vs Camille
    <h3>architectures étudiées</h3>

    - unimodaux + voteurs
    
    - unimodaux + nouveau modèle
    
    - CLIP + nouveau modèle
    """
    , unsafe_allow_html=True)
    next_section()
    next_section()
    next_section()

