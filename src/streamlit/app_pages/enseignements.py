import streamlit as st

from assets import style
from assets import PATHS

sections = [
    "Enseignements",
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
    st.title("Conclusion : Une quête constante de l’optimisation")

    next_section()

    st.markdown("### Optimisation, notre moteur commun")

    st.markdown("""
    - **Optimisation de l’architecture** :  
    Organisation modulaire du code, séparation claire des fonctions, notebooks structurés (du moins, on a essayé!!), et pipelines revus pour plus d’efficacité.

    - **Optimisation de la communication** :  
    README bien renseigné, fichiers partagés sur drive, bonne synchronisation d’équipe – même si Slack a connu des pics d'activité.

    - **Optimisation des temps de traitement** :  
    Tests sur des DataFrames réduits, parallélisation, astuces techniques partagées, et quelques miracles de dernière minute.

    - **Une armée de modèles et de paramètres** :  
    Grid Search, Random Search, tuning à l’infini… Et bien sûr, un **modèle qui tournait encore la veille de la présentation** 

    - **Rigueur & entraide** :  
    Chaque optimisation a été portée par des échanges, des idées partagées, et une belle dynamique d’équipe.
    """)

    st.markdown("Un projet où performance rime avec collaboration (et caféine) !")
