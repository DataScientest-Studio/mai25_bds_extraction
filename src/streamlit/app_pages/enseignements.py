import streamlit as st

from assets import style
from assets import PATHS

sections = [
    "Enseignements",
    "Conclusions"
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
    st.title("Enseignements et conclusions")

    next_section()
    with st.expander("Le temps et l'espace : des ressources précieuses"):
        st.markdown(f"""Rappel de quelques ordres de grandeur:
- Nombre d'images dans IIT-CDIP: 7 millions
- Nombre d'images dans RVL-CDIP: 400 000
- Espace de disque nécessaire pour stocker ces 400 000 images (uniquement): 82 Go
                    """)

        st.markdown(f"""Les bons reflexes pour la gestion de l'espace:
- enregistrer les dataframes en .parquet (csv trop lourds)
- travailler par batchs d'images pour ne pas dépasser les capacités de la RAM
- créer des chemins clairs de chargement et de sauvegarde des fichiers générés pour éviter les doublons
                """)
    
        st.markdown(f"""Les bons reflexes pour la gestion du temps:
- paralléliser le plus possible dans les algorithmes pour utiliser plusieurs coeurs (même quand ce n'est pas prévu par la fonction)
- créer des fonctions claires et générales à utiliser plusieurs fois
- créer des chemins clairs de chargement et de sauvegarde des fichiers générés pour éviter les doublons
- sauvegarder ce qui prend du temps à générer: dataframes, résultats de lazy classifiers ou grid search, modèles...
- travailler sur de petits échantillons pour mettre en place les algorithmes avant de se lancer sur le dataset complet
                """)

    with st.expander("Travailler ensemble, un défi commun"):
        st.markdown(f"""Pour être capable de partager le travail:
- architecture commune dès le début.
- conventions d'écritures communes pour un visuel cohérent/utilisation et partage de templates
- rédiger des README dans les différents dossiers partagés
- git push et pull réguliers pour limiter les soucis lors des merges
- se tenir au courant des avancées (slack/réunions régulières) pour rester cohérents
""")

    with st.expander("Spécifiquement pour des projets de ML et DL"):
        st.markdown(f"""Pour un preprocessing bien orienté, il faut:
- bien connaître ses données 
- avoir une idée précise des algorithmes qu'on veut lancer dessus. 
""")
        st.markdown(f"""xxx:
- xxx 
- xxx
""")

    next_section()
    st.markdown(f"""Ce projet a été l'occasion d'explorer en profondeur les défis et les richesses du machine learning et du deep learning. 
L’un des principaux obstacles rencontrés a été la gestion du poids des fichiers, qui a mis en lumière l'importance d'une préparation rigoureuse des données et d’une infrastructure adaptée. 
Très vite, nous avons compris que sans une base solide et bien structurée, les efforts investis dans les modèles eux-mêmes risquaient d’être vains.\n
L'architecture générale du projet s’est révélée être un point clé : elle a nécessité des ajustements pour garantir la clarté, la cohérence et la pérennité du travail. \n
À cela s'est ajoutée la complexité du choix entre la multitude de modèles disponibles, chacun demandant une compréhension fine de ses spécificités et un réglage souvent délicat des hyperparamètres.
Naviguer dans cet écosystème a été formateur, mais également exigeant : entre ajustements techniques, optimisations, essais-erreurs et recherches, chaque étape a contribué à affiner notre compréhension du domaine.\n
En résumé, un projet dense, stimulant, parfois un peu frustrant, mais toujours enrichissant.""")

