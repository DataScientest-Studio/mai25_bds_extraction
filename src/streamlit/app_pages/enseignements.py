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
    st.title("Conclusion")

    next_section()
    with st.expander("Le temps et l'espace : des ressources précieuses"):
        st.markdown(f"""<span style='color:#ECB959; font-weight:bold'>Rappel de quelques ordres de grandeur:</span><br>
- Nombre d'images dans IIT-CDIP: 7 millions<br>
- Nombre d'images dans RVL-CDIP: 400 000<br>
- Espace de disque nécessaire pour stocker ces 400 000 images (uniquement): 82 Go<br>""", unsafe_allow_html=True)

        st.markdown(f"""<span style='color:#ECB959; font-weight:bold'>Les bons reflexes pour la gestion de l'espace:</span><br>
- enregistrer les dataframes en .parquet (csv trop lourds)<br>
- travailler par batchs d'images pour ne pas dépasser les capacités de la RAM<br>
- créer des chemins clairs de chargement et de sauvegarde des fichiers générés pour éviter les doublons<br>""", unsafe_allow_html=True)
    
        st.markdown(f"""<span style='color:#ECB959; font-weight:bold'>Les bons reflexes pour la gestion du temps:</span><br>
- paralléliser le plus possible dans les algorithmes pour utiliser plusieurs coeurs (même quand ce n'est pas prévu par la fonction)<br>
- créer des fonctions claires et générales à utiliser plusieurs fois<br>
- créer des chemins clairs de chargement et de sauvegarde des fichiers générés pour éviter les doublons<br>
- sauvegarder ce qui prend du temps à générer: dataframes, résultats de lazy classifiers ou grid search, modèles...<br>
- travailler sur de petits échantillons pour mettre en place les algorithmes avant de se lancer sur le dataset complet<br>""", unsafe_allow_html=True)

    with st.expander("Travailler ensemble, un défi commun"):
        st.markdown(f"""<span style='color:#ECB959; font-weight:bold'>Pour être capable de partager le travail:</span><br>
- architecture commune dès le début.<br>
- conventions d'écritures communes pour un visuel cohérent/utilisation et partage de templates<br>
- rédiger des README dans les différents dossiers partagés
- git push et pull réguliers pour limiter les soucis lors des merges<br>
- se tenir au courant des avancées (slack/réunions régulières) pour rester cohérents<br>""", unsafe_allow_html=True)

    with st.expander("Spécifiquement pour des projets de ML et DL"):
        st.markdown(f"""<span style='color:#ECB959; font-weight:bold'>Prétraitement (préprocessing)</span><br>
- Bien connaître ses données : format, distribution, qualité, biais potentiels<br>
- Adapter le nettoyage et les transformations en fonction des modèles envisagés (e.g., normalisation pour les réseaux de neurones, encodage des catégories)<br>
- Réduire la dimensionnalité quand c’est pertinent (PCA, UMAP, sélection de features) pour gagner en performance et lisibilité""", unsafe_allow_html=True)

        st.markdown(f"""<span style='color:#ECB959; font-weight:bold'>Choix et évaluation des modèles</span><br>
- Tester plusieurs familles de modèles (baseline simples → modèles complexes)<br>
- Construire une pipeline claire <br>""", unsafe_allow_html=True)

        st.markdown(f"""<span style='color:#ECB959; font-weight:bold'>Entraînement et optimisation</span><br>
- Penser aux callbacks pour le deep learning (early stopping, checkpoints)<br>
- Sauvegarder systématiquement les modèles entraînés (même s'ils ne semblent pas très bons)<br>
- Partager des notebooks ou scripts clairs pour relancer facilement l’apprentissage""", unsafe_allow_html=True)

    st.markdown("<hr style='border: 1px solid #56BDED;'>", unsafe_allow_html=True)
    st.subheader("📌 Synthèse visuelle des enseignements clés")

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown("##### 🕒 Temps & 💾 Espace")
        st.markdown("""
        <ul style='padding-left: 1em'>
            <li>Fichiers légers : `.parquet`, batchs d’images</li>
            <li>RAM & stockage anticipés</li>
            <li>Optimisation via parallélisation</li>
            <li>Sauvegarde systématique</li>
            <li>Tests sur petits échantillons</li>
        </ul>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("##### 🤝 Collaboration")
        st.markdown("""
        <ul style='padding-left: 1em'>
            <li>Architecture de projet commune</li>
            <li>Conventions de code & visuel</li>
            <li>Utilisation de README et Git</li>
            <li>Communication régulière</li>
        </ul>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("##### 🤖 Spécificités ML/DL")
        st.markdown("""
        <ul style='padding-left: 1em'>
            <li>Préprocessing orienté modèle</li>
            <li>Choix éclairés des algos</li>
            <li>Pipelines claires</li>
            <li>Tracking des modèles</li>
        </ul>
        """, unsafe_allow_html=True)



#     next_section()
#     st.markdown(f"""Ce projet a été l'occasion d'explorer en profondeur les défis et les richesses du machine learning et du deep learning. 
# L’un des principaux obstacles rencontrés a été la gestion du poids des fichiers, qui a mis en lumière l'importance d'une préparation rigoureuse des données et d’une infrastructure adaptée. 
# Très vite, nous avons compris que sans une base solide et bien structurée, les efforts investis dans les modèles eux-mêmes risquaient d’être vains.\n
# L'architecture générale du projet s’est révélée être un point clé : elle a nécessité des ajustements pour garantir la clarté, la cohérence et la pérennité du travail. \n
# À cela s'est ajoutée la complexité du choix entre la multitude de modèles disponibles, chacun demandant une compréhension fine de ses spécificités et un réglage souvent délicat des hyperparamètres.
# Naviguer dans cet écosystème a été formateur, mais également exigeant : entre ajustements techniques, optimisations, essais-erreurs et recherches, chaque étape a contribué à affiner notre compréhension du domaine.\n
# En résumé, un projet dense, stimulant, parfois un peu frustrant, mais toujours enrichissant.""")


    # # region Bilan
    # next_section()

    # with st.expander(""):
    #     image_path = PATHS.streamlit_images / "multimodal" / "plomberie.png"
    #     st.image(image_path, use_container_width=True)

    # st.markdown(f"""
    # - {style.highlight('Efficacité de la combinaisons des forces')}
    
    # - {style.highlight('Croissance de performance au cours des différentes explorations')}

    # - {style.highlight('Explosion combinatoire du nombre de possibilité, qui impose de faire des choix humains')}

    # """, unsafe_allow_html=True)

