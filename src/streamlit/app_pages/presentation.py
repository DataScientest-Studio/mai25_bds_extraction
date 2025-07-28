from pyvis.network import Network
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import json
from assets import style
from assets import PATHS
import os
base_path = os.path.dirname(__file__)  # chemin du dossier app_pages/
csv_path = os.path.join(base_path, "illustrations", "descriptions.csv")

sections = [
    "Description du dataset",
    "Extraits",
    "Un dataset équilibré"
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
    st.title("Classification de documents scannés")
    
    next_section()
    st.markdown(f"""IIT CDIP Test Collection contient des images haute résolution de documents numérisés (et leur métadonnées), recueillis à partir des archives publiques de procès intentés contre des compagnies de tabac américaines.
                [Harley et al.](https://adamharley.com/rvl-cdip/) ont nettoyé ces images pour créer RVL-CDIP: 400 000 images en niveaux de gris réparties en 16 catégories, avec 25 000 images par catégorie. Il comprend 320 000 images d'entraînement, 40 000 images de validation et 40 000 images de test. Les images sont redimensionnées de façon à ce que leur plus grande dimension ne dépasse pas 1000 pixels. 
                """)
    # Création du graphe avec fond noir et texte blanc
    net = Network(height="600px", width="100%", bgcolor="#111111", font_color="white", directed=True)

    # ----- Définition des nœuds -----
    net.add_node("A", fixed={'x': False, 'y': True}, y=0, label="IIT-CDIP", title="7 000 000 images scannées", color="#56BDED")
    net.add_node("B", fixed={'x': False, 'y': True}, y=100, label="RVL-CDIP \nHarvey et al", title='400 000 images', color="#56BDED")
    net.add_node("C", fixed={'x': False, 'y': True}, y=200, label="Images initiales", title="Images originales scannées", color="#5098F8")
    net.add_node("C2", fixed={'x': False, 'y': True}, y=200, label="Metadata", title="Métadonnées associées aux images", color="#5098F8")
    net.add_node("D", fixed={'x': False, 'y': True}, y=200, label="Texte océrisé", title="Texte obtenu via OCR", color="#5098F8")
    net.add_node("E", fixed={'x': False, 'y': True}, y=200, label="Images pré-processées", title="\nNuances de gris, \n1000 pixels de large", color="#5098F8")
    net.add_node("E2", fixed={'x': False, 'y': True}, y=200, label="Répartition", title="16 catégories, \nSets de train, test et validation", color="#5098F8")
    net.add_node("F", fixed={'x': False, 'y': True}, y=300, label="Caractéristiques", title="Features visuelles extraites", color="#E24768")
    net.add_node("F2", fixed={'x': False, 'y': True}, y=300, label="Images pour ML", title="Sans marge, \n 100x100 pixels", color="#E24768")
    net.add_node("F3", fixed={'x': False, 'y': True}, y=300, label="Images pour DL", title="3 canaux, \n JPEG", color="#E24768")
    net.add_node("G", fixed={'x': False, 'y': True}, y=300, label="Echantillons", title="Échantillons de travail", color="#E24768")
    net.add_node("H", fixed={'x': False, 'y': True}, y=300, label="Texte pour ML et DL", title="Nettoyage, \nTokenization, \nLemmatisation", color="#E24768")
    net.add_node("I", fixed={'x': False, 'y': True}, y=300, label="Récupération", title="eje42e00 corrompue", color="#E24768")

    # ----- Connexions (flèches) -----
    net.add_edge("A", "B")
    net.add_edge("A", "C2")
    net.add_edge("A", "C")
    net.add_edge("A", "D")
    net.add_edge("B", "E")
    net.add_edge("B", "E2")
    net.add_edge("E", "F")
    net.add_edge("E", "F2")
    net.add_edge("E", "F3")
    net.add_edge("E2", "G")
    net.add_edge("D", "H")
    net.add_edge("C", "I")


    
    options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "directed"
            }
        },
        "physics": {"enabled": False},
        "edges": {"arrows": {"to": {"enabled": True}}},
        "nodes": {"font": {"color": "white"}}
    }

    net.set_options(json.dumps(options))

    # Génère et affiche le graphe
    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    components.html(html, height=600, scrolling=True)

    st.markdown(f"""
    - 2000 documents n'ont pas de données textuelles associées. 
    - 25 echantillons équilibrés (conservation des proportions sur train/val/test et sur les labels), pour nos développements et tests internes sur des jeux de données réduits : 
        - 5 tailles (1000, 4000, 10000, 40000 et 10000 documents) 
        - 5 random states différents
    - Les caractéristiques extraites des images RVL-CDIP:
        - longueur (puisque la largeur est fixée à 1000 pixels)
        - netteté (grâce à la variance du laplacien, normalisée par la taille de l’image
        - bruit (“grain”)
        - proportion de pixels blancs (valeur > 200) et la proportion de pixels noirs (valeur <50)
        - marges hautes, basses, gauches et droites,
        - nombre de lignes,
        - nombre de colonnes.
                """)
    
    next_section()

    st.markdown(f"""Pour chaque catégorie, une image et le texte qui en a été extrait vous sont présentés
                """)
    # Chargement du CSV
    df = pd.read_csv(csv_path)

    # Fonction pour tronquer à N mots
    def truncate_text(text, max_words=50):
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]) + "..."
        return text

    # Affichage des parties
    for index, row in df.iterrows():
        with st.expander(f"Catégorie {index}: {row['categorie']}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                image_path = os.path.join(base_path, "illustrations", row["image"])
                st.image(image_path, use_container_width=True)
            with col2:
                st.markdown(truncate_text(row["text"], max_words=50))

    next_section()
    st.markdown(f"""Les 16 catégories sont parfaitement équilibrées: 
                """)
    file_path = os.path.join(base_path, "illustrations", "categories_equilibrees.html")
    with open(file_path, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=500)