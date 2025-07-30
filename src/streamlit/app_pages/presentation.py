from pyvis.network import Network
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import json
from assets import style
from assets import PATHS
from PIL import Image
import os

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
    st.markdown(f"""Dans le nuage de mots représentatif de nos données, on retrouve bien le champ lexical de l'industrie du tabac
                """)
    st.image(PATHS.streamlit_images / "nuage_mots.png", use_container_width=True)

    st.markdown(f"""Pour chaque catégorie de RVL-CDIP, une image et le texte qui en a été extrait vous sont présentés
                """)
    # Chargement du CSV
    df = pd.read_csv(os.path.join(PATHS.streamlit, "assets", "images", "descriptions.csv"))
    df = df.sort_values(by="label").reset_index(drop=True)
    # Dictionnaire des noms de catégories
    LABELS = {
        0: 'letter',
        1: 'form',
        2: 'email',
        3: 'handwritten',
        4: 'advertisement',
        5: 'scientific report',
        6: 'scientific publication',
        7: 'specification',
        8: 'file folder',
        9: 'news article',
        10: 'budget',
        11: 'invoice',
        12: 'presentation',
        13: 'questionnaire',
        14: 'resume',
        15: 'memo'
    }

    # Fonction pour tronquer à N mots
    def truncate_text(text, max_words=50):
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]) + "..."
        return text

    # Affichage des parties
    for _, row in df.iterrows():
        label_id = row["label"]
        label_name = LABELS.get(label_id, "inconnu")  # fallback au cas où
        with st.expander(f"Catégorie {label_id} : {label_name}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                image_path = os.path.join(PATHS.streamlit, "assets", "images", f"label_{row['label']}.png")
                st.image(image_path, use_container_width=True)
            with col2:
                st.markdown(truncate_text(row["text"], max_words=50))

    next_section()
    st.markdown(f"""Les 16 catégories sont parfaitement équilibrées: 
                """)
    file_path = os.path.join(PATHS.streamlit, "assets", "images", "categories_equilibrees.html")
    with open(file_path, 'r', encoding='utf-8') as f:
        components.html(f.read(), height=500)

    st.markdown(f"""Et comme illustré par les graphes suivants, même si on considère la répartition des différentes valeurs obtenues pour les caractéristiques extraites des images, elles sont également bien réparties dans les sets d'entrainement, de test et de validation.
                Pour chaque caractéristique extraite des images, vous trouverez: 
                - un histogramme de la répartition de leurs valeurs
                - leur répartition (en considérant leurs quantiles) dans les sets d'entrainement, de test et de validation
                """)
    with st.expander("La largeur"):

        st.image(os.path.join(PATHS.streamlit, "assets", "images", "histogramme_largeurs.png"), use_container_width=True)
        st.image(os.path.join(PATHS.streamlit, "assets", "images", "hauteur_equilibrees.png"), use_container_width=True)

    with st.expander("La netteté"):

        st.image(os.path.join(PATHS.streamlit, "assets", "images", "histogramme_sharpness.png"), use_container_width=True)
        st.image(os.path.join(PATHS.streamlit, "assets", "images", "sharpness_equilibrees.png"), use_container_width=True)


    with st.expander("Le bruit"):
        st.image(os.path.join(PATHS.streamlit, "assets", "images", "histogramme_noise.png"), use_container_width=True)
        st.image(os.path.join(PATHS.streamlit, "assets", "images", "noise_equilibrees.png"), use_container_width=True)


    with st.expander("Le ratio de pixels blancs"):
        st.image(os.path.join(PATHS.streamlit, "assets", "images", "histogramme_ratio_b.png"), use_container_width=True)
        st.image(os.path.join(PATHS.streamlit, "assets", "images", "ratio_b_equilibrees.png"), use_container_width=True)


    with st.expander("Le ratio de pixels noirs"):
        st.image(os.path.join(PATHS.streamlit, "assets", "images", "histogramme_ratio_n.png"), use_container_width=True)
        st.image(os.path.join(PATHS.streamlit, "assets", "images", "ratio_n_equilibrees.png"), use_container_width=True)

    st.markdown(f"""Pour le nombre de lignes et de colonnes détectées par computer vision, il est intéressant de remarquer que même si les valeurs ne sont pas exactes, elles offrent un bon ordre de grandeur, voici quelques illustrations:
                """)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(os.path.join(PATHS.streamlit, "assets", "images", '2072197187.tif_lignes.png'), width=600)
    with col2:
        st.image(os.path.join(PATHS.streamlit, "assets", "images", '2072197187.tif_colonnes.png'), width=600)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(os.path.join(PATHS.streamlit, "assets", "images", '514409402_514409407.tif_lignes.png'), width=600)
    with col2:
        st.image(os.path.join(PATHS.streamlit, "assets", "images", '514409402_514409407.tif_colonnes.png'), width=600)


    #debugging
    #img1 = Image.open(os.path.join(PATHS.streamlit, "assets", "images", '2072197187.tif_lignes.png'))
    #img2 = Image.open(os.path.join(PATHS.streamlit, "assets", "images", '2072197187.tif_colonnes.png'))
    #st.write("Lignes DPI:", img1.info.get("dpi", "Non défini")) 
    #st.write("Colonnes DPI:", img2.info.get("dpi", "Non défini"))
    #st.write("Taille en pixels (lignes):", img1.size)
    #st.write("Taille en pixels (colonnes):", img2.size)


    with st.expander("Le nombre de lignes"):
        st.image(os.path.join(PATHS.streamlit, "assets", "images", "histogramme_nb_lignes.png"), use_container_width=True)
        st.image(os.path.join(PATHS.streamlit, "assets", "images", "nb_lignes_equilibrees.png"), use_container_width=True)


    with st.expander("Le nombre de colonnes"):
        st.image(os.path.join(PATHS.streamlit, "assets", "images", "histogramme_nb_colonnes.png"), use_container_width=True)
        st.image(os.path.join(PATHS.streamlit, "assets", "images", "nb_col_equilibrees.png"), use_container_width=True)
