import streamlit as st

from assets import PATHS
from assets import style
from app_pages import presentation, texte, images, multimodal, enseignements, demo

# non utilisé car trop de conflits avec les styles injectes par streamlit.
# A la place, utilisation de balises <style> prédéfinies dans assets.style.py 
# stylesheet = PATHS.streamlit / "style.css"
# with open(stylesheet) as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="Classification de documents", layout="wide")

pages = {
    "Présentation": presentation,
    "Texte": texte,
    "Images": images,
    "Multimodal": multimodal,
    "Enseignements": enseignements,
    "Démonstration": demo,
}


# nécessaire pour resetter les compteurs en cas de rechargement de pages
# pas très joli en termes d'architecture, mais bon...
for page in pages.values():
    page.section_counter = 0

main_menu = st.sidebar.selectbox("Menu principal", list(pages.keys()))

current_page = pages[main_menu]

# st.markdown("### Sommaire")
for section in current_page.sections:
    anchor = section.lower().replace(" ", "-")
    st.sidebar.markdown(f"""<a href="#{section}" style="{style.navbar_link}">{section}</a>""", unsafe_allow_html=True)
# Rendu page
current_page.show()
