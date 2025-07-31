from types import SimpleNamespace
from matplotlib.colors import ListedColormap

colors = SimpleNamespace(
    black1 = "#111111",
    grey1 = "#31333E",
    grey2 = "#818494",
    grey3 = "#A4A8B7",
    white1 = "#FAFAFA",
    yellow1 = "#ECB959",
    blue1 = "#56BDED",
    blue2 = "#5098F8",
    pink1 = "#E24768"
)

def highlight(text, color=colors.yellow1, bold=False, italic=False):
    style = f"color: {color};"
    if bold:
        style += " font-weight: bold;"
    if italic:
        style += " font-style: italic;"
    return f"<span style='{style}'>{text}</span>"

graph_colors = [
    "#93C7FA", "#2B66C2", "#F3AFAD",
    "#EA4339", "#9AECA8", "#57AD9D", "#F8D37A"
]
colormap = ListedColormap(graph_colors, name='my_palette')

divider = f"""
    border: none;
    border-top: 3px solid {colors.blue1};
"""

navbar_link = f"""
    text-decoration: none;
    display: block;
    padding: 0.5rem 1rem;
    margin: 0.3rem 0;
    border-radius: 5px;
    font-weight: 500;
    color: {colors.white1};
"""
import streamlit as st
from assets import style
from app_pages import presentation, texte, images, multimodal, enseignements, demo

st.set_page_config(page_title="Classification de documents", layout="wide")

pages = {
    "Présentation": presentation,
    "Texte": texte,
    "Images": images,
    "Multimodal": multimodal,
    "Enseignements": enseignements,
    "Démonstration": demo,
}

for page in pages.values():
    page.section_counter = 0

main_menu = st.sidebar.selectbox("Menu principal", list(pages.keys()))
current_page = pages[main_menu]

for section in current_page.sections:
    anchor = section.lower().replace(" ", "-")
    st.sidebar.markdown(
        f"""<a href="#{section}" style="{style.navbar_link}">{section}</a>""",
        unsafe_allow_html=True,
    )

current_page.show()
import streamlit as st
from assets import style
from app_pages import presentation, texte, images, multimodal, enseignements, demo

st.set_page_config(page_title="Classification de documents", layout="wide")

pages = {
    "Présentation": presentation,
    "Texte": texte,
    "Images": images,
    "Multimodal": multimodal,
    "Enseignements": enseignements,
    "Démonstration": demo,
}

for page in pages.values():
    page.section_counter = 0

main_menu = st.sidebar.selectbox("Menu principal", list(pages.keys()))
current_page = pages[main_menu]

for section in current_page.sections:
    anchor = section.lower().replace(" ", "-")
    st.sidebar.markdown(
        f"""<a href="#{section}" style="{style.navbar_link}">{section}</a>""",
        unsafe_allow_html=True,
    )

current_page.show()

import streamlit as st
from utils.data_loader import load_ocr_dataframe
from utils.display import show_ocr_example
from assets import style

sections = [
    "Chargement des données",
    "Exploration visuelle",
    "Exemple image + OCR",
    "Pré-traitement OCR",
    "Modélisation",
]

def show():
    st.title(style.highlight("Analyse des Textes OCR", color=style.colors.blue2, bold=True))

    st.markdown(f"<hr style='{style.divider}'>", unsafe_allow_html=True)
    st.header("1. Chargement des données")
    df = load_ocr_dataframe()
    st.write(df.head())

    st.markdown(f"<hr style='{style.divider}'>", unsafe_allow_html=True)
    st.header("2. Exploration visuelle")
    st.markdown("Distribution par `label` et `data_set`")
    st.bar_chart(df['label'].value_counts())

    st.markdown(f"<hr style='{style.divider}'>", unsafe_allow_html=True)
    st.header("3. Exemple image + OCR")
    doc_id = st.text_input("Document ID", value="X0000000056")
    if st.button("Afficher OCR et Image"):
        show_ocr_example(doc_id)

    st.markdown(f"<hr style='{style.divider}'>", unsafe_allow_html=True)
    st.header("4. Pré-traitement OCR")
    st.markdown(style.highlight("""
    - Suppression des `pgNbr`
    - Déséchappement HTML
    - Correction Jamspell
    - Filtrage de caractères
    - Suppression des stopwords
    """, color=style.colors.yellow1), unsafe_allow_html=True)

    st.markdown(f"<hr style='{style.divider}'>", unsafe_allow_html=True)
    st.header("5. Modélisation")
    st.markdown(style.highlight("""
    - TF-IDF Vectorizer
    - Modèles : Logistic Regression, Random Forest, Naive Bayes
    - Affichage de rapport de classification
    """, color=style.colors.pink1), unsafe_allow_html=True)

st.markdown(
    f"""<style>
        body {{
            background-color: {style.colors.black1};
            color: {style.colors.white1};
        }}
        .stApp {{
            background-color: {style.colors.black1};
        }}
    </style>""",
    unsafe_allow_html=True
)
import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sections = ["Deep Learning - MLP"]

def show():
    st.title("🧠 Deep Learning - MLPClassifier")

    st.subheader("1. Chargement des données")
    home = os.path.expanduser("~")
    documents_path = os.path.join(home, "Documents")

    train_df = pd.read_csv(os.path.join(documents_path, "train_df.csv"))
    val_df = pd.read_csv(os.path.join(documents_path, "val_df.csv"))

    # Nettoyage
    train_df = train_df.dropna(subset=["raw_ocr_clean"])
    val_df = val_df.dropna(subset=["raw_ocr_clean"])

    st.write("✅ Données chargées :", len(train_df), "train -", len(val_df), "validation")

    st.subheader("2. Vectorisation (TF-IDF)")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df["raw_ocr_clean"])
    X_val = vectorizer.transform(val_df["raw_ocr_clean"])

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df["label"])
    y_val = encoder.transform(val_df["label"])

    st.write("✅ TF-IDF shape :", X_train.shape)

    st.subheader("3. Entraînement du modèle MLP (Scikit-learn)")
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    with st.spinner("Entraînement en cours..."):
        mlp.fit(X_train, y_train)

    st.success("✅ Entraînement terminé")

    y_pred = mlp.predict(X_val)

    st.subheader("4. Rapport de classification")
    report = classification_report(y_val, y_pred, target_names=encoder.classes_.astype(str), output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap="Blues"))

    st.subheader("5. Matrice de confusion")
    cm = confusion_matrix(y_val, y_pred)
    fig_cm, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_,
                ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title("Matrice de confusion - MLPClassifier")
    st.pyplot(fig_cm)

    st.subheader("6. Courbe de perte (loss)")
    fig_loss, ax2 = plt.subplots()
    ax2.plot(mlp.loss_curve_)
    ax2.set_title("Courbe de perte - MLP")
    ax2.set_xlabel("Itérations")
    ax2.set_ylabel("Loss")
    ax2.grid(True)
    st.pyplot(fig_loss)
