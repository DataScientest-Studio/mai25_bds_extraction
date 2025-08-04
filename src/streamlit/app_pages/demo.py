import streamlit as st

from assets import style
from assets import PATHS
from src import PATHS, LABELS
import pandas as pd
from PIL import Image
from pathlib import Path

sections = [
    "Sélection des modèles",
    "Sélection d'une image",
    "Prédiction"
    ]

section_counter = 0
def next_section():
    global section_counter
    st.markdown(f'<a name="{sections[section_counter]}"></a>', unsafe_allow_html=True)
    st.markdown(f"<hr style='{style.divider}'>",unsafe_allow_html=True)
    # st.markdown("""<div style="padding-top: 240px; margin-top: -240px;"></div>""",unsafe_allow_html=True)
    st.header(sections[section_counter])
    section_counter += 1

from src.models.model_wrappers import ModelWrapperFactory, AGG_FN, png_image_paths



# Genre de "State" de la page... (conservé entre les rendus)
non_voter_wrappers = []
voter_wrappers = []
predictions = {}
current_image_preview = None
# chargement des images
documents = pd.read_parquet(PATHS.metadata / "df_documents.parquet").index
documents_rvl = pd.read_parquet(PATHS.metadata / "df_documents_save.parquet").index
imported_ids = [i for i in documents if i not in documents_rvl]
df_images = pd.read_parquet(PATHS.metadata / "df_filepaths.parquet")[["filename", "rvl_image_path"]].loc[imported_ids]\
    .apply({"filename": lambda x:x, "rvl_image_path": lambda x: str(PATHS.data/x)})\
    .rename(columns={"rvl_image_path":"path", "filename": "name"})\
    .sort_values("name")
selected_images = df_images.index.tolist()


def show():
    global current_image_preview

    def display_sidebar_footer(non_voter_wrappers, voter_wrappers, selected_images):
        """
        Affiche un récapitulatif fixé en bas de la sidebar :
        - Nombre total de modèles non-voteurs
        - Nombre de voteurs
        - Nombre d'images sélectionnées
        """
        # CSS pour fixer la boîte en bas de la sidebar
        st.markdown("""
            <style>
            .sidebar-footer {
                position: fixed;
                bottom: 0;
                left: 0;
                padding: 1rem;
                border-top: 1px solid #ddd;
                font-size: 0.875rem;
            }
            </style>
        """, unsafe_allow_html=True)

        footer_placeholder = st.sidebar.empty()
        # Boîte HTML personnalisée
        footer_placeholder.markdown(f"""
            <div class="sidebar-footer">
                <strong>🧠 État de la sélection</strong><br>
                Modèles non-voteurs : <strong>{len(non_voter_wrappers)}</strong><br>
                Voteurs : <strong>{len(voter_wrappers)}</strong><br>
                Images sélectionnées : <strong>{len(selected_images)}</strong>
            </div>
        """, unsafe_allow_html=True)

    next_section()

    model_names = ModelWrapperFactory.get_registered()

    c1, c2 = st.columns([1,3])
    with c1:
        st.markdown("<h5>📦 Modèles disponibles<h5>", unsafe_allow_html=True)
    with c2:
        c21, c22 = st.columns([4, 1])
        with c21:
            selected_models = st.multiselect("Sélectionnez un ou plusieurs modèles", model_names)
        with c22:
            if st.button("Appliquer"):
                non_voter_wrappers.clear()
                for name in selected_models:
                    non_voter_wrappers.append(ModelWrapperFactory.load_existing(name))
                    st.markdown(
                        "<div style='background-color:#d4edda; color:#155724; padding:6px 10px; "
                        "border-radius:5px; font-size:0.4rem; border:1px solid #c3e6cb;'>"
                        f"✅ {name}"
                        "</div>",
                        unsafe_allow_html=True
                    )



    # selected_wrappers = [all_wrappers[name] for name in selected_models]

    # === 2. Définir un voteur ===

    c1, c2 = st.columns([1,3])
    with c1:
        st.markdown("<h5>🧮 Définir un voteur personnalisé<h5>", unsafe_allow_html=True)
    with c2:
        with st.expander("Créer un voteur"):

            c21, c22 = st.columns([1, 1])
            with c21:
                voter_name = st.text_input("Nom du voteur", value="MonVoteur")
                agg_fn_label = st.selectbox("Fonction d'agrégation", [fn.name for fn in AGG_FN])
                agg_fn = AGG_FN[agg_fn_label]
            with c22:
                submodel_names = st.multiselect("Choisissez les modèles à inclure", model_names, key="submodel_names")

            weights = None
            if agg_fn in [AGG_FN.WEIGHTED, AGG_FN.CLASS_WEIGHTED] and submodel_names:
                st.markdown("### ⚖️ Définir les poids")
                weights = []
                for name in submodel_names:
                    w = st.slider(f"Poids pour {name}", min_value=0.0, max_value=1.0, value=1.0)
                    weights.append(w)

            if st.button("➕ Ajouter ce voteur"):
                new_voter = ModelWrapperFactory.make_mmo_voter_wrapper(
                    voter_name,
                    [ModelWrapperFactory.load_existing(submodel_name) for submodel_name in submodel_names],
                    agg_fn=agg_fn,
                    weights=weights
                    )
                voter_wrappers.append(new_voter)
                submodel_names.clear()




                # st.session_state.setdefault("custom_voters", {})[voter_name] = new_voter

    # === 3. Affichage des performances ===
#    st.subheader("📊 Performances des modèles")

    # Fusion des modèles classiques et voteurs créés
    # all_models_to_show = selected_wrappers + list(st.session_state.get("custom_voters", {}).values())

    # if all_models_to_show:
    #     for mw in all_models_to_show:
    #         st.markdown(f"### ✅ {mw.name}")
    #         # Tu peux mettre ici des métriques / confusion matrix / courbes
    #         st.write("À venir : affichage des graphes")
    # else:
    #     st.info("Aucun modèle sélectionné ou créé.")


    #region Sélection image
    next_section()



    col1, col2 = st.columns([2, 2])
    checkboxes = {}
    with col1:
        st.markdown("##### Images disponibles")

        for img_id, row in df_images.iterrows():
            # Checkbox de sélection
#            checked = img_id in st.session_state.selected_image_ids
            name_col, preview_col, select_col = st.columns([5, 3, 2])
            with name_col:
                st.write(row["name"])
            with preview_col:
                if st.button("preview", key=f"btn_{img_id}"):
                    current_image_preview = img_id
            with select_col:
                checkboxes[img_id] = st.checkbox("", key=f"chk_{img_id}", value=img_id in selected_images)
    with col2:
        title_col, button_col = st.columns([3, 2])
        with title_col:
            st.markdown("### 🔍 Aperçu")
        with button_col:
            if st.button("Appliquer", key="img_sel_apply"):
                selected_images.clear()
                for img_id, selected in checkboxes.items():
                    if selected:
                        selected_images.append(img_id)
        if current_image_preview:
            path = df_images.loc[current_image_preview]["path"]
            try:
                img = Image.open(path)
                st.image(img, caption=f"Aperçu de {df_images.loc[current_image_preview,"name"]}", use_container_width=True)
            except Exception as e:
                st.error(f"Erreur de lecture de l’image : {e}")
        else:
            st.info("Cliquez sur un nom d’image à gauche pour afficher un aperçu.")





    #region Prédiction
    next_section()
    if st.button("Predict", key="compute_predictions"):
        predictions.clear()
        for model in non_voter_wrappers + voter_wrappers:
            preds = model.predict(selected_images)
            # probas = model.predict_probas(selected_images)
            predictions[model.name] = {
                img_id: {
                    "prediction": pred,
                    # "probas": probs
                    }
                    # for img_id, pred, probs in zip(selected_images, preds, probas)
                    for img_id, pred in zip(selected_images, preds)
            }

    if predictions:
        data = {}

        for model_name, preds_dict in predictions.items():
            data[model_name] = {
                img_id: info["prediction"] for img_id, info in preds_dict.items()
            }

        df_preds = pd.DataFrame(data).T
        df_preds.index.name = "Image ID"

        st.subheader("Synthèse")
        st.dataframe(df_preds)

            
            # st.text(predictions)
            
        #     for model in non_voter_wrappers:
        # row = {"model": model.name}
        # for img_id in st.session_state.selected_image_ids:
        #     pred_class = model.predict(img_id)
        #     row[img_id] = pred_]()

        for img_id in selected_images:
            img_name = str(Path(png_image_paths[img_id]).name)
            # st.markdown(f"### 🖼️ Image : `{img_name}`")
            
            cols = st.columns([1, 2])  # [gauche, droite]

            # Colonne de gauche : preview de l'image
            with cols[0]:
                image_path = png_image_paths[img_id]
                st.image(image_path, caption=img_name, use_container_width=True)

            # Colonne de droite : prédictions des modèles
            with cols[1]:
                for model_name in predictions:
                    pred_label_code = predictions[model_name][img_id]["prediction"]
                    pred_label_plain = LABELS.get(pred_label_code, f"Label inconnu ({pred_label_code})")
                    st.markdown(f"**{model_name}**          -> {pred_label_plain}", unsafe_allow_html=True)



        









    display_sidebar_footer(non_voter_wrappers, voter_wrappers, selected_images)
