# UN MODULE POUR METTRE TOUTES LES FONCTIONS COMMUNES
import os
import numpy as np
import pandas as pd

from assets import PATHS, CLASSES

def get_rvl_image_path(document_id):
    """Return given image_path"""
    df = pd.read_parquet(
        os.path.join(PATHS.metadata, "df_filepaths.parquet"))
    row = df.loc[document_id, :]
    return os.path.join(PATHS.data, row.rvl_image_path)

def get_random_image_ids(
    quantity=1,
    labels=None,
    random_state = None
    ):
    df = pd.read_parquet(
        os.path.join(PATHS.metadata, "df_filepaths.parquet"))
    df_labels = pd.read_parquet(
        os.path.join(PATHS.metadata, 'df_encoded_labels.parquet'))
    if isinstance(labels, int):
        labels=[labels]
    if labels is not None:    
        df_labels = df_labels[df_labels.label.isin(labels)]
    return list(df_labels.sample(quantity, random_state=random_state).index)

def draw_spider_graph(model_names:list[str], y_preds:list[np.array], y_evals:list[np.array], axe):

    assert len(model_names = len(y_preds == len(y_evals)))

    # draw graphs
    plain_labels = CLASSES.values()
    angles = np.linspace(0, 2*np.pi, len(plain_labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    for y_pred, y_eval, model_name in zip(y_preds, y_evals, model_names):    
        precisions = precision_score(y_eval, y_pred, average=None).tolist() #, labels=list(range(16)))
        precisions.append(precisions[0])
        axe.plot(angles, precisions, '.-', linewidth=1, label=model_name)
        axe.fill(angles, precisions, alpha=0.25)

    # improve visual render        
    axe.set_theta_offset(np.pi / 2)
    axe.set_theta_direction(-1)
    axe.set_ylim(0, 1)
    axe.set_rlabel_position(0)
    axe.grid(True)
    axe.set_thetagrids([], [])  # Supprime les labels d’angle (0°, 45°, etc.)
    axe.set_rlabel_position(angles[1]/np.pi*180/2)  # angle en degrés
    for label in axe.get_yaxis().get_ticklabels():
        label.set_fontsize(8)  # ou 7, 6, etc.
    label_angles = np.degrees(angles[:-1])    
    for i, label in enumerate(plain_labels):
        angle_rad = angles[i]
        alignment = 'left' if 0 < round(label_angles[i]) < 180 else ('right' if 180 < round(label_angles[i]) < 360 else 'center')
        axe.text(
            angle_rad,
            1.05,  # rayon légèrement supérieur à 1 pour éviter le chevauchement
            label,
            horizontalalignment=alignment,
            verticalalignment='center',
            size=9
        )
    axe.set_title("Précision par classe", size=15, y=1.08)
    axe.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    
        
