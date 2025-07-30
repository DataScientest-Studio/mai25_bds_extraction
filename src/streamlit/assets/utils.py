# UN MODULE POUR METTRE TOUTES LES FONCTIONS COMMUNES
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

from assets import PATHS, LABELS

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


def draw_spider_graph_dark(y_true, y_pred, title="Précision par classe", label_dict=None, save_path=None):
    n_classes = 16
    labels = list(range(n_classes))
    if label_dict:
        label_names = [label_dict[i] for i in labels]
    else:
        label_names = [str(i) for i in labels]

    precisions = precision_score(y_true, y_pred, average=None, labels=labels)
    precisions = np.append(precisions, precisions[0])  # boucle

    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False).tolist()
    angles += angles[:1]

    # Création figure + axe polar
    fig, axe = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 8))
    fig.patch.set_facecolor('black')
    axe.set_facecolor('black')

    # Couleurs
    line_color = '#4FC3F7'
    fill_color = '#4FC3F7'
    text_color = 'white'
    grid_color = '#888888'

    axe.plot(angles, precisions, 'o-', linewidth=2, color=line_color)
    axe.fill(angles, precisions, alpha=0.25, color=fill_color)

    axe.set_theta_offset(np.pi / 2)
    axe.set_theta_direction(-1)
    axe.set_ylim(0, 1)

    axe.set_xticks(angles[:-1])
    axe.set_xticklabels([])
    axe.set_thetagrids([])
    axe.set_rlabel_position(12)

    axe.tick_params(colors=text_color)
    axe.spines['polar'].set_color(grid_color)
    axe.grid(color=grid_color)

    for i in range(n_classes):
        angle_rad = angles[i]
        axe.text(
            angle_rad,
            1.15,
            label_names[i],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12,
            color=text_color
        )
        angle_rad = angles[i]
    
        # Ligne depuis le centre jusqu'au bord
        axe.plot([angle_rad, angle_rad], [0, 1], color=grid_color, linewidth=0.5, linestyle='dashed')

    axe.set_yticks([0.25, 0.5, 0.75, 1.0])
    axe.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], color=text_color, fontsize=8)
    axe.set_title(title, size=14, y=1.1, color=text_color)


    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, facecolor='black')
    else:
        plt.show()

    plt.close()
