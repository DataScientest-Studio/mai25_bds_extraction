import os
import time
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score


from src import PATHS, LABELS

def plot_confusion_matrix(cm, axe):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axe)
    axe.set_title("Matrice de confusion")
    axe.set_xlabel("Classe prédite")
    axe.set_ylabel("Classe réelle")
    
def plot_perf_summary(accuracy, speed, axe):
    axe.text(0.5, 0.7, f"Accuracy: {accuracy*100:.2f}%", ha="center", fontsize = 20)
    axe.text(0.5, 0.3, f"Speed: {speed*1000:.2f} ms/image", ha="center", fontsize = 16)
    axe.set_axis_off()

def plot_spider_graph(precisions, model_names, axe):
    plain_labels = list(LABELS.values())
    angles = np.linspace(0, 2*np.pi, len(plain_labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    
    for precision, model_name in zip(precisions, model_names):
        values = precision + [precision[0]]
        axe.plot(angles, values, '.-', linewidth=1, label=model_name)
        axe.fill(angles, values, alpha=0.25)
        
    axe.set_theta_offset(np.pi / 2)
    axe.set_theta_direction(-1)
    axe.set_ylim(0, 1)
    axe.set_rlabel_position(0)
    axe.grid(True)
    
    # 🔧 Placer manuellement les labels à l’extérieur du cercle
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
    
    # Titre et légende
    axe.set_title("Précision par classe", size=15, y=1.08)
    axe.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    

                      
def visual_classification_report(model_name, performance_summary, compare_with = []):
    """
    plot a visual_report of a model with a spider graph on the left and its confusion matrix on the right.
    spider graph can be augmented by other models that should be passed across compare_with attribute as tuples (model_name, precisions)
    """
    fig = plt.figure(figsize=(20,10))
    gs = GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[1, 1.5])

    text_ax = fig.add_subplot(gs[0, 0])
    spider_ax = fig.add_subplot(gs[1, 0], projection='polar')
    cm_ax = fig.add_subplot(gs[:, 1])
    
    model_names = [model_name]
    precisions = [performance_summary.precisions]
    confusion_matrix = performance_summary.confusion_matrix

    for other_name, other_performance_summary in compare_with:
        model_names.append(other_name)
        precisions.append(other_performance_summary.precisions)

    plot_spider_graph(precisions, model_names, spider_ax)
    plot_confusion_matrix(confusion_matrix, cm_ax)
    plot_perf_summary(
        performance_summary.accuracy,
        performance_summary.inference_speed,
        text_ax
        )
    plt.suptitle(model_name, fontsize=20)
    plt.show()
    
def plot_history(history, title_prefix="Model"):
    sns.set(style="whitegrid")
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    plt.figure(figsize=(12, 5))

    # --- Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict['loss'], label='Train Loss', color='tab:blue', linewidth=2)
    if 'val_loss' in history_dict:
        plt.plot(epochs, history_dict['val_loss'], label='Validation Loss', color='tab:orange', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"{title_prefix} Loss per Epoch", fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Accuracy ---
    plt.subplot(1, 2, 2)
    if 'accuracy' in history_dict:
        plt.plot(epochs, history_dict['accuracy'], label='Train Accuracy', color='tab:blue', linewidth=2)
    if 'val_accuracy' in history_dict:
        plt.plot(epochs, history_dict['val_accuracy'], label='Validation Accuracy', color='tab:orange', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(f"{title_prefix} Accuracy per Epoch", fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()








# si on veut des graphes comparatifs
# a terminer le cas échéant
# def visual_classification_report(model, X_eval, y_eval, model_name="", models_to_compare = []):
    # if not model_name:
    #     model_name =  model.__class__.__name__
    
    # fig = plt.figure(figsize=(20,10))
    # gs = GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[1, 1.5])

    # text_ax = fig.add_subplot(gs[0, 0])
    # spider_ax = fig.add_subplot(gs[1, 0], projection='polar')
    # cm_ax = fig.add_subplot(gs[:, 1])
    
    # t0 = time.time()
    # y_pred = model.predict(X_eval)
    # predict_time = time.time() - t0
    # predict_speed = predict_time / len(X_eval)

    # accuracy = accuracy_score(y_eval, y_pred)

    # y_preds = [y_pred]
    # model_names = [model_name]
    # for name, predict_function in models_to_compare:
    #     model_names.append(name)
    #     y_preds.append(predict_function(X_eval))
    # plot_spider_graph(y_eval, y_preds, model_names, spider_ax)
    # plot_confusion_matrix(y_eval, y_pred, cm_ax)
    # plot_perf_summary(accuracy, predict_speed, text_ax)

    # plt.suptitle(model_name, fontsize=20)
    # plt.show()




def draw_spider_graph_dark(y_true, y_pred, title="Précision par classe", save_path=None):
    n_classes = 16
    label_dict= {
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

def conf_matrix_dark(cm, save_path):
    plt.figure(figsize=(10, 8), facecolor='black')

    # Créer le heatmap sans annotations
    ax = sns.heatmap(
        cm,
        annot=False,
        fmt='d',
        cmap='Blues_r',
        cbar=False,
        linewidths=0.5,
        linecolor='gray'
    )
    
    # Ajouter les annotations manuellement
    threshold = cm.max() / 2  # Seuil pour changer la couleur du texte
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = 'black' if value > threshold else 'white'
            ax.text(j + 0.5, i + 0.5, value,
                    ha='center', va='center',
                    color=color, fontsize=12)
    
    # Ajustements esthétiques
    plt.title("Matrice de confusion - Validation", color='white', fontsize=16)
    plt.xlabel("Classe prédite", color='white')
    plt.ylabel("Classe réelle", color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=True)
    plt.close()
    