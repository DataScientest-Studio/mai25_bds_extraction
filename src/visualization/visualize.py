import os
import time
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score

from src import PATHS

def plot_confusion_matrix(y_eval, y_pred, axe):
    cm = confusion_matrix(y_eval, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axe)
    axe.set_title("Matrice de confusion")
    axe.set_xlabel("Classe prédite")
    axe.set_ylabel("Classe réelle")
    
def plot_perf_summary(accuracy, speed, axe):
    axe.text(0.5, 0.7, f"Accuracy: {accuracy*100:.2f}%", ha="center", fontsize = 20)
    axe.text(0.5, 0.3, f"Speed: {speed*1000:.2f} ms/image", ha="center", fontsize = 16)
    axe.set_axis_off()

def plot_spider_graph(y_eval, y_preds, model_names, axe):
    plain_labels = pd.read_parquet(os.path.join(PATHS.metadata, "df_labels_mapping.parquet")).plain_label.values
    angles = np.linspace(0, 2*np.pi, len(plain_labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    
    for y_pred, model_name in zip(y_preds, model_names):    
        precisions = precision_score(y_eval, y_pred, average=None).tolist() #, labels=list(range(16)))
        precisions.append(precisions[0])
        axe.plot(angles, precisions, '.-', linewidth=1, label=model_name)
        axe.fill(angles, precisions, alpha=0.25)
        
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
    

                      
def visual_classification_report(model, X_eval, y_eval, model_name="", compare_with_components = False):
    if not model_name:
        model_name =  model.__class__.__name__
    
    fig = plt.figure(figsize=(20,10))
    gs = GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[1, 1.5])

    text_ax = fig.add_subplot(gs[0, 0])
    spider_ax = fig.add_subplot(gs[1, 0], projection='polar')
    cm_ax = fig.add_subplot(gs[:, 1])
    
    t0 = time.time()
    y_pred = model.predict(X_eval)
    predict_time = time.time() - t0
    predict_speed = predict_time / X_eval.shape[0]

    accuracy = accuracy_score(y_eval, y_pred)

    y_preds = [y_pred]
    model_names = [model_name]
    if compare_with_components:
        models_to_compare = [
            ["image model", lambda x: np.argmax(model.predict_proba(x, mode="img"), axis=1)],
            ["text model", lambda x: np.argmax(model.predict_proba(x, mode="txt"), axis=1)]
        ]
    else:
        models_to_compare = []
    for name, predict_function in models_to_compare:
        model_names.append(name)
        y_preds.append(predict_function(X_eval))
    plot_spider_graph(y_eval, y_preds, model_names, spider_ax)
    plot_confusion_matrix(y_eval, y_pred, cm_ax)
    plot_perf_summary(accuracy, predict_speed, text_ax)

    plt.suptitle(model_name, fontsize=20)
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
    