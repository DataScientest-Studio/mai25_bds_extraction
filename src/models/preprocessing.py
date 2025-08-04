import os
import re
import sys
import cv2
import time
import html
import hashlib
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
# initialisation des chemins
project_root = Path(__file__).resolve().parents[2]
if not project_root in [Path(p).resolve() for p in sys.path]:
    sys.path.append(str(project_root))

from src import PATHS

from scipy.ndimage import gaussian_filter, laplace
from PIL import Image, UnidentifiedImageError
from PIL.TiffTags import TAGS
from scipy.ndimage import gaussian_filter, laplace
from scipy.stats import entropy as scipy_entropy

#region TEXT PROCESSING

# IMPORTS
import jamspell
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import PunktSentenceTokenizer

import pytesseract
try:
    version = pytesseract.get_tesseract_version()
    print("Tesseract version:", version)
    print("pytesseract est bien configuré.")
except Exception as e:
    # EN CAS D'ERREUR
    # Si tesseract est bien installé mais ne se trouve pas dans le path, il est possible de définir son chemin avec cette commande:
    print("Tesseract n'a pas été trouvé en utilisant le PATH. Tentative avec la commande /opt/hombrew/bin/tesseract")
    try:
        pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract" # Chemin à mettre à jour le cas echeant
        version = pytesseract.get_tesseract_version()
        print("Tesseract version:", version)
        print("pytesseract est bien configuré.")
    except Exception as e:
        print("Erreur:", e)  

pg_regex = re.compile(r'pgNbr=[0-9]+')
def remove_pgNbr(text):
    text = pg_regex.sub('', text)
    return text

def unescape_html(text):
    if not text:
        return text
    return html.unescape(text)

jamspell_model_path = PATHS.models / 'jamspell' / 'en.bin'
assert jamspell_model_path.exists()
jamspell_model_path = str(jamspell_model_path)
corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel(jamspell_model_path)
def apply_jamspell(text):
    if not text:
        return text
    return corrector.FixFragment(text)

def basic_word_filter(text):
    if not text:
        return text
    text = text.lower()
    # Attention, c'est brutal, ca supprime tous les chiffres aussi...
    word_regex = re.compile(r'[a-z]{2,}')
    text = ' '.join(word_regex.findall(text))
    return text

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    # if pd.isnull(text): 
    #     return ""
    words = word_tokenize(text.lower())
    filtered = [word for word in words if word.isalpha() and word not in stop_words]
    return " ".join(filtered)

def text_preprocessing1(text, with_tqdm = False):
    # if not text:
    #     return text
    if isinstance(text, str):
        cleaned_text = remove_stopwords(
            basic_word_filter(
            apply_jamspell(
            unescape_html(
            remove_pgNbr(text)))))
        return cleaned_text
    else:
        assert isinstance(text[0], str), f"unexpected input in text_preprocessing (received data type {type(text)})"
        if with_tqdm:
            return [text_preprocessing1(t) for t in tqdm(text)]
        else:
            return [text_preprocessing1(t) for t in text]

#region IMAGE PROCESSING
def nettete(img):
    """
    Mesure la netteté via la variance du Laplacien, normalisée par le nombre de pixels.
    """
    try:
        lap = laplace(img)
        return np.var(lap) / img.size
    except Exception as e:
        print("Erreur dans nettete():", e)
        return None

def bruit(img):
    """
    Estime le bruit en comparant l'image à une version lissée avec un filtre gaussien.
    """
    sigma=1.0
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {file_path}")
    img = img.astype(np.float32)
    smoothed = cv2.GaussianBlur(img, (0, 0), sigma)
    noise = img - smoothed
    return np.std(noise)

def ratio_b(img):
    """
    Retourne le ratio de pixels dont la valeur est supérieure à 200 (pixels blancs).
    """
    white_pixels = np.sum(img > 200)
    return white_pixels / img.size

def ratio_n(img):
    """
    Retourne le ratio de pixels dont la valeur est inférieure à 50 (pixels noirs).
    """
    black_pixels = np.sum(img < 50)
    return black_pixels / img.size

def entropy(img):
    """
    Calcule l'entropie d'une image en niveaux de gris à partir d'un fichier image.

    L'entropie est une mesure statistique qui quantifie la quantité d'information
    ou la complexité présente dans l'image. Plus l'entropie est élevée, plus
    l'image contient de détails ou de variations.

    Cette fonction lit l'image en utilisant OpenCV en mode niveaux de gris,
    calcule son histogramme normalisé des intensités de pixel, puis calcule
    l'entropie de cette distribution.

    Paramètres:
    -----------
    file_path : str
        Chemin vers le fichier image à analyser.
    resize_max_dim : int ou None, optionnel (par défaut None)
        Si spécifié, redimensionne l'image pour que la plus grande dimension
        soit égale à cette valeur, afin d'accélérer le calcul sur les grandes images.

    Retour:
    --------
    float
        Valeur de l'entropie de l'image (en nats).

    """
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {file_path}")
    # Pas besoin de .load(), img est déjà un np.array
    hist, _ = np.histogram(img, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    return scipy_entropy(hist)

def compression(img):
    """
    Récupère les informations de compression d'une image.

    Retourne :
    - Le type de compression utilisé dans le fichier (ex: 'packbits', 'jpeg', None, etc.)
    - Un ratio de compression approximatif (taille fichier / taille brute image)

    Parameters
    ----------
    file_path : str
        Chemin vers le fichier image.

    Returns
    -------
    tuple (str or None, float)
        (compression_type, compression_ratio)
    """
    compression_type = img.info.get('compression', None)
    width, height = img.size
    channels = len(img.getbands())
    # Hypothèse : 1 octet par canal (ex: mode 'L', 'RGB')
    depth_bytes = 1
    uncompressed_size = width * height * channels * depth_bytes

    file_size = os.path.getsize(file_path)
    compression_ratio = file_size / uncompressed_size if uncompressed_size > 0 else None

    return compression_type, compression_ratio


def resolution(img):
    #Extrait la résolution (DPI) horizontale et verticale d'une image.
    #Retourne (dpi_x, dpi_y) si disponible, sinon (np.nan, np.nan).
    
    try:
        dpi = img.info.get('dpi')
        if isinstance(dpi, (tuple, list)) and len(dpi) == 2:
            return dpi[0], dpi[1]
    except Exception as e:
        pass  # ou log l'erreur si besoin

    return np.nan, np.nan

def marges(img):
    """
    Détecte les marges haut, bas, gauche, droite d'une image TIFF en mode L.

    Params:
        image_path (str): chemin vers l'image .tiff
    Returns:
        top, bottom, left, right: tailles des marges en pixels
    """
    # 1. récupérer l'image
    img_np = np.array(img)
    height, width = img_np.shape
    
    # Filtre médian pour réduire le bruit ponctuel
    denoised = cv2.medianBlur(img_np, ksize=5)  # ksize impair, typiquement 3 ou 5

    # Floutage gaussien pour lisser avant seuillage
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)

    # Binarisation adaptative sur image débruitée
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=15
    )
    
    # 4. Projections
    rows_sum = binary.sum(axis=1)
    cols_sum = binary.sum(axis=0)
    # Filtrer les lignes à analyser
    top_bottom_ignore = 30
    side_ignore=50 
    # On enlève les bords parce qu'il peut y avoir des traits noirs dus à un scan de mauvaise qualité
    valid_rows = rows_sum[top_bottom_ignore:height - top_bottom_ignore]
    valid_cols = cols_sum[side_ignore:width - side_ignore]

    if valid_rows.max() == 0 or valid_cols.max() == 0:
        # Aucun texte détecté
        return 0, height, 0, width

    # Marges haut/bas (avec décalage)
    top = np.argmax(valid_rows > 0) + top_bottom_ignore
    bottom = height - top_bottom_ignore - np.argmax(valid_rows[::-1] > 0)

    # Marges gauche/droite (avec décalage)
    left = np.argmax(valid_cols > 0) + side_ignore
    right = width - side_ignore - np.argmax(valid_cols[::-1] > 0)

    return top, bottom,  left, right


#pour récupérer le nombre de lignes
def merge_close_lines(lines, min_distance):
    """
    Fusionne les lignes trop proches les unes des autres.
    """
    if not lines:
        return []

    merged = [lines[0]]
    
    for start, end in lines[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < min_distance:
            # Fusionner avec la précédente
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def show_detection_nb_lignes(img, show=True, debug=False):
    """
    Détecte et (optionnellement) affiche les lignes de texte dans une image en niveaux de gris.

    Params:
        img (ndarray): Image (déjà rognée) en niveaux de gris.
        show (bool): Affiche l’image avec les lignes détectées si True.
        debug (bool): Affiche aussi le profil de projection si True.

    Returns:
        lines (List of tuples): liste de tuples (start_row, end_row) pour chaque ligne détectée.
    """
    img_np = np.array(img)

    # Binarisation (texte noir sur fond blanc)
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_inv = 255 - binary  # Texte = 255 maintenant

    # Morphologie verticale
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    eroded = cv2.erode(binary_inv, kernel, iterations=1)

    # Profil de projection verticale
    projection = np.sum(eroded // 255, axis=1)

    # Seuil basé sur le percentile 95
    max_proj = np.percentile(projection, 95)
    threshold = max_proj * 0.25

    # Détection des lignes
    lines = []
    in_line = False
    start = 0

    for i, val in enumerate(projection):
        if val > threshold:
            if not in_line:
                start = i
                in_line = True
        else:
            if in_line:
                end = i
                lines.append((start, end))
                in_line = False
    if in_line:
        lines.append((start, len(projection)))

    # Fusionner les lignes trop proches
    lines = merge_close_lines(lines, min_distance=4)

    # Affichage (si demandé)
    if show:
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.imshow(img_np, cmap='gray')
        for start, end in lines:
            rect = plt.Rectangle((0, start), img_np.shape[1], end - start,
                                 linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        ax.set_title(f"Lignes détectées : {len(lines)}")
        ax.axis('off')
        plt.show()

    if debug:
        plt.figure(figsize=(8, 2))
        plt.plot(projection)
        plt.axhline(y=threshold, color='red', linestyle='--')
        plt.title("Projection verticale des pixels noirs (profil)")
        plt.show()

    return lines

def show_detection_nb_colonnes(img, show=True, debug=False):
    """
    Détecte et (optionnellement) affiche les colonnes de texte dans une image (numpy, niveaux de gris).

    Params:
        img (ndarray): Image en niveaux de gris (déjà rognée).
        show (bool): Affiche l’image avec les colonnes détectées si True.
        debug (bool): Affiche le profil de projection horizontal si True.

    Returns:
        columns (List of tuples): liste de tuples (start_col, end_col) pour chaque colonne détectée.
    """
    img_np = np.array(img)

    # Binarisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img_np)
    _, binary = cv2.threshold(img_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_inv = 255 - binary  # texte = 255

    # Morphologie horizontale pour séparer les colonnes (éventuellement)
    kernel_width = max(10, img_eq.shape[1] // 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    eroded = cv2.erode(binary_inv, kernel, iterations=1)

    # Projection horizontale (sur les colonnes)
    window = 50
    proj = np.sum(eroded // 255, axis=0)
    projection = np.convolve(proj, np.ones(window)/window, mode='same')
    

    # Seuil pour distinguer une colonne active, selon le contraste
    contrast = np.std(img_eq)
    if contrast < 43:  # image très pâle
        factor = 0.8
    elif contrast < 70:  # contraste moyen
        factor = 1.4
    else:  # bon contraste
        factor = 2
    
    mean_proj = np.mean(projection)
    median_proj = np.median(projection)
    threshold = min(mean_proj, median_proj) * factor
    
    columns = []
    in_column = False
    start = 0

    for i, val in enumerate(projection):
        if val > threshold:
            if not in_column:
                start = i
                in_column = True
        else:
            if in_column:
                end = i
                columns.append((start, end))
                in_column = False
    if in_column:
        columns.append((start, len(projection)))

    # Fusionner les colonnes trop proches (si nécessaire)
    min_dist = img_np.shape[1] // 40
    columns = merge_close_lines(columns, min_distance=min_dist)
    #columns = merge_close_lines(columns, min_distance=20)

    # Enlever les colonnes trop fines
    min_width = 30 #pixels
    columns = [(start, end) for (start, end) in columns if (end - start) > min_width]
    
    # Affichage (si demandé)
    if show:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img_np, cmap='gray')
        for start, end in columns:
            rect = plt.Rectangle((start, 0), end - start, img_np.shape[0],
                                 linewidth=1, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
        ax.set_title(f"Colonnes détectées : {len(columns)}")
        ax.axis('off')
        plt.show()

    if debug:
        plt.figure(figsize=(8, 2))
        plt.plot(projection)
        plt.axhline(y=threshold, color='blue', linestyle='--')
        plt.title("Projection horizontale des pixels noirs (profil)")
        plt.show()

    return columns

def get_features(img):
#    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = np.array(img)
    try:
        Nettete = nettete(img)
        Bruit = bruit(img)
        Ratio_b = ratio_b(img)
        Ratio_n = ratio_n(img)
        Entropie = entropy(img)
        top, bottom, left, right = marges(img)
        # Recadrer l'image selon les marges
        cropped = img[top:bottom, left:right]
        img_np = np.array(cropped)
        height, width = img_np.shape
        # Filtre médian pour réduire le bruit ponctuel
        denoised = cv2.medianBlur(img_np, ksize=3)  # ksize impair, typiquement 3 ou 5
        # Floutage gaussien pour lisser
        blurred = cv2.GaussianBlur(denoised, (3,3), 0)
        lines = show_detection_nb_lignes(blurred, show=False)
        nb_lignes = len(lines)
        colonnes = show_detection_nb_colonnes(denoised, show=False)
        nb_colonnes = len(colonnes)
        width = img.shape[1] 
#    except (UnidentifiedImageError, FileNotFoundError, OSError):
    except (FileNotFoundError, OSError):
        Nettete = Bruit = Ratio_b = Ratio_n = Entropie = Compression = width  = top = bottom = left = right = nb_lignes = nb_colonnes  = np.nan
    return [
        top , bottom , left , right,
        nb_lignes, nb_colonnes,
        Nettete, Bruit, Ratio_b, Ratio_n, Entropie, width]


def get_10K_pix(img):
    try:
        # Étape 1 : Rogner
        top, bottom, left, right = marges(img)
        cropped = img[top:bottom, left:right]

        # Étape 2 : Redimensionner proportionnellement
        h, w = cropped.shape
        if h > w:
            new_h = 100
            new_w = int(w * 100 / h)
        else:
            new_w = 100
            new_h = int(h * 100 / w)
        resized = cv2.resize(cropped, (new_w, new_h))

        # Étape 3 : Créer un fond blanc 100x100
        canvas = np.full((100, 100), 255, dtype=np.uint8)

        # Étape 4 : Coller au centre
        y_offset = 0  # image collée en haut
        x_offset = (100 - new_w) // 2  # centrage horizontal
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas.flatten()
    except Exception as e:
        print(f"Erreur traitement: {e}")

#region MAIN
def generate_image_id(image_path, prefix="", length=9):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    hash_hex = hashlib.sha1(image_bytes).hexdigest()
    return f"{prefix}_{hash_hex[:length]}x00"

def extract_features(image_path) -> None:

    img = Image.open(image_path).convert("L")  # RVL = noir et blanc
    documents = pd.read_parquet(PATHS.metadata / "df_documents.parquet")

    w, h = img.size
    scale = max(w, h) / max_dim if max(w, h) > 1000 else 1
    new_size = (int(w / scale), int(h / scale))
    img = img.resize(new_size)

    img_id = generate_image_id(image_path)
    if img_id in documents.index:
        print(f"{image_path} already imported. Skipping")
        return
    img_name = Path(image_path).stem + ".png" 
    rvl_path = PATHS.other_images / img_id / img_name
    converted_path = Path(str(rvl_path).replace(str(PATHS.rvl_cdip_images), str(PATHS.converted_images))[:-4]+'.jpg') # redondant ici mais permet de garder la structure rvl-cdip
    for p in [rvl_path, converted_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
    
    new_id_df = pd.DataFrame([], index=[img_id], )
    new_id_df.index.name = "document_id"
    documents = pd.concat((documents,new_id_df))
    documents.to_parquet(PATHS.metadata / "df_documents.parquet")
    print(f"{(PATHS.metadata / "df_documents.parquet")} saved")

    df_filepaths = pd.read_parquet(PATHS.metadata / "df_filepaths.parquet")
    # cols = ['filename', 'rvl_image_path', 'iit_image_path', 'iit_individual_xml_path', 'iit_collective_xml_path']
    row = [img_name, str(rvl_path.relative_to(PATHS.data)), "", "", ""]
    df_filepaths.loc[img_id] = row
    df_filepaths.to_parquet(PATHS.metadata / "df_filepaths.parquet")
    print(f"{(PATHS.metadata / "df_filepaths.parquet")} saved")

    raw_text = pytesseract.image_to_string(img)
    # print("extracted following raw text")
    # print(raw_text)
    df_raw_ocr = pd.read_parquet(PATHS.processed_data / "df_raw_ocr.parquet")
    df_raw_ocr.loc[img_id] = [raw_text]
    df_raw_ocr.to_parquet(PATHS.processed_data / "df_raw_ocr.parquet")
    print(f"{(PATHS.processed_data / "df_raw_ocr.parquet")} saved")
    del df_raw_ocr

    ocr1 = text_preprocessing1([raw_text])[0]
    # print("extracted following pre-processed text")
    # print(ocr1)
    df_ocr1 = pd.read_parquet(PATHS.processed_data / "df_txt_ocr1.parquet")
    df_ocr1.loc[img_id] = [ocr1]
    df_ocr1.to_parquet(PATHS.processed_data / "df_txt_ocr1.parquet")
    print(f"{(PATHS.processed_data / "df_txt_ocr1.parquet")} saved")
    del df_ocr1

    row = get_features(img) + get_10K_pix(np.array(img)).tolist()
    # print("extracted image-related features")
    df_features_pixels = pd.read_parquet(PATHS.processed_data / "df_img_features_pixels.parquet")
    df_features_pixels.loc[img_id] = row
    df_features_pixels.to_parquet(PATHS.processed_data / "df_img_features_pixels.parquet")
    print(f"{(PATHS.processed_data / "df_img_features_pixels.parquet")} saved")
    del df_features_pixels

    print(f"Successfully imported image {image_path} with id {img_id}")


def reset_to_rvl_state():
    rvl_ids = pd.read_parquet(PATHS.metadata / "df_documents_save.parquet").index
    for df_path in [
        PATHS.metadata / "df_documents.parquet",
        PATHS.metadata / "df_data_sets.parquet",
        PATHS.metadata / "df_plain_labels.parquet",
        PATHS.metadata / "df_encoded_labels.parquet",
        PATHS.metadata / "df_filepaths.parquet",

        PATHS.processed_data / "df_raw_ocr.parquet",
        PATHS.processed_data / "df_txt_ocr1.parquet",
        PATHS.processed_data / "df_img_features_pixels.parquet",
        PATHS.processed_data / "df_clip_embeddings.parquet",
        
        
    ]:
        print(df_path, end="...")
        df = pd.read_parquet(df_path)
        df = df.loc[rvl_ids,:]
        assert len(df) == len(rvl_ids)
        df.to_parquet(df_path)
        print(" sucessfully reset")
        rvl_path = PATHS.other_images
        converted_path = Path(str(rvl_path).replace(str(PATHS.rvl_cdip_images), str(PATHS.converted_images)))
    for folder in [str(rvl_path), str(converted_path)]:
        try:
            print(str(folder), end="...")
            shutil.rmtree(folder)
            print(" sucessfully removed")
        except Exception as e:
            print(f"Erreur lors de la suppression de {folder}: {e}")    