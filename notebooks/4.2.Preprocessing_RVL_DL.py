import os
import sys
import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path

project_root = Path().resolve().parent
if not project_root in [Path(p).resolve() for p in sys.path]:
    sys.path.append(str(project_root))

from src import PATHS

BATCH_SIZE = 10000

df_documents = PATHS.data / pd.read_parquet(PATHS.metadata / "df_filepaths.parquet")[["rvl_image_path"]] 
tif_paths = df_documents.rvl_image_path.map(str).to_list()
jpg_paths = [p.replace(str(PATHS.raw_images), str(PATHS.converted_images))[:-4] + '.jpg' for p in tif_paths]
total_images = len(tif_paths)
print(f"Nombre total d'images à traiter : {total_images}")

error_log_path = PATHS.project / 'conversion_errors.txt'
# Réinitialiser le fichier log d’erreurs
with open(error_log_path, 'w', encoding='utf-8') as f:
    f.write("Erreurs de conversion :\n")

# Traitement en batchs
for batch_start in range(0, total_images, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, total_images)
    current_batch = zip(tif_paths[batch_start:batch_end], jpg_paths[batch_start:batch_end])
    print(f"\nTraitement du batch {batch_start} → {batch_end - 1}...")

    for tif_path, jpg_path in tqdm(current_batch, desc=f"Batch {batch_start//BATCH_SIZE + 1}", unit="img"):
        if not os.path.exists(tif_path):
            with open(error_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"FICHIER INTROUVABLE : {tif_path}\n")
            continue
        os.makedirs(os.path.dirname(jpg_path), exist_ok=True)

        try:
            img = cv2.imread(tif_path, cv2.IMREAD_COLOR)  # Convertit directement en RGB
            if img is None:
                raise ValueError("cv2.imread a renvoyé None")
            cv2.imwrite(jpg_path, img)
        except Exception as e:
            with open(error_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"{tif_path} → ERREUR : {e}\n")



    