import os
import cv2
from tqdm import tqdm
import pandas as pd

BATCH_SIZE = 10000

project_path = "/Users/ben/Work/mle/ds-project/mai25_bds_extraction"
data_path = os.path.join(project_path, 'data')
output_base = os.path.join(data_path, 'converted')
extracted_data_path = os.path.join(data_path, 'extracted')
error_log_path = os.path.join(project_path, 'conversion_errors.txt')

# Chargement des chemins
df_documents = pd.read_parquet(os.path.join(extracted_data_path, "df_documents.parquet"), engine="fastparquet")
tif_paths = df_documents["rvl_image_path"].tolist()
total_images = len(tif_paths)

print(f"Nombre total d'images à traiter : {total_images}")

# Réinitialiser le fichier log d’erreurs
with open(error_log_path, 'w', encoding='utf-8') as f:
    f.write("Erreurs de conversion :\n")

# Traitement en batchs
for batch_start in range(0, total_images, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, total_images)
    current_batch = tif_paths[batch_start:batch_end]
    print(f"\nTraitement du batch {batch_start} → {batch_end - 1}...")

    for relative_path in tqdm(current_batch, desc=f"Batch {batch_start//BATCH_SIZE + 1}", unit="img"):
        input_path = os.path.join(data_path, relative_path)

        if not os.path.exists(input_path):
            with open(error_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"FICHIER INTROUVABLE : {input_path}\n")
            continue

        # Nettoyage du chemin pour output
        if relative_path.startswith("raw/"):
            cleaned_relative_path = relative_path[len("raw/"):]
        else:
            cleaned_relative_path = relative_path

        output_relative_path = os.path.splitext(cleaned_relative_path)[0] + ".jpg"
        output_path = os.path.join(output_base, output_relative_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)  # Convertit directement en RGB
            if img is None:
                raise ValueError("cv2.imread a renvoyé None")
            cv2.imwrite(output_path, img)
        except Exception as e:
            with open(error_log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"{input_path} → ERREUR : {e}\n")
