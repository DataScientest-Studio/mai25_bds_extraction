import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
import torch
import joblib
import tensorflow as tf
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



import sys
from pathlib import Path

project_root = Path().resolve().parent
if not project_root in [Path(p).resolve() for p in sys.path]:
    sys.path.append(str(project_root))

from src import PATHS


class CLIP:
    """A CLIP model, dealt as a singleton to limit memory impact"""
    _instance = None

    # Singleton class to save memory (CLIP + embeddings pre-computing)
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CLIP, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # lazy loading pour les embeddings rvl precalcules
        columns = [f"img_{i:03d}" for i in range(1, 513)]+[f"txt_{i:03d}" for i in range(1, 513)]
        self._embeddings = pd.DataFrame(columns = columns)
        self._embeddings_were_augmented = False
        self.clip_saved_embeddings_path = PATHS.processed_data / "df_clip_embeddings.parquet"
        image_paths = PATHS.data / pd.read_parquet(PATHS.metadata / "df_filepaths.parquet").rvl_image_path
        self.image_paths = image_paths.apply(
            lambda x: str(x).replace(str(PATHS.raw_images), str(PATHS.converted_images))[:-4] + '.jpg')
        self.ocrs = pd.read_parquet(PATHS.processed_data / "df_txt_ocr1.parquet").ocr

    def get_embeddings(self, document_ids):
        """Return clip embeddings from a list of document ids
        Use lazy loading of previously saved embeddings
        returned value is a dataframe of shape (len(document_ids), 1024)
        """
        if len(self._embeddings) == 0:
            try:
                self._embeddings = pd.read_parquet(self.clip_saved_embeddings_path)
            except Exception:
                print("No pre-computed CLIP embeddings found. You should save them at the end using self.save_embeddings()")
        missing_embedding_ids = [d for d in document_ids if d not in self._embeddings.index]
        if missing_embedding_ids:
            # Computation using CLIP preprocessor
            texts = self.ocrs[missing_embedding_ids].values.tolist()
            images = self.image_paths[missing_embedding_ids].apply(
                lambda x: Image.open(x).convert("RGB")
            ).values.tolist()
            clip_preprocessed = self.processor(
                        text=texts,
                        images=images,
                        return_tensors="pt", padding=True,
                        truncation=True,  # tronque automatiquement à la taille max supportée
                    )
            clip_preprocessed = {k: v.to(self.device) for k, v in clip_preprocessed.items()}
            with torch.no_grad():
                outputs = self.model(**clip_preprocessed)
                image_emb = outputs.image_embeds.cpu().numpy()
                text_emb = outputs.text_embeds.cpu().numpy()
            data = np.concatenate((image_emb, text_emb), axis=1)
            columns = [f"img_{i:03d}" for i in range(1, 513)]+[f"txt_{i:03d}" for i in range(1, 513)]
            additionnal_embeddings = pd.DataFrame(data, columns=columns, index=missing_embedding_ids)
            if len(self._embeddings) > 0: # evite un warning concatenation avec DF vide
                self._embeddings = pd.concat([self._embeddings, additionnal_embeddings])
            else:
                self._embeddings = additionnal_embeddings
            self._embeddings_were_augmented = True
        return self._embeddings.loc[document_ids,:]

    def save_embeddings(self):
        if not self._embeddings_were_augmented:
            print("No further embedding was computed. No save performed")
        else:
            self._embeddings.to_parquet(self.clip_saved_embeddings_path)
            print(f"Embeddings saved to {str(self.clip_saved_embeddings_path)}")


class MultiModalCLIPBasedClassifier:
    def __init__(self, classifier=None):
        self.clip = CLIP()
        self.classifier = classifier or LogisticRegression(max_iter=1000, n_jobs=-1)
    

    def fit(self, document_ids, y_train, embeddings=None, **kwargs):
        if embeddings is None:
            embeddings = self.clip.get_embeddings(document_ids)
        if "validation_data" in kwargs.keys():
            document_ids_val, y_val = kwargs["validation_data"]
            embeddings_val = self.clip.get_embeddings(document_ids_val)
            kwargs["validation_data"] = [embeddings_val, y_val]
        return self.classifier.fit(embeddings, y_train, **kwargs)


    
    def predict(self, document_ids):
        preprocessed_X = self.clip.get_embeddings(document_ids)
        return self.classifier.predict(preprocessed_X)
    
    def predict_proba(self, document_ids):
        preprocessed_X = self.clip.get_embeddings(document_ids)
        if hasattr(self.classifier, 'predict_proba'):           # on est sur un modele sklearn
            probas = self.classifier.predict_proba(preprocessed_X)
        else:
            probas = self.classifier.predict(preprocessed_X)    # on est sur un modele keras
        return probas


        preprocessed_X = self.clip.get_embeddings(document_ids)
        return self.classifier.predict_proba(preprocessed_X)
    
    def evaluate(self, document_ids, y_true):
        y_pred = self.predict(document_ids)
        report = classification_report(y_true, y_pred)
        print(report)
        return report
        
    def save(self, file_path):
        # on ne sauvegare que le classifieur associé, avec 2 possibilités: keras ou joblib
        if hasattr(self.classifier, "save"):
            assert str(file_path).endswith(".keras")
            self.classifier.save(file_path)
        elif hasattr(self.classifier, "get_params"):
            assert str(file_path).endswith(".joblib")
            joblib.dump(self.classifier, file_path)
        else:
            raise NotImplementedError("Unable to detect model kind (keras vs sklearn)")

    @classmethod
    def load(cls, file_path):
        if str(file_path).endswith(".joblib"):
            model = joblib.load(file_path)
        elif str(file_path).endswith(".keras"):
            model = tf.keras.models.load_model(file_path)
        else:
            raise NotImplementedError("Unable to detect model kind (keras vs sklearn)")
        return cls(classifier=model)