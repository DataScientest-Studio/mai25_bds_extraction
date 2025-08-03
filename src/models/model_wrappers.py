import time
import json
import joblib
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any
from types import SimpleNamespace
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import sys
from pathlib import Path
from tensorflow.keras.applications.resnet50 import preprocess_input

project_root = Path().resolve().parent
if not project_root in [Path(p).resolve() for p in sys.path]:
    sys.path.append(str(project_root))

from src import PATHS

from src.visualization.visualize import visual_classification_report
from src.models.multimodal import MultiModalCompositeModel
from src.models.multimodal_clip import MultiModalCLIPBasedClassifier

documents = pd.read_parquet(PATHS.metadata / "df_data_sets.parquet")
data_sets = pd.read_parquet(PATHS.metadata / "df_data_sets.parquet")
labels = pd.read_parquet(PATHS.metadata / "df_encoded_labels.parquet")

class MODEL_KIND(Enum):
    TEXT = 1
    IMAGE = 2
    MULTIMODAL = 3

@dataclass
class PerformanceSummary:
    accuracy: float
    precisions: list[float]
    confusion_matrix: list[list[float]] # instead of np.array, for json saving
    inference_speed: float #(ms / document)

class ModelWrapper:
    """a prepretrained model wrapper
    Allow to have a unique interface, whatever the kind od model is used
    X is expected to be a list of document_ids
    preprocessed_X is expected to be a list of preprocessed id (will be directly provided to model prediction function)
    y is expected to be a list of encoded labels (integers inside [0,15] interval)
    """
    def __init__(
        self,
        name: str,
        kind: MODEL_KIND,
        model: Any,
        preprocessing_function: Callable,
        proba_prediction_function: Callable,
        performance_summary:PerformanceSummary = None
    ):
        self.name = name
        self.kind = kind
        self.model = model
        self.preprocessing_function = preprocessing_function
        self.proba_prediction_function = proba_prediction_function
        self._performance_summary = performance_summary

    
    def predict_proba(self, X=None, preprocessed_X=None):
        if preprocessed_X is None:
            preprocessed_X = self.preprocessing_function(X)
        result = self.proba_prediction_function(preprocessed_X)
        return result

    def predict(self, X=None, preprocessed_X=None):
        probas = self.predict_proba(X=X, preprocessed_X=preprocessed_X)
        return np.argmax(probas, axis=1)

    def prepreprocess(self):
        pass
    
    def _get_performance_summary(self):
        if self._performance_summary is None:
            performance_summaries_filepath = PATHS.models / "performance_summaries.json"
            if performance_summaries_filepath.exists():
                with open(performance_summaries_filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = dict()
            if self.name in data:
                self._performance_summary = PerformanceSummary(*data[self.name].values())
            else:
                # on calcule...
                X_test = documents[data_sets.data_set == "test"].index
                y_test = labels[data_sets.data_set == "test"].label
                t0 = time.time()
                y_preds = self.predict(X_test)
                tf = time.time()
                inference_speed = (tf - t0) / len(X_test)
                report = classification_report(y_test, y_preds, output_dict=True)
                precisions = [v['precision'] for v in [report[str(n)] for n in range(16)]]
                self._performance_summary = PerformanceSummary(
                    report['accuracy'],
                    precisions,
                    confusion_matrix(y_test, y_preds).tolist(),
                    inference_speed
                    )
                # et on écrit
                data[self.name] = self.performance_summary.__dict__
                with open(performance_summaries_filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f)
        return self._performance_summary
    
    performance_summary = property(_get_performance_summary)
    
    def visual_report(self):
        if isinstance(self.model, list): # on est dans le cas d'un Voteur
            compare_with = [
                [m.name, m.performance_summary]
                for m in self.model
            ]
        else:
            compare_with = []
        visual_classification_report(self.name, self.performance_summary, compare_with=compare_with)


        



df_ocr = pd.read_parquet(PATHS.processed_data / "df_txt_ocr1.parquet")
vectorizer = joblib.load(PATHS.models / "txt_tfid_vectorizer.joblib")

# region TEXT MAKERS
def make_txt_ml_model_wrapper(name, model_path):
    model = joblib.load(model_path)

    def preprocess(document_list):
        ocr = df_ocr.loc[document_list, "ocr"]
        vect = vectorizer.transform(ocr)
        return vect

    def predict_proba(preprocessed_X):
        return model.predict_proba(preprocessed_X)

    return ModelWrapper(
        name=name,
        kind=MODEL_KIND.TEXT,
        model=model,
        preprocessing_function=preprocess,
        proba_prediction_function=predict_proba
    )


def make_txt_dl_sklearn_model_wrapper(name, model_path):
    model = joblib.load(model_path)
    def preprocess(document_list):
        ocr = df_ocr.loc[document_list, "ocr"]
        vect = vectorizer.transform(ocr)
        return vect

    def predict_proba(preprocessed_X):
        return model.predict_proba(preprocessed_X)

    return ModelWrapper(
        name=name,
        kind=MODEL_KIND.TEXT,
        model=model,
        preprocessing_function=preprocess,
        proba_prediction_function=predict_proba
    )

def make_txt_dl_keras_model_wrapper(name, model_path):
    model = tf.keras.models.load_model(model_path)
    def preprocess(document_list):
        ocr = df_ocr.loc[document_list, "ocr"]
        vect = vectorizer.transform(ocr).toarray()
        return vect

    def predict_proba(preprocessed_X):
        return model.predict(preprocessed_X, verbose=0)

    return ModelWrapper(
        name=name,
        kind=MODEL_KIND.TEXT,
        model=model,
        preprocessing_function=preprocess,
        proba_prediction_function=predict_proba
    )

def make_txt_dl_model_wrapper(name, model_path):
    if str(model_path).endswith(".joblib"):
        return make_txt_dl_sklearn_model_wrapper(name, model_path)
    elif str(model_path).endswith(".keras"):
        return make_txt_dl_keras_model_wrapper(name, model_path)
    else:
        raise NotImplementedError("ModelWrapper must be created from '.joblib' or '.keras' model")


# region IMAGE MAKERS
df_image_features = pd.read_parquet(PATHS.processed_data / "df_img_features_pixels.parquet")
png_image_paths = (PATHS.data / pd.read_parquet(PATHS.metadata / "df_filepaths.parquet").rvl_image_path)\
                    .apply(lambda x:str(x).replace(str(PATHS.rvl_cdip_images), str(PATHS.converted_images))[:-4] + '.jpg')
img_preprocessor = joblib.load(PATHS.models / "img_ml_pipeline.joblib")

def make_img_ml_model_wrapper(name, model_path):
    model = joblib.load(model_path)
    def preprocess(document_list):
        features = df_image_features.loc[document_list, :]
        preprocessed = img_preprocessor.transform(features)
        return preprocessed

    def predict_proba(preprocessed_X):
        # Pour eviter un warning de LGBM:
        if hasattr(model, "feature_name_"):
            preprocessed_X = pd.DataFrame(preprocessed_X, columns=model.feature_name_)
        return model.predict_proba(preprocessed_X)

    return ModelWrapper(
        name=name,
        kind=MODEL_KIND.IMAGE,
        model=model,
        preprocessing_function=preprocess,
        proba_prediction_function=predict_proba
    )


def make_img_dl_model_wrapper(name, model_path):
    model = tf.keras.models.load_model(model_path)

    def preprocess_image(image_paths):
        if 'resnet50' in str(model._layers):
            image = tf.io.read_file(image_paths)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [224, 224])
            image = preprocess_input(image)
            return image
        elif 'vgg16' in str(model._layers):
            image = tf.io.read_file(image_paths)
            image = tf.image.decode_jpeg(image, channels=3) #parce que VGG16 attend 3 canaux
            image = tf.image.resize(image, [224, 224])  # taille attendue par VGG16
            image = image / 255.0  # Normalisation entre 0 et 1
            return image
        else:
            raise NotImplementedError("Unexpected model kind")
        return image

    def preprocess(document_list):
        paths = png_image_paths[document_list]
        file_paths = png_image_paths[document_list].values
        dataset = tf.data.Dataset.from_tensor_slices(file_paths)
        dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def predict_proba(preprocessed_X):
        return model.predict(preprocessed_X, verbose=0)

    return ModelWrapper(
        name=name,
        kind=MODEL_KIND.IMAGE,
        model=model,
        preprocessing_function=preprocess,
        proba_prediction_function=predict_proba
    )


# region MULTIMODAL MAKERS
class AGG_FN(Enum):
    AVERAGE = 1
    MAX = 2
    MIN = 3
    WEIGHTED = 4
    CLASS_WEIGHTED = 5


def make_mmo_voter_wrapper(name, model_wrappers, agg_fn = AGG_FN.AVERAGE, weights=None):
    match agg_fn:
        case AGG_FN.WEIGHTED:
            if weights is None:
                print("WARNING: Weights are None for WEIGHTED aggregation. Using uniform weights.")
                weights = np.ones(len(model_wrappers), dtype=np.float32)
            elif np.array(weights).shape != (len(model_wrappers),):
                raise ValueError("Length of weights must match number of models")
            weights_array = np.array(weights, dtype=np.float32)
            weights_array = weights_array / np.sum(weights_array)  # normalisation
            weights_array = weights_array.reshape(-1, 1, 1)  # broadcast (n_models, 1, 1)

        case AGG_FN.CLASS_WEIGHTED:
            if weights is None:
                print("WARNING: Weights are None for WEIGHTED aggregation. Using uniform weights.")
                weights = np.ones((len(model_wrappers), 16), dtype=np.float32)
            elif np.array(weights).shape != (len(model_wrappers), 16):
                raise ValueError("shape of weights must match (number of models, number of classes)")
            weights_array = np.array(weights, dtype=np.float32)
            weights_array = weights_array / np.sum(weights_array, axis=0, keepdims=True)
            weights_array = weights_array[:, np.newaxis, :]  # (n_models, 1, n_classes) — broadcastable

    def agg_function(prediction_list):
        predictions = np.array(prediction_list)
            # shape = len(model_wrappers), len(document_list), 16
        match agg_fn:
            case AGG_FN.AVERAGE:
                aggregated_predictions = predictions.mean(axis=0)
            case AGG_FN.MAX:
                aggregated_predictions = predictions.max(axis=0)
            case AGG_FN.MIN:
                aggregated_predictions = predictions.min(axis=0)
            case AGG_FN.WEIGHTED:            
                weighted_preds = predictions * weights_array
                return weighted_preds.sum(axis=0)
            case AGG_FN.CLASS_WEIGHTED:
                weighted_preds = predictions * weights_array  # (n_models, n_samples, n_classes)
                aggregated_predictions = weighted_preds.sum(axis=0)
            case _:
                raise ValueError("Unsupported aggregation function")
        return aggregated_predictions
    
    def predict_proba(document_list):
        predictions = [mw.predict_proba(document_list) for mw in model_wrappers]
            # len(model_wrappers) prédictions de shape: len(document_list), 16
        aggregated_predictions = agg_function(predictions)
            # prédictions de shape: len(document_list), 16
        return aggregated_predictions

    
    
    return ModelWrapper(
        name=name,
        kind=MODEL_KIND.MULTIMODAL,
        model=model_wrappers,
        preprocessing_function=lambda x:x,
        proba_prediction_function=predict_proba
    )


def make_mmo_composite_wrapper(name, model_file):
    model = MultiModalCompositeModel.load(model_file)
    
    def predict_proba(preprocessed_X):
        return model.predict_proba(preprocessed_X)

    return ModelWrapper(
        name=name,
        kind=MODEL_KIND.MULTIMODAL,
        model=model,
        preprocessing_function=lambda x:x,
        proba_prediction_function=predict_proba
    )

def make_mmo_clip_wrapper(name, model_file):
    model = MultiModalCLIPBasedClassifier.load(model_file)
    
    def predict_proba(preprocessed_X):
        return model.predict_proba(preprocessed_X)
    
    return ModelWrapper(
        name=name,
        kind=MODEL_KIND.MULTIMODAL,
        model=model,
        preprocessing_function=lambda x:x,
        proba_prediction_function=predict_proba
    )


# region MW FACTORY
class _ModelWrapperFactoryClass:
    _registry = [
        # maker_name,                    model_name,                                              model_filepath
        ['make_img_dl_model_wrapper',   'Image-based ResNet50 frozen',                          'img_ResNet50_best_50_epocs_sample_40_000.keras'],
        ['make_img_dl_model_wrapper',   'Image-based ResNet50 unfreezed',                       'img_ResNet50_best_50_epocs_sample_40_000_unfreeze_step_by_step.keras'],
        ['make_img_dl_model_wrapper',   'Image-based VGG16 frozen-10k only',                    'img_best_VGG16_30_epochs_sample_10000.keras'],
        ['make_img_dl_model_wrapper',   'Image-based VGG16 frozen-40k',                         'img_VGG16_best_50_epocs_sample_40_000.keras'],
        ['make_img_dl_model_wrapper',   'Image-based VGG16 unfreezed',                          'img_VGG16_best_progressive_unfreeze_40_000.keras'],
        ['make_img_ml_model_wrapper',   'Image-based LGBM',                                     'img_lgbm.joblib'],
        ['make_img_ml_model_wrapper',   'Image-based SGD',                                      'img_sgd.joblib'],
        ['make_img_ml_model_wrapper',   'Image-based XGBoost',                                  'img_xgboost.joblib'],
        ['make_txt_ml_model_wrapper',   'Text-based Logistic Regressor',                        'txt_logistic_regressor.joblib'],
        ['make_txt_ml_model_wrapper',   'Text-based Random Forest',                             'txt_random_forest.joblib'],
        ['make_txt_ml_model_wrapper',   'Text-based Naive Bayes',                               'txt_naive_bayes.joblib'],
        ['make_txt_dl_model_wrapper',   'Text-based MLP1',                                      'txt_mlp1.keras'],
        ['make_txt_dl_model_wrapper',   'Text-based MLP2',                                      'txt_mlp2.joblib'],
        ['make_txt_dl_model_wrapper',   'Text-based MLP3',                                      'txt_mlp3.joblib'],
        ['make_mmo_composite_wrapper',  'MMO-Composite LogReg on img-LGBM+txt-LogReg',          'mmo_comp_logreg_on_img-lgbm+txt-logreg.joblib'],
        ['make_mmo_clip_wrapper',       'MMO-CLIP-Based Logistic Regressor',                    'mmo_clip_logistic_regressor.joblib'],
        ['make_mmo_clip_wrapper',       'MMO-CLIP-Based MLP1',                                  'mmo_clip+mlp1.keras'],
        ['make_mmo_clip_wrapper',       'MMO-CLIP-Based MLP2',                                  'mmo_clip+mlp2.keras'],
        ['make_mmo_clip_wrapper',       'MMO-CLIP-Based MLP3',                                  'mmo_clip+mlp3.keras'],
        ['make_mmo_clip_wrapper',       'MMO-CLIP-Based MLP4',                                  'mmo_clip+mlp4.keras'],
        ['make_mmo_clip_wrapper',       'MMO-CLIP-Based MLP5',                                  'mmo_clip+mlp5.keras'],
        ['make_mmo_clip_wrapper',       'MMO-CLIP-Based MLP6',                                  'mmo_clip+mlp6.keras'],
        ['make_mmo_clip_wrapper',       'MMO-CLIP-Based MLP7',                                  'mmo_clip+mlp7.keras'],
        ['make_mmo_clip_wrapper',       'MMO-CLIP-Based MLP8',                                  'mmo_clip+mlp8.keras'],
        ['make_mmo_clip_wrapper',       'MMO-CLIP-Based MLP9',                                  'mmo_clip+mlp9.keras'],
        ['make_mmo_clip_wrapper',       'MMO-CLIP-Based MLP10',                                  'mmo_clip+mlp10.keras'],
        ['make_mmo_clip_wrapper',       'MMO-CLIP-Based MLP11',                                  'mmo_clip+mlp11.keras'],
        ['make_mmo_clip_wrapper',       'MMO-CLIP-Based MLP12',                                  'mmo_clip+mlp12.keras'],
    ]

    def __init__(self):
        self.make_img_dl_model_wrapper = make_img_dl_model_wrapper
        self.make_img_ml_model_wrapper = make_img_ml_model_wrapper
        self.make_txt_dl_model_wrapper = make_txt_dl_model_wrapper
        self.make_txt_ml_model_wrapper = make_txt_ml_model_wrapper
        self.make_mmo_voter_wrapper = make_mmo_voter_wrapper
        self.make_mmo_composite_wrapper = make_mmo_composite_wrapper
        self.make_mmo_clip_wrapper = make_mmo_clip_wrapper

    def load_existing(self, model_name):
        """Load a Wrapper from registry
        List of valid names can be accessed by ModelWrapperFactory.get_registered()
        """
        results = list(filter(lambda x: x[1] == model_name, self._registry))
        assert len(results) == 1, "Could not find model_name in regestry. Please check spelling in comparison to ModelWrapperFactory.get_registered() results."
        maker, name, file = results[0]
        model_wrapper = getattr(self, maker)(name, PATHS.models / file)
        return model_wrapper

    def get_registered(self):
        return [m[1] for m in self._registry]



ModelWrapperFactory = _ModelWrapperFactoryClass()