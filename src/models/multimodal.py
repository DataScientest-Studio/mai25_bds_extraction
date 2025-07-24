import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_score

class MultiModalVoter(BaseEstimator, ClassifierMixin):
    def __init__(self, img_model, img_pipeline, txt_model, txt_pipeline, method="average", weights=(0.5, 0.5)):
        self.img_model = img_model
        self.img_pipeline = img_pipeline
        self.txt_model = txt_model
        self.txt_pipeline = txt_pipeline
        self.method = method
        self.weights = weights

    def fit(self, X, y):
        print("This model will not fit")
        return self

    def predict_img_proba(self, X):
        X_img = X.drop(columns="ocr")
        proba_img = self.img_model.predict_proba(self.img_pipeline.transform(X_img))
        return proba_img

    def predict_txt_proba(self, X):
        X_txt = X[["ocr"]]
        proba_txt = self.txt_model.predict_proba(self.txt_pipeline.transform(X_txt))
        return proba_txt
    
    def predict_proba(self, X, mode="mmo"):
        if mode == "txt":
            return self.predict_txt_proba(X)
        if mode == "img":
            return self.predict_img_proba(X)
        proba_img = self.predict_img_proba(X)
        proba_txt = self.predict_txt_proba(X)
        if self.method == "average":
            return (proba_img + proba_txt) / 2
        elif self.method == "weighted":
            w1, w2 = self.weights
            return w1 * proba_img + w2 * proba_txt
        elif self.method == "max":
            return np.maximum(proba_img, proba_txt)
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")

    def predict(self, X, mode="mmo"):
        return np.argmax(self.predict_proba(X, mode=mode), axis=1)
    
class MultiModalClassWeightedVoter(BaseEstimator, ClassifierMixin):
    def __init__(self, img_model, img_pipeline, txt_model, txt_pipeline):
        self.img_model = img_model
        self.img_pipeline = img_pipeline
        self.txt_model = txt_model
        self.txt_pipeline = txt_pipeline
        # self.txt_weights = txt_weights.values.reshape((-1,16))

    def fit(self, X, y):
        y_pred = np.argmax(self.predict_img_proba(X), axis=1)
        self.img_weights = precision_score(y, y_pred, average=None).reshape((-1,16))
        y_pred = np.argmax(self.predict_txt_proba(X), axis=1)
        self.txt_weights = precision_score(y, y_pred, average=None).reshape((-1,16))
        self.weights_sum  = self.img_weights + self.txt_weights
        print("Model fitted")
        return self

    def predict_img_proba(self, X):
        X_img = X.drop(columns="ocr")
        proba_img = self.img_model.predict_proba(self.img_pipeline.transform(X_img))
        return proba_img

    def predict_txt_proba(self, X):
        X_txt = X[["ocr"]]
        proba_txt = self.txt_model.predict_proba(self.txt_pipeline.transform(X_txt))
        return proba_txt
    
    def predict_proba(self, X, mode="mmo"):
        if mode == "txt":
            return self.predict_txt_proba(X)
        if mode == "img":
            return self.predict_img_proba(X)
        if not hasattr(self, "img_weights"):
            print("Please fit the model first")
            return
        proba_img = self.predict_img_proba(X)
        proba_txt = self.predict_txt_proba(X)
        proba = (proba_img * self.img_weights + proba_txt * self.txt_weights) / self.weights_sum
        return proba
        
    def predict(self, X, mode="mmo"):
        if not hasattr(self, "img_weights"):
            print("Please fit the model first")
            return
        return np.argmax(self.predict_proba(X, mode=mode), axis=1)
    
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import numpy as np

class MultiModalLogisticRegressor(BaseEstimator, ClassifierMixin):
    def __init__(self, img_pipeline, img_model, txt_pipeline, txt_model):
        self.img_pipeline = img_pipeline
        self.img_model = img_model
        self.txt_pipeline = txt_pipeline
        self.txt_model = txt_model
        self.lr = LogisticRegression(max_iter = 1000, n_jobs=-1)            
        
    def fit(self, X, y):
        X_tr = np.concatenate((
            self.predict_img_proba(X),
            self.predict_txt_proba(X)
        ), axis=1)
        self.lr.fit(X_tr, y)
        print("Model fitted")
        return self

    def predict_img_proba(self, X):
        X_img = X.drop(columns="ocr")
        proba_img = self.img_model.predict_proba(self.img_pipeline.transform(X_img))
        return proba_img

    def predict_txt_proba(self, X):
        X_txt = X[["ocr"]]
        proba_txt = self.txt_model.predict_proba(self.txt_pipeline.transform(X_txt))
        return proba_txt
    
    def predict_proba(self, X, mode="mmo"):
        if mode == "img":
            return self.predict_img_proba(X)
        elif mode == "txt":
            return self.predict_txt_proba(X)
        proba_img = self.predict_img_proba(X)
        proba_txt = self.predict_txt_proba(X)
        proba_concatenated = np.concatenate((proba_img, proba_txt), axis=1)
        proba = self.lr.predict_proba(proba_concatenated)
        return proba
        
    def predict(self, X, mode="mmo"):
        return np.argmax(self.predict_proba(X, mode=mode), axis=1)