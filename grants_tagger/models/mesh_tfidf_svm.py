import pickle
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
import scipy.sparse as sp
import numpy as np

from grants_tagger.modes.utils import get_params_for_component


class MeshTfidfSVM():
    def __init__(self, y_batch_size=256, nb_labels=None, model_path=None,
            threshold=0.5):
        """
        y_batch_size: int, default 256. Size of column batches for Y i.e. tags that each classifier will train on
        nb_labels: int, default None. Number of tags that will be trained.
        model_path: path, default None. Model path being used to save intermediate classifiers
        threshold: float, default 0.5. Threshold probability on top of which a tag is assigned

        Note that model_path needs to be provided as it is used to save
        intermediate classifiers trained to reduce memory usage.
        """
        self.y_batch_size=y_batch_size
        self.model_path=model_path
        self.nb_labels=None
        self.threshold=threshold
    
    def _init_vectorizer(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english', max_df=0.95,
            min_df=5, ngram_range=(1,1)
        )

    def _init_classifier(self):
        self.classifier = OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='l2'))

    def set_params(self, **params):
        if not hasattr(self, 'vectorizer'):
            self._init_vectorizer()
        if not hasattr(self, 'classifier'):
            self._init_classifier()

        tfidf_params = get_params_for_component(params, 'tfidf')
        svm_params = get_params_for_component(params, 'svm')
        self.vectorizer.set_params(**tfidf_params)
        self.classifier.set_params(**svm_params)
        # TODO: Create function that checks in params for arguments available in init
        if 'model_path' in params:
            self.model_path = params['model_path']
        if 'y_batch_size' in params:
            self.y_batch_size = params['y_batch_size']
        if 'nb_labels' in params:
            self.nb_labels = params['nb_labels']

    def fit(self, X, Y):
        """
        X: list of texts
        Y: sparse csr_matrix of tags assigned
        """
        if not hasattr(self, 'vectorizer'):
            self._init_vectorizer()
        if not hasattr(self, 'classifier'):
            self._init_classifier()

        # TODO: Currently Y is expected to be sparse, otherwise predict does not
        # work, add a check and warn user.

        print(f"Creating {self.model_path}")
        Path(self.model_path).mkdir(exist_ok=True)
        print("Fitting vectorizer")
        self.vectorizer.fit(X)
        with open(f"{self.model_path}/vectorizer.pkl", "wb") as f:
            f.write(pickle.dumps(self.vectorizer))
        print("Training model")
        self.nb_labels = Y.shape[1]
        for tag_i in range(0, self.nb_labels, self.y_batch_size):
            print(tag_i)
            X_vec = self.vectorizer.transform(X)
            self.classifier.fit(X_vec, Y[:,tag_i:tag_i+self.y_batch_size])
            with open(f"{self.model_path}/{tag_i}.pkl", "wb") as f:
                f.write(pickle.dumps(self.classifier))

        return self

    def _predict(self, X, return_probabilities=False): 
        Y_pred = []
        for tag_i in range(0, self.nb_labels, self.y_batch_size):
            with open(f"{self.model_path}/{tag_i}.pkl", "rb") as f:
                classifier = pickle.loads(f.read())
            X_vec = self.vectorizer.transform(X)
            if return_probabilities:
                Y_pred_i = classifier.predict_proba(X_vec)
            elif self.threshold != 0.5:
                Y_pred_i = classifier.predict_proba(X_vec) > self.threshold
                Y_pred_i = sp.csr_matrix(Y_pred_i)
            else:
                Y_pred_i = classifier.predict(X_vec)
            Y_pred.append(Y_pred_i)

        if return_probabilities:
            Y_pred = np.hstack(Y_pred)
        else:
            Y_pred = sp.hstack(Y_pred)
        return Y_pred

    def predict(self, X):
        return self._predict(X)

    def predict_proba(self, X):
        return self._predict(X, return_probabilities=True)

    def save(self, model_path):
        if model_path != self.model_path:
            print(f"{model_path} is different from self.model_path {self.model_path}. This will result in model and meta.json be saved in different paths")

        meta = {
            "name": "MeshTfidfSVM",
            "approach": "mesh-tfidf-svm",
            "y_batch_size": self.y_batch_size,
            "nb_labels": self.nb_labels
        }
        meta_path = os.path.join(model_path, "meta.json")
        with open(meta_path, "w") as f:
            f.write(json.dumps(meta))

    def load(self, model_path):
        vectorizer_path = os.path.join(model_path, "vectorizer.pkl")
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.loads(f.read())

        meta_path = os.path.join(model_path, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.loads(f.read())
        self.set_params(**meta)

        self.model_path = model_path
