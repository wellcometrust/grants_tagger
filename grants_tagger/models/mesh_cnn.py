import json
import os

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from wellcomeml.ml.cnn import CNNClassifier
from wellcomeml.ml.keras_vectorizer import KerasVectorizer
from grants_tagger.models.utils import get_params_for_component
from grants_tagger.utils import save_pickle, load_pickle


class MeshCNN:
    def __init__(
        self,
        threshold=0.5,
        batch_size=256,
        shuffle=True,
        buffer_size=1000,
        data_cache=None,
        random_seed=42,
        cutoff_prob=0.1,
    ):
        """
        threshold: float, default 0.5. Probability threshold on top of which a tag should be assigned.
        batch_size: int, default 256. Size of batches used for training and prediction.
        shuffle: bool, default True. Flag on whether to shuffle data before fit.
        buffer_size: int, default 1000. Buffer size used for shuffling or transforming data before fit.
        data_cache: path, default None. Path to use for caching data transformations.
        random_seed: int, default 42. Random seed that controls reproducibility.
        cutoff_prob: float, default 0.1. Prob below which we zero the probs to create sparse probs
        """
        self.threshold = threshold
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.data_cache = data_cache
        self.random_seed = random_seed
        self.cutoff_prob = cutoff_prob

    def _yield_data(self, X, vectorizer, Y=None):
        """
        Generator to yield vectorized X and Y data one by one

        X: list of texts
        vectorizer: vectorizer class that implements transform which transforms texts to integers
        Y: 2d numpy array or sparse csr_matrix that represents targets (tags) assigned.

        If Y is missing, for example when called by predict, yield_data yields only X vectorized
        """

        def yield_transformed_data(X_buffer, Y_buffer):
            # TODO: This could move to WellcomeML to enable CNN to receive generators
            Y_den = None
            X_vec = self.vectorizer.transform(X_buffer)
            if Y_buffer:
                # Y_buffer list of np or sparse arrays
                if type(Y_buffer[0]) == np.ndarray:
                    Y_den = np.vstack(Y_buffer)
                else:  # sparse
                    Y_buffer = sp.vstack(Y_buffer)
                    Y_den = np.asarray(Y_buffer.todense())

            for i in range(len(X_buffer)):
                if Y_den is not None:
                    yield X_vec[i], Y_den[i]
                else:
                    yield X_vec[i]

        def data_gen():
            """
            Wrapper on top of yield_transformed_data to get a callable function
            which enables to restart the iterator.

            This function also implements buffering for more efficient transformations
            """
            X_buffer = []
            Y_buffer = []

            X_gen = X()
            if Y:
                Y_gen = Y()
                data_zip = zip(X_gen, Y_gen)
            else:
                data_zip = X_gen
            for data_example in data_zip:
                if Y:
                    x, y = data_example
                    Y_buffer.append(y)
                else:
                    x = data_example
                X_buffer.append(x)

                if len(X_buffer) >= self.buffer_size:
                    yield from yield_transformed_data(X_buffer, Y_buffer)

                    X_buffer = []
                    Y_buffer = []

            if X_buffer:
                yield from yield_transformed_data(X_buffer, Y_buffer)

        output_types = (tf.int32, tf.int32) if Y else (tf.int32)
        data = tf.data.Dataset.from_generator(data_gen, output_types=output_types)

        if self.data_cache:
            data = data.cache(self.data_cache)
        return data

    def _init_vectorizer(self):
        self.vectorizer = KerasVectorizer(vocab_size=5_000, sequence_length=400)

    def _init_classifier(self):
        self.classifier = CNNClassifier(
            learning_rate=0.01,
            dropout=0.1,
            sparse_y=True,
            nb_epochs=20,
            nb_layers=4,
            multilabel=True,
            threshold=self.threshold,
            batch_size=self.batch_size,
        )

    def set_params(self, **params):
        if not hasattr(self, "vectorizer"):
            self._init_vectorizer()
        if not hasattr(self, "classifier"):
            self._init_classifier()
        vec_params = get_params_for_component(params, "vec")
        clf_params = get_params_for_component(params, "cnn")
        self.vectorizer.set_params(**vec_params)
        self.classifier.set_params(**clf_params)

    def fit(self, X, Y):
        """
        X: list or generator of texts
        Y: 2d numpy array or sparse csr_matrix or generator of 2d numpy array of tags assigned

        If X is a generator it need to be callable i.e. return
        the generator by calling it X_gen = X(). This is so
        we can iterate on the data again.
        """
        if not hasattr(self, "vectorizer"):
            self._init_vectorizer()
        if not hasattr(self, "classifier"):
            self._init_classifier()

        if type(X) in [list, np.ndarray]:
            print("Fitting vectorizer")
            self.vectorizer.fit(X)
            X_vec = self.vectorizer.transform(X)
            print(X_vec.shape)
            print("Fitting classifier")
            self.classifier.fit(X_vec, Y)
        else:
            print("Fitting vectorizer")
            X_gen = X()
            self.vectorizer.fit(X_gen)
            print("Fitting classifier")
            params_from_vectorizer = {
                "sequence_length": self.vectorizer.sequence_length,
                "vocab_size": self.vectorizer.vocab_size,
            }
            self.classifier.set_params(**params_from_vectorizer)
            train_data = self._yield_data(X, self.vectorizer, Y)
            # TODO: This should move inside CNNClassifier
            if self.shuffle:
                train_data = train_data.shuffle(self.buffer_size, seed=self.random_seed)
            self.classifier.fit(train_data)

        return self

    def predict(self, X):
        if type(X) in [list, np.ndarray]:
            X_vec = self.vectorizer.transform(X)
            Y_pred = self.classifier.predict(X_vec)
        else:
            pred_data = self._yield_data(X, self.vectorizer)
            Y_pred = self.classifier.predict(pred_data)
        return Y_pred

    def predict_proba(self, X):
        if type(X) in [list, np.ndarray]:
            X_vec = self.vectorizer.transform(X)
            Y_pred_proba = []
            for i in range(0, X_vec.shape[0], self.batch_size):
                Y_pred_proba_batch = self.classifier.predict_proba(
                    X_vec[i : i + self.batch_size]
                )
                if self.cutoff_prob:
                    Y_pred_proba_batch[Y_pred_proba_batch < self.cutoff_prob] = 0
                    Y_pred_proba_batch = sp.csr_matrix(Y_pred_proba_batch)
                Y_pred_proba.append(Y_pred_proba_batch)
            if self.cutoff_prob:
                Y_pred_proba = sp.hstack(Y_pred_proba)
            else:
                Y_pred_proba = np.hstack(Y_pred_proba)

        else:
            pred_data = self._yield_data(X, self.vectorizer)
            Y_pred_proba = self.classifier.predict_proba(pred_data)
        return Y_pred_proba

    def save(self, model_path):
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        meta = {"name": "MeshCNN", "approach": "mesh-cnn"}
        meta_path = os.path.join(model_path, "meta.json")
        with open(meta_path, "w") as f:
            f.write(json.dumps(meta))

        vectorizer_path = os.path.join(model_path, "vectorizer.pkl")
        save_pickle(vectorizer_path, self.vectorizer)
        self.classifier.save(model_path)

    def load(self, model_path):
        meta_path = os.path.join(model_path, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.loads(f.read())
        self.set_params(**meta)

        vectorizer_path = os.path.join(model_path, "vectorizer.pkl")
        self.vectorizer = load_pickle(vectorizer_path)

        self._init_classifier()
        self.classifier.load(model_path)
