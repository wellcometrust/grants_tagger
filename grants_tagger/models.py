from pathlib import Path
import pickle
import json
import ast
import os

from skmultilearn.problem_transform import ClassifierChain, LabelPowerset, BinaryRelevance
from skmultilearn.adapt import BRkNNaClassifier, MLkNN
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer, OneHotEncoder, FunctionTransformer
from scipy import sparse as sp
import tensorflow as tf
import numpy as np

from wellcomeml.ml import BiLSTMClassifier, CNNClassifier, KerasVectorizer, SpacyClassifier, BertVectorizer, BertClassifier, Doc2VecVectorizer, Sent2VecVectorizer, WellcomeVotingClassifier


class ApproachNotImplemented(Exception):
    pass


def get_params_for_component(params, component):
    component_params = {}
    for k, v in params.items():
        if k.startswith(component):
            _, component_arg = k.split(f"{component}__")
            component_params[component_arg] = v
    return component_params


class ScienceEnsemble():
    def __init__(self):
        pass

    def fit(self, X, Y):
        # Implement now or later?
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # TODO: Replace with self.model.predict_proba(X) when implmented
        Y_probs = np.array([
            est.predict_proba(X) for est in self.estimators])
        Y_prob = np.mean(Y_probs, axis=0)
        return Y_prob

    def save(self):
        # Implement along with fit
        pass

    def _load(self, model_path):
        # TODO: find out model from meta.json
        if "tfidf_svm" in model_path:
            with open(model_path, "rb") as f:
                model = pickle.loads(f.read())
            return model
        elif "scibert" in model_path:
            model = BertClassifier(pretrained="scibert")
            model.load(model_path)
            return model
        else:
            print(f"Did not recognise model in {model_path} to be one of tfidf-svm or scibert")
            raise NotImplementedError

    def load(self, model_paths):
        self.estimators = []
        for model_path in model_paths.split(","):
            estimator = self._load(model_path)
            self.estimators.append(estimator)
        
        self.model = WellcomeVotingClassifier(
            estimators=self.estimators,
            voting="soft",
            multilabel=True
        )


class MeshCNN():
    def __init__(self, threshold=0.5, batch_size=256, shuffle=True,
            buffer_size=1000, data_cache=None, random_seed=42):
        self.threshold = threshold
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.data_cache = data_cache
        self.random_seed = random_seed

    def _yield_data(self, X, vectorizer, Y=None):
        """
        Add docstring
        """
        def yield_transformed_data(X_buffer, Y_buffer):
            Y_den = None
            X_vec = self.vectorizer.transform(X_buffer)
            if Y_buffer:
                # Y_buffer list of np or sparse arrays
                if type(Y_buffer[0]) == np.ndarray:
                    Y_den = np.vstack(Y_buffer)
                else: # sparse
                    Y_buffer = sp.vstack(Y_buffer)
                    Y_den = np.asarray(Y_buffer.todense())

            for i in range(len(X_buffer)):
                if Y_den is not None:
                    yield X_vec[i], Y_den[i]
                else:
                    yield X_vec[i]

        def data_gen():
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
                    # Use yield from?
                    for data_item in yield_transformed_data(X_buffer, Y_buffer):
                        yield data_item

                    X_buffer = []
                    Y_buffer = []

            if X_buffer:
                # Use yield from?
                for data_item in yield_transformed_data(X_buffer, Y_buffer):
                    yield data_item

        output_types = (tf.int32, tf.int32) if Y else (tf.int32)
        data = tf.data.Dataset.from_generator(
                data_gen, output_types=output_types)
         
        # Note we need to enable shuffling after every epoch in model
        #   here we just shuffle once
        #   possibly we should not shuffle at all here
        if self.shuffle:
            data = data.shuffle(self.buffer_size, seed=self.random_seed, reshuffle_each_iteration=False)
        if self.data_cache:
            data = data.cache(self.data_cache)
        return data

    def _init_model(self):
        self.vectorizer = KerasVectorizer(
                vocab_size=5_000, sequence_length=400)
        self.classifier = CNNClassifier(
                learning_rate=0.01, dropout=0.1, 
                nb_epochs=20, nb_layers=4, multilabel=True,
                threshold=self.threshold, batch_size=self.batch_size)
    
    def set_params(self, **params):
        if not (hasattr(self, 'vectorizer') and hasattr(self, 'classifier')):
            self._init_model()
        vec_params = get_params_for_component(params, 'vec')
        clf_params = get_params_for_component(params, 'cnn')
        self.vectorizer.set_params(**vec_params)
        self.classifier.set_params(**clf_params)

    def fit(self, X, Y):
        """
        Add docstring
        """
        if not (hasattr(self, "vectorizer") and hasattr(self, "classifier")):
            self._init_model()

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
            # Need to pass in vocab size, sequence length and nb_outputs
            # to classifier through set_params
            train_data = self._yield_data(X, self.vectorizer, Y)
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
        # TODO: Should that be here or CNN, should it exist at all?
        # what is the purpose, if Y is too large for memory whose
        # responsibility is it?
        if type(X) in [list, np.ndarray]:
            X_vec = self.vectorizer.transform(X)
            Y_pred_proba = []
            for i in range(0, X_vec.shape[0], self.batch_size):
                Y_pred_proba_batch = self.classifier.predict_proba(X_vec[i:i+self.batch_size])
                Y_pred_proba.append(Y_pred_proba_batch)
            Y_pred_proba = np.vstack(Y_pred_proba)
        else:
            pred_data = self._yield_data(X, self.vectorizer)
            Y_pred_proba = self.classifier.predict_proba(pred_data)
        return Y_pred_proba

    def save(self, model_path):
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        vectorizer_path = os.path.join(model_path, "vectorizer.pkl")
        with open(vectorizer_path, "wb") as f:
            f.write(pickle.dumps(self.vectorizer))
        self.classifier.save(model_path)

    def load(self, model_path):
        # TODO: Compare with tfidf load and consider adding meta 
        vectorizer_path = os.path.join(model_path, "vectorizer.pkl")
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.loads(f.read())
        print(self.vectorizer.sequence_length)

        # TODO: Use init?
        self.classifier = CNNClassifier(
            threshold = self.threshold,
            multilabel = True
        )
        self.classifier.load(model_path)


class MeshTfidfSVM():
    def __init__(self, y_batch_size=256, nb_labels=None, model_path=None,
            threshold=0.5):
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

        vec_params = get_params_for_component(params, 'vec')
        clf_params = get_params_for_component(params, 'clf')
        self.vectorizer.set_params(**vec_params)
        self.classifier.set_params(**clf_params)
        # TODO: Generalise
        if 'model_path' in params:
            self.model_path = params['model_path']
        if 'y_batch_size' in params:
            self.y_batch_size = params['y_batch_size']
        if 'nb_labels' in params:
            self.nb_labels = params['nb_labels']

    def fit(self, X, Y):
        if not hasattr(self, 'vectorizer'):
            self._init_vectorizer()
        if not hasattr(self, 'classifier'):
            self._init_classifier()

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
        # model saved during fit
        meta = {
            "name": "MeshTfidfSVM",
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

def create_model(approach, parameters=None):
    if approach == 'tfidf-svm':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english', max_df=0.95,
                min_df=0.0, ngram_range=(1,1)
            )),
            ('svm', OneVsRestClassifier(SVC(kernel='linear', probability=True)))
        ])
    elif approach == 'bert-svm':
        model = Pipeline([
            ('bert', BertVectorizer(pretrained='bert')),
            ('svm', OneVsRestClassifier(SVC(kernel='linear', probability=True)))
        ])
    elif approach == 'scibert-svm':
        model = Pipeline([
            ('scibert', BertVectorizer(pretrained='scibert')),
            ('svm', OneVsRestClassifier(SVC(kernel='linear', probability=True)))
        ])
    elif approach == 'spacy-textclassifier':
        model = SpacyClassifier()
    elif approach == 'bert':
        model = BertClassifier()
    elif approach == 'scibert':
        model = BertClassifier(pretrained='scibert')
    elif approach == 'classifierchain-tfidf-svm':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english', max_df=0.95,
                min_df=0.0, ngram_range=(1,1)
            )),
            ('svm', ClassifierChain(classifier=SVC(kernel='linear', probability=True)))
        ])
    elif approach == 'labelpowerset-tfidf-svm':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english', max_df=0.95,
                min_df=0.0, ngram_range=(1,1)
            )),
            ('svm', LabelPowerset(SVC(kernel='linear', probability=True)))
        ])
    elif approach == 'binaryrelevance-tfidf-svm':
        # same as OneVsRestClassifier
        model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english', max_df=0.95,
                min_df=0.0, ngram_range=(1,1)
            )),
            ('svm', BinaryRelevance(classifier=SVC(kernel='linear', probability=True)))
        ])
    elif approach == 'binaryrelevance-tfidf-knn':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english', max_df=0.95,
                min_df=0.0, ngram_range=(1,1)
            )),
            ('knn', BinaryRelevance(classifier=KNeighborsClassifier))
        ])
    elif approach == 'hashing_vectorizer-svm':
        model = Pipeline([
            ('hashing_vectorizer', HashingVectorizer()),
            ('svm', OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='l2')))
        ])
    elif approach == 'hashing_vectorizer-nb':
        model = Pipeline([
            ('hashing_vectorizer', HashingVectorizer(binary=True, n_features=2 ** 18)),
            ('nb', OneVsRestClassifier(MultinomialNB()))
        ])
    elif approach == 'tfidf-sgd':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english', max_df=0.95,
                min_df=5, ngram_range=(1,1)
            )),
            ('svm', OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='l2')))
        ])
    elif approach == 'cnn':
        model = Pipeline([
            ('vec', KerasVectorizer(vocab_size=5_000)),
            ('cnn', CNNClassifier(learning_rate=0.01, dropout=0.1, nb_epochs=20, nb_layers=4, multilabel=True))
        ])
    elif approach == 'bilstm':
        model = Pipeline([
            ('vec', KerasVectorizer(vocab_size=5_000, sequence_length=678)),
            ('bilstm', BiLSTMClassifier(learning_rate=0.01, dropout=0.1, nb_epochs=20, multilabel=True))
        ])
    elif approach == 'doc2vec-sgd':
        model = Pipeline([
            ('vec', Doc2VecVectorizer()),
            ('sgd', OneVsRestClassifier(SGDClassifier(penalty="l2", alpha=1e-8), n_jobs=-1))
        ])
    elif approach == 'doc2vec-tfidf-sgd':
        model = Pipeline([
            ('vec', FeatureUnion([
                ('doc2vec', Pipeline([
                    ('doc2vec_unscaled', Doc2VecVectorizer()),
                    ('scale_doc2vec', Normalizer())
                ])),
                ('tfidf', Pipeline([
                    ('tfidf_unscaled', TfidfVectorizer(min_df=5, stop_words='english', ngram_range=(1,2))),
                    ('scale_tfidf', Normalizer())
                ]))
            ])),
            ('sgd', OneVsRestClassifier(SGDClassifier(penalty="l2", alpha=1e-6), n_jobs=-1))
        ])
    elif approach == 'sent2vec-sgd':
        model = Pipeline([
            ('vec', Sent2VecVectorizer(pretrained="biosent2vec")),
            ('sgd', OneVsRestClassifier(SGDClassifier(penalty="l2", alpha=1e-8), n_jobs=-1))
        ])
    elif approach == 'sent2vec-tfidf-sgd':
        model = Pipeline([
            ('vec', FeatureUnion([
                ('sent2vec', Pipeline([
                    ('sent2vec_unscaled', Sent2VecVectorizer(pretrained="biosent2vec")),
                    ('scale_sent2vec', Normalizer())
                ])),
                ('tfidf', Pipeline([
                    ('tfidf_unscaled', TfidfVectorizer(min_df=5, stop_words='english', ngram_range=(1,2))),
                    ('scale_tfidf', Normalizer())
                ]))
            ])),
            ('sgd', OneVsRestClassifier(SGDClassifier(penalty="l2", alpha=1e-8), n_jobs=-1))
        ])
    elif approach == 'tfidf-adaboost':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(min_df=5, stop_words='english', ngram_range=(1,2))),
            ('adaboost', OneVsRestClassifier(AdaBoostClassifier(DecisionTreeClassifier())))
        ])
    elif approach == 'tfidf-gboost':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(min_df=5, stop_words='english', ngram_range=(1,2))),
            ('gboost', OneVsRestClassifier(GradientBoostingClassifier()))
        ])
    elif approach == 'tfidf+onehot_team-svm':
        model = Pipeline([
            ('vectorizer', FeatureUnion([
                ("text_features", Pipeline([
                    ("selector", FunctionTransformer(lambda x: x["text"])),
                    ("tfidf", TfidfVectorizer(min_df=5, ngram_range=(1,2), stop_words="english"))
                ])),
                ("team_features", Pipeline([
                    ("selector", FunctionTransformer(lambda x: x[["Team"]])),
                    ("one hot", OneHotEncoder(handle_unknown='ignore'))
                ]))
            ])),
            ('svm', OneVsRestClassifier(SVC(class_weight="balanced", kernel="linear")))
        ])
    elif approach == 'tfidf+onehot_scheme-svm':
        model = Pipeline([
            ('vectorizer', FeatureUnion([
                ("text_features", Pipeline([
                    ("selector", FunctionTransformer(lambda x: x["text"])),
                    ("tfidf", TfidfVectorizer(min_df=5, ngram_range=(1,2), stop_words="english"))
                ])),
                ("team_features", Pipeline([
                    ("selector", FunctionTransformer(lambda x: x[["Scheme"]])),
                    ("one hot", OneHotEncoder(handle_unknown='ignore'))
                ]))
            ])),
            ('svm', OneVsRestClassifier(SVC(class_weight="balanced", kernel="linear")))
        ])
    elif approach == 'mesh-tfidf-svm':
        model = MeshTfidfSVM()
    elif approach == 'mesh-cnn':
        model = MeshCNN()
    else:
        raise ApproachNotImplemented
    if parameters:
        params = ast.literal_eval(parameters)
        model.set_params(**params)
    else:
        parameters = {}
    return model
