import logging
import os
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin

from pecos.xmc.xlinear.model import XLinearModel
from pecos.xmc import Indexer, LabelEmbeddingFactory

from grants_tagger.utils import save_pickle, load_pickle

logger = logging.getLogger(__name__)


class MeshXLinear(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        stop_words="english",
        min_df=5,
        max_df=1.0,
        max_features=400_000,
        ngram_range=(1, 1),
        lowercase=True,
        cluster_chain=True,
        negative_sampling_scheme="tfn",
        beam_size=10,
        only_topk=20,
        min_weight_value=0.1,
        imbalanced_ratio=0,
        vectorizer_library="sklearn",
        threshold=0.5,
    ):
        # Sklearn estimators need all arguments to be assigned to variables with the same name

        # Those are Tf-idf params
        self.stop_words = stop_words
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.lowercase = lowercase

        # Those are XLinear params
        self.cluster_chain = cluster_chain

        self.negative_sampling_scheme = negative_sampling_scheme
        self.beam_size = beam_size
        self.only_topk = only_topk
        self.min_weight_value = min_weight_value
        self.imbalanced_ratio = imbalanced_ratio
        self.vectorizer_library = vectorizer_library

        # Those are MeshXLinear params
        self.threshold = threshold

    def _init_vectorizer(self):
        # Sklearn estimators need variables introduced during training to have a trailing comma
        if self.vectorizer_library == "sklearn":
            self.vectorizer_ = TfidfVectorizer(
                stop_words=self.stop_words,
                min_df=self.min_df,
                max_df=self.max_df,
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                lowercase=self.lowercase,
            )
        elif self.vectorizer_library == "pecos":
            # Let it raise an error if it doesn't have the right dependencies,
            # But only import Tfidf if necessary
            from pecos.utils.featurization.text.vectorizers import Tfidf

            self.vectorizer_ = Tfidf()
        else:
            raise ValueError("Vectorizer library has to be pecos or sklearn")

    def fit(self, X, Y):
        # Basic tabs and linebreak removal because pecos tfidf doesnt do it
        logger.info("Removing punctuation")
        X = [x.replace("\n", "").replace("\t", "") for x in X]

        logger.info("Fitting vectorizer")

        self._init_vectorizer()

        if self.vectorizer_library == "sklearn":
            X_vec = self.vectorizer_.fit_transform(X).astype("float32")
        else:
            self.vectorizer_ = self.vectorizer_.train(
                X,
                config={
                    "ngram_range": self.ngram_range,
                    "max_feature": self.max_features,
                    "min_df_cnt": self.min_df,
                },
            )
            X_vec = self.vectorizer_.predict(X).astype("float32")

        Y = Y.astype("float32")

        logger.info("Creating cluster chain")

        label_feat = LabelEmbeddingFactory.create(Y, X_vec, method="pifa")
        cluster_chain = Indexer.gen(label_feat, indexer_type="hierarchicalkmeans")
        xlinear_model = XLinearModel()

        logger.info("Training model")

        # Sklearn estimators need variables introduced during training to have a trailing underscore
        self.xlinear_model_ = xlinear_model.train(
            X_vec,
            Y,
            C=cluster_chain,
            cluster_chain=self.cluster_chain,
            negative_sampling_scheme=self.negative_sampling_scheme,
            only_topk=self.only_topk,
            threshold=self.min_weight_value,
        )
        return self

    def predict(self, X):
        return self.predict_proba(X) > self.threshold

    def predict_proba(self, X):
        if self.vectorizer_library == "sklearn":
            return self.xlinear_model_.predict(
                self.vectorizer_.transform(X).astype("float32"),
                only_topk=self.only_topk,
                beam_size=self.beam_size,
            )
        else:
            return self.xlinear_model_.predict(
                self.vectorizer_.predict(X).astype("float32"),
                only_topk=self.only_topk,
                beam_size=self.beam_size,
            )

    def save(self, model_path):
        model_path = str(model_path)  # In case a Posix is passed
        params_path = os.path.join(model_path, "params.json")
        vectorizer_path = os.path.join(model_path, "vectorizer.pkl")
        with open(params_path, "w") as f:
            json.dump(self.__dict__, f, indent=4, default=str)

        if self.vectorizer_library == "sklearn":
            save_pickle(vectorizer_path, self.vectorizer_)
        else:
            self.vectorizer_.save(model_path)

        self.xlinear_model_.save(model_path)

    def load(self, model_path, is_predict_only=True):
        model_path = str(model_path)  # In case a Posix is passed
        params_path = os.path.join(model_path, "params.json")
        vectorizer_path = os.path.join(model_path, "vectorizer.pkl")
        with open(params_path, "r") as f:
            self.__dict__.update(json.load(f))

        if self.vectorizer_library == "sklearn":
            self.vectorizer_ = load_pickle(vectorizer_path)
        else:
            from pecos.utils.featurization.text.vectorizers import Tfidf

            self.vectorizer_ = Tfidf()
            self.vectorizer_ = self.vectorizer_.load(model_path)

        self.xlinear_model_ = XLinearModel.load(
            model_path, is_predict_only=is_predict_only
        )
