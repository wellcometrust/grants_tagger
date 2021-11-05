import logging
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from pecos.xmc.xlinear.model import XLinearModel
from pecos.xmc import Indexer, LabelEmbeddingFactory

from grants_tagger.models.utils import get_params_for_component
from grants_tagger.utils import save_pickle, load_pickle

logger = logging.getLogger(__name__)


class MeshXLinear:
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
    ):
        self.stop_words = stop_words
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        self.cluster_chain = cluster_chain
        self.model_params = {
            "negative_sampling_scheme": negative_sampling_scheme,
            "beam_size": beam_size,
            "only_topk": only_topk,
            "threshold": min_weight_value,
        }

    def _init_vectorizer(self):
        self.vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=self.lowercase,
        )

    def set_params(self, **params):
        if not hasattr(self, "vectorizer"):
            self._init_vectorizer()
        vec_params = get_params_for_component(params, "tfidf")
        self.vectorizer.set_params(**vec_params)
        # XLinear params are passed during train (fit)
        model_params = get_params_for_component(params, "xlinear")
        self.model_params.update(model_params)

    def fit(self, X, Y):
        logger.info("Fitting vectorizer")
        if not hasattr(self, "vectorizer"):
            self._init_vectorizer()
        self.vectorizer.fit(X)

        X_vec = self.vectorizer.transform(X).astype("float32")
        Y = Y.astype("float32")

        logger.info("Creating cluster chain")
        label_feat = LabelEmbeddingFactory.create(Y, X_vec, method="pifa")
        cluster_chain = Indexer.gen(label_feat, indexer_type="hierarchicalkmeans")

        logger.info("Training model")
        model = XLinearModel()
        self.model = model.train(X_vec, Y, C=cluster_chain, **self.model_params)
        return self

    def predict(self, X):
        return self.predict_proba(X) > 0.5

    def predict_proba(self, X):
        return self.model.predict(self.vectorizer.transform(X).astype("float32"))

    def save(self, model_path):
        vectorizer_path = os.path.join(model_path, "vectorizer.pkl")
        save_pickle(vectorizer_path, self.vectorizer)
        self.model.save(model_path)

    def load(self, model_path, is_predict_only=True):
        vectorizer_path = os.path.join(model_path, "vectorizer.pkl")
        self.vectorizer = load_pickle(vectorizer_path)
        self.model = XLinearModel.load(model_path, is_predict_only=is_predict_only)
