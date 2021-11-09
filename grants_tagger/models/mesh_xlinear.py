import logging
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin

from pecos.xmc.xlinear.model import XLinearModel
from pecos.xmc import Indexer, LabelEmbeddingFactory

from grants_tagger.models.utils import get_params_for_component
from grants_tagger.utils import save_pickle, load_pickle

logger = logging.getLogger(__name__)


class MeshXLinear(BaseEstimator, ClassifierMixin):
    def __init__(self, stop_words="english", min_df=5, max_df=1.0,
            max_features=400_000, ngram_range=(1, 1), lowercase=True,
            cluster_chain=True, negative_sampling_scheme="tfn",
            beam_size=10, only_topk=20, min_weight_value=0.1):
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

    def _init_vectorizer(self):
        # Sklearn estimators need variables introduced during training to have a trailing comma
        self.vectorizer_ = TfidfVectorizer(
            stop_words=self.stop_words, min_df=self.min_df, max_df=self.max_df,
            max_features=self.max_features, ngram_range=self.ngram_range,
            lowercase=self.lowercase)

    def fit(self, X, Y):
        logger.info("Fitting vectorizer")
        
        self._init_vectorizer()
        self.vectorizer_.fit(X)

        X_vec = self.vectorizer_.transform(X).astype("float32")
        Y = Y.astype("float32")
        
        logger.info("Creating cluster chain") 
        label_feat = LabelEmbeddingFactory.create(Y, X_vec, method="pifa")
        cluster_chain = Indexer.gen(label_feat, indexer_type="hierarchicalkmeans")
        xlinear_model = XLinearModel()

        logger.info("Training model")

        # Sklearn estimators need variables introduced during training to have a trailing comma
        self.xlinear_model_ = xlinear_model.train(
            X_vec,
            Y,
            C=cluster_chain,
            cluster_chain=self.cluster_chain,
            negative_sampling_scheme=self.negative_sampling_scheme,
            only_topk=self.only_topk,
            treshold=self.min_weight_value
        )
        return self

    def predict(self, X):
        return self.predict_proba(X) > 0.5

    def predict_proba(self, X):
        return self.xlinear_model_.predict(
            self.vectorizer_.transform(X).astype("float32")
        )

    def save(self, model_path):
        vectorizer_path = os.path.join(model_path, "vectorizer.pkl")
        save_pickle(vectorizer_path, self.vectorizer_)
        self.xlinear_model_.save(model_path)

    def load(self, model_path, is_predict_only=True):
        vectorizer_path = os.path.join(model_path, "vectorizer.pkl")
        self.vectorizer_ = load_pickle(vectorizer_path)
        self.xlinear_model_ = XLinearModel.load(model_path, is_predict_only=is_predict_only)
