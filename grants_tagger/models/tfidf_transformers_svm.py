from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from wellcomeml.ml.transformers_tokenizer import TransformersTokenizer


class TfidfTransformersSVM:
    def _init_model(self):
        self.tokenizer = TransformersTokenizer()
        self.model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        max_df=0.95,
                        min_df=0.0,
                        ngram_range=(1, 1),
                        tokenizer=self.tokenizer.tokenize,
                    ),
                ),
                ("svm", OneVsRestClassifier(SVC(kernel="linear", probability=True))),
            ]
        )

    def set_params(self, **params):
        if not hasattr(self, "model"):
            self._init_model()

        # TODO: Pass params to TransformersTokenizer
        self.model.set_params(**params)

    def fit(self, X, Y):
        if not hasattr(self, "model"):
            self.model = self._init_model()

        self.tokenizer.fit(X)
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
