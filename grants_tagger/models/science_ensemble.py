import pickle

import numpy as np

from wellcomeml.ml.bert_classifier import BertClassifier
from wellcomeml.ml.voting_classifier import WellcomeVotingClassifier


class ScienceEnsemble:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, Y):
        raise NotImplementedError

    def predict(self, X):
        if self.threshold != 0.5:
            Y_pred = self.model.predict_proba(X) > self.threshold
        else:
            Y_pred = self.model.predict(X)
        return Y_pred

    def predict_proba(self, X):
        # TODO: Replace with self.model.predict_proba(X) when implmented
        Y_probs = np.array([est.predict_proba(X) for est in self.estimators])
        Y_prob = np.mean(Y_probs, axis=0)
        return Y_prob

    def save(self):
        raise NotImplementedError

    def _load(self, model_path):
        # TODO: Retrieve from predefined locations that save will use
        if "tfidf-svm" in model_path:
            with open(model_path, "rb") as f:
                model = pickle.loads(f.read())
            return model
        elif "scibert" in model_path:
            model = BertClassifier(pretrained="scibert")
            model.load(model_path)
            return model
        else:
            print(
                f"Did not recognise model in {model_path} to be one of tfidf-svm or scibert"
            )
            raise NotImplementedError

    def load(self, model_paths):
        self.estimators = []
        for model_path in model_paths.split(","):
            estimator = self._load(model_path)
            self.estimators.append(estimator)

        self.model = WellcomeVotingClassifier(
            estimators=self.estimators, voting="soft", multilabel=True
        )
