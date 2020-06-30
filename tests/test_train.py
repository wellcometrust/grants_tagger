from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import pytest

from science_tagger.train import train_and_evaluate, ApproachNotImplemented, create_model

DATA_PATH = 'science_tagger/data/processed/science_grants_tagged.jsonl'
LABEL_BINARIZER_PATH = 'science_tagger/models/label_binarizer.pkl'


def test_unknown_approach():
    with pytest.raises(ApproachNotImplemented):
        train_and_evaluate(
            DATA_PATH,
            LABEL_BINARIZER_PATH,
            approach='None'
        )

def test_create_hashing_vectorizer_svm():
    model = create_model('hashing_vectorizer-svm')
    vec = model.steps[0][1]
    ovr = model.steps[1][1]
    clf = ovr.get_params()['estimator']
    assert isinstance(model, Pipeline)
    assert isinstance(vec, HashingVectorizer)
    assert isinstance(ovr, OneVsRestClassifier)
    assert isinstance(clf, SGDClassifier)

def test_create_hashing_vectorizer_nb():
    model = create_model('hashing_vectorizer-nb')
    vec = model.steps[0][1]
    clf = model.steps[1][1]
    assert isinstance(model, Pipeline)
    assert isinstance(vec, HashingVectorizer)
    assert isinstance(clf, OneVsRestClassifier)
