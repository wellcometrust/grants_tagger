import tempfile
import pickle
import json

from skmultilearn.problem_transform import ClassifierChain, BinaryRelevance, LabelPowerset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import pytest

from wellcomeml.ml import BertVectorizer, BertClassifier, BiLSTMClassifier, CNNClassifier, KerasVectorizer, Doc2VecVectorizer, Sent2VecVectorizer
from grants_tagger.train import train_and_evaluate, ApproachNotImplemented, create_model

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

def test_create_tfidf_svm():
    model = create_model('tfidf-svm')
    vec = model.steps[0][1]
    ovr = model.steps[1][1]
    clf = ovr.get_params()['estimator']
    assert isinstance(model, Pipeline)
    assert isinstance(vec, TfidfVectorizer)
    assert isinstance(ovr, OneVsRestClassifier)
    assert isinstance(clf, SVC)

def test_create_bert_svm():
    model = create_model('bert-svm')
    vec = model.steps[0][1]
    ovr = model.steps[1][1]
    clf = ovr.get_params()['estimator']
    assert isinstance(model, Pipeline)
    assert isinstance(vec, BertVectorizer)
    assert isinstance(ovr, OneVsRestClassifier)
    assert isinstance(clf, SVC)
    assert vec.pretrained == 'bert'

def test_create_scibert_svm():
    model = create_model('scibert-svm')
    vec = model.steps[0][1]
    ovr = model.steps[1][1]
    clf = ovr.get_params()['estimator']
    assert isinstance(model, Pipeline)
    assert isinstance(vec, BertVectorizer)
    assert isinstance(ovr, OneVsRestClassifier)
    assert isinstance(clf, SVC)
    assert vec.pretrained == 'scibert'

def test_create_bert():
    model = create_model('bert')
    assert isinstance(model, BertClassifier)
    assert model.pretrained == 'bert'

def test_create_scibert():
    model = create_model('scibert')
    assert isinstance(model, BertClassifier)
    assert model.pretrained == 'scibert'

def test_create_classifierchain_tfidf_svm():
    model = create_model('classifierchain-tfidf-svm')
    vec = model.steps[0][1]
    cc = model.steps[1][1]
    clf = cc.get_params()['classifier']
    assert isinstance(model, Pipeline)
    assert isinstance(vec, TfidfVectorizer)
    assert isinstance(cc, ClassifierChain)
    assert isinstance(clf, SVC)

def test_create_labelpowerset_tfidf_svm():
    model = create_model('labelpowerset-tfidf-svm')
    vec = model.steps[0][1]
    lp = model.steps[1][1]
    clf = lp.get_params()['classifier']
    assert isinstance(model, Pipeline)
    assert isinstance(vec, TfidfVectorizer)
    assert isinstance(lp, LabelPowerset)
    assert isinstance(clf, SVC)

def test_create_binaryrelevance_tfidf_svm():
    model = create_model('binaryrelevance-tfidf-svm')
    vec = model.steps[0][1]
    br = model.steps[1][1]
    clf = br.get_params()['classifier']
    assert isinstance(model, Pipeline)
    assert isinstance(vec, TfidfVectorizer)
    assert isinstance(br, BinaryRelevance)
    assert isinstance(clf, SVC)

def test_create_binaryrelevance_tfidf_knn():
    model = create_model('binaryrelevance-tfidf-knn')
    vec = model.steps[0][1]
    br = model.steps[1][1]
#    clf = br.get_params()['classifier']
    assert isinstance(model, Pipeline)
    assert isinstance(vec, TfidfVectorizer)
    assert isinstance(br, BinaryRelevance)
#    assert isinstance(clf, KNeighborsClassifier)

def test_create_bilstm():
    model = create_model('bilstm')
    vec = model.steps[0][1]
    clf = model.steps[1][1]
    assert isinstance(vec, KerasVectorizer)
    assert isinstance(clf, BiLSTMClassifier)

def test_create_cnn():
    model = create_model('cnn')
    vec = model.steps[0][1]
    clf = model.steps[1][1]
    assert isinstance(vec, KerasVectorizer)
    assert isinstance(clf, CNNClassifier)

def test_create_doc2vec_sgd():
    model = create_model('doc2vec-sgd')
    vec = model.steps[0][1]
    ovr = model.steps[1][1]
    clf = ovr.get_params()['estimator']
    assert isinstance(model, Pipeline)
    assert isinstance(vec, Doc2VecVectorizer)
    assert isinstance(ovr, OneVsRestClassifier)
    assert isinstance(clf, SGDClassifier)

def test_create_sent2vec_sgd():
    model = create_model('sent2vec-sgd')
    vec = model.steps[0][1]
    ovr = model.steps[1][1]
    clf = ovr.get_params()['estimator']
    assert isinstance(model, Pipeline)
    assert isinstance(vec, Sent2VecVectorizer)
    assert isinstance(ovr, OneVsRestClassifier)
    assert isinstance(clf, SGDClassifier)

def test_create_tfidf_adaboost():
    model = create_model('tfidf-adaboost')
    vec = model.steps[0][1]
    ovr = model.steps[1][1]
    clf = ovr.get_params()['estimator']
    assert isinstance(model, Pipeline)
    assert isinstance(vec, TfidfVectorizer)
    assert isinstance(ovr, OneVsRestClassifier)
    assert isinstance(clf, AdaBoostClassifier)

def test_create_tfidf_gboost():
    model = create_model('tfidf-gboost')
    vec = model.steps[0][1]
    ovr = model.steps[1][1]
    clf = ovr.get_params()['estimator']
    assert isinstance(model, Pipeline)
    assert isinstance(vec, TfidfVectorizer)
    assert isinstance(ovr, OneVsRestClassifier)
    assert isinstance(clf, GradientBoostingClassifier)

def test_train_and_evaluate():
    approach = "tfidf-svm"

    texts = ["one", "one two", "two"]
    tags = [["one"], ["one","two"], ["two"]]
    with tempfile.NamedTemporaryFile("r+") as train_data_tmp:
        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit(tags)
        with tempfile.NamedTemporaryFile() as label_binarizer_tmp:
            label_binarizer_tmp.write(pickle.dumps(label_binarizer))

            for text, tags_ in zip(texts, tags):
                train_data_tmp.write(json.dumps({"text": text, "tags": tags_, "meta": {}}))
                train_data_tmp.write("\n")

            train_data_tmp.seek(0)
            label_binarizer_tmp.seek(0)

            train_and_evaluate(train_data_tmp.name, label_binarizer_tmp.name, approach,
                               parameters="{'tfidf__min_df': 1, 'tfidf__stop_words': None}")
