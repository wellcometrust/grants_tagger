from skmultilearn.problem_transform import ClassifierChain, BinaryRelevance, LabelPowerset
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from wellcomeml.ml import BertVectorizer, BertClassifier, BiLSTMClassifier, CNNClassifier, KerasVectorizer, Doc2VecVectorizer, Sent2VecVectorizer, SpacyClassifier

from grants_tagger.models import create_model, ApproachNotImplemented


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
    assert model.pretrained == 'bert-base-uncased'


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


def test_create_spacy_textclassifier():
    model = create_model("spacy-textclassifier")
    assert isinstance(model, SpacyClassifier)


def test_create_doc2vec_tfidf_sgd():
    model = create_model('doc2vec-tfidf-sgd')
    vec = model.steps[0][1]
    ovr = model.steps[1][1]
    doc2vec = vec.transformer_list[0][1].steps[0][1]
    tfidf = vec.transformer_list[1][1].steps[0][1]
    sgd = ovr.get_params()['estimator']
    assert isinstance(model, Pipeline)
    assert isinstance(doc2vec, Doc2VecVectorizer)
    assert isinstance(tfidf, TfidfVectorizer)
    assert isinstance(sgd, SGDClassifier)


def test_create_sent2vec_tfidf_sgd():
    model = create_model('sent2vec-tfidf-sgd')
    vec = model.steps[0][1]
    ovr = model.steps[1][1]
    sent2vec = vec.transformer_list[0][1].steps[0][1]
    tfidf = vec.transformer_list[1][1].steps[0][1]
    sgd = ovr.get_params()['estimator']
    assert isinstance(model, Pipeline)
    assert isinstance(sent2vec, Sent2VecVectorizer)
    assert isinstance(tfidf, TfidfVectorizer)
    assert isinstance(sgd, SGDClassifier)


def test_create_tfidf_onehot_team_svm():
    model = create_model('tfidf+onehot_team-svm')
    vec = model.steps[0][1]
    ovr = model.steps[1][1]
    tfidf = vec.transformer_list[0][1].steps[1][1]
    onehot = vec.transformer_list[1][1].steps[1][1]
    sgd = ovr.get_params()['estimator']
    assert isinstance(model, Pipeline)
    assert isinstance(onehot, OneHotEncoder)
    assert isinstance(tfidf, TfidfVectorizer)
    assert isinstance(sgd, SVC)


def test_create_tfidf_onehot_scheme_svm():
    model = create_model('tfidf+onehot_team-svm')
    vec = model.steps[0][1]
    ovr = model.steps[1][1]
    tfidf = vec.transformer_list[0][1].steps[1][1]
    onehot = vec.transformer_list[1][1].steps[1][1]
    sgd = ovr.get_params()['estimator']
    assert isinstance(model, Pipeline)
    assert isinstance(onehot, OneHotEncoder)
    assert isinstance(tfidf, TfidfVectorizer)
    assert isinstance(sgd, SVC)
