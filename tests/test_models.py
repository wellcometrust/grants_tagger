import math
import os

from skmultilearn.problem_transform import ClassifierChain, BinaryRelevance, LabelPowerset
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import pytest

from wellcomeml.ml import BertVectorizer, BertClassifier, BiLSTMClassifier, CNNClassifier, KerasVectorizer, Doc2VecVectorizer, Sent2VecVectorizer, SpacyClassifier
from grants_tagger.models.create_model import create_model, ApproachNotImplemented
from grants_tagger.models.mesh_cnn import MeshCNN
from grants_tagger.models.mesh_tfidf_svm import MeshTfidfSVM
from grants_tagger.models.tfidf_transformers_svm import TfidfTransformersSVM
try:
    from grants_tagger.models.mesh_xlinear import MeshXLinear
    MESH_XLINEAR_IMPORTED = True
except ImportError:
    MESH_XLINEAR_IMPORTED = False

X = [
    "all",
    "one two",
    "two",
    "four",
    "one thousand"
]

Y_mesh = [
    [str(i) for i in range(5000)],
    ["1", "2"],
    ["2"],
    ["200"],
    ["1000"]
]

@pytest.mark.skipif(not MESH_XLINEAR_IMPORTED, reason="MeshXLinear missing")
def test_mesh_xlinear(tmp_path):
    label_binarizer = MultiLabelBinarizer(sparse_output=True)
    label_binarizer.fit(Y_mesh)

    Y_vec = label_binarizer.transform(Y_mesh)
   
    model = MeshXLinear(max_features=2000, min_df=1, vectorizer_library='sklearn')
    assert model.max_features == 2000
    assert model.min_df == 1
    
    model.fit(X, Y_vec)
    model.save(tmp_path)

    vectorizer_path = os.path.join(tmp_path, "vectorizer.pkl")
    param_path = os.path.join(tmp_path, "param.json")
    assert os.path.exists(vectorizer_path)
    assert os.path.exists(param_path)

def test_mesh_cnn(tmp_path):
    label_binarizer = MultiLabelBinarizer(sparse_output=True)
    label_binarizer.fit(Y_mesh)

    Y_vec = label_binarizer.transform(Y_mesh)

    model = MeshCNN(batch_size=32)
    assert model.batch_size == 32

    params = {"vec__vocab_size": 1000, "cnn__batch_size": 64}
    model.set_params(**params)
    assert model.vectorizer.vocab_size == 1000
    assert model.classifier.batch_size == 64

    model.fit(X, Y_vec)
    model.save(tmp_path)
    
    meta_path = os.path.join(tmp_path, "meta.json")
    vectorizer_path = os.path.join(tmp_path, "vectorizer.pkl")
    assets_path = os.path.join(tmp_path, "assets")
    assert os.path.exists(meta_path)
    assert os.path.exists(vectorizer_path)
    assert os.path.exists(assets_path)

def test_mesh_tfidf_svm(tmp_path):
    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit(Y_mesh)

    Y_vec = label_binarizer.transform(Y_mesh)

    model = MeshTfidfSVM(y_batch_size=32, model_path=tmp_path)
    assert model.y_batch_size == 32
    assert model.model_path == tmp_path

    model_path = os.path.join(tmp_path, "model")
    params = {
        "tfidf__stop_words": None, 
        "tfidf__min_df": 1,
        "svm__estimator__loss": "log",
        "model_path": model_path,
        "y_batch_size": 64}
    model.set_params(**params)
    assert model.vectorizer.stop_words is None
    assert model.classifier.estimator.loss == "log"
    assert model.model_path == model_path
    assert model.y_batch_size == 64

    model.fit(X, Y_vec)
    model.save(model_path)
   
    meta_path = os.path.join(model_path, "meta.json")
    vectorizer_path = os.path.join(model_path, "vectorizer.pkl")
    clf_paths = [
        p for p in os.listdir(model_path)
        if "meta" not in p and "vectorizer" not in p]
    assert os.path.exists(meta_path)
    assert os.path.exists(vectorizer_path)
    assert len(clf_paths) == math.ceil(5000 / 64)

def test_tfidf_transformers_svm():
    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit(Y_mesh)

    Y_vec = label_binarizer.transform(Y_mesh)

    model = TfidfTransformersSVM()

    params = {
        "tfidf__stop_words": None,
        "tfidf__min_df": 1
    }
    model.set_params(**params)

    model.fit(X, Y_vec)

@pytest.mark.skipif(not MESH_XLINEAR_IMPORTED, reason="MeshXLinear missing")
def test_create_mesh_xlinear():
    model = create_model('mesh-xlinear')
    assert isinstance(model, MeshXLinear)

def test_create_tfidf_transformers_svm():
    model = create_model('tfidf-transformers-svm')
    assert isinstance(model, TfidfTransformersSVM)

def test_create_mesh_cnn():
    model = create_model('mesh-cnn')
    assert isinstance(model, MeshCNN)

def test_create_mesh_tfidf_svm():
    model = create_model('mesh-tfidf-svm')
    assert isinstance(model, MeshTfidfSVM)

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
