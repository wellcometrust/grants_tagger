from unittest.mock import patch
import tempfile
import shutil
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from wellcomeml.ml import BertClassifier, CNNClassifier, KerasVectorizer
import numpy as np

from grants_tagger.predict import predict_tfidf_svm, predict_tags, predict

X = [
    "all",
    "one two",
    "two",
    "four",
    "twenty four"
]

Y = [
    [str(i) for i in range(24)],
    ["1", "2"],
    ["2"],
    ["4"],
    ["24"]
]

Y_mesh = [
    [str(i) for i in range(5000)],
    ["1", "2"],
    ["2"],
    ["200"],
    ["1000"]
]


def train_test_tfidf_svm_model(model_path, label_binarizer_path):
    tfidf = TfidfVectorizer()
    tfidf.fit(X)
    with open(f"{model_path}/tfidf.pkl", "wb") as f:
        f.write(pickle.dumps(tfidf))

    X_vec = tfidf.transform(X)

    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit(Y_mesh)
    with open(f"{label_binarizer_path}", "wb") as f:
        f.write(pickle.dumps(label_binarizer))
        f.seek(0)

    Y_vec = label_binarizer.transform(Y_mesh)

    for tag_i in range(0, Y_vec.shape[1], 512):
        model = OneVsRestClassifier(SGDClassifier(loss='log'))
        model.fit(X_vec, Y_vec[:, tag_i:tag_i+512])
        with open(f"{model_path}/{tag_i}.pkl", "wb") as f:
            f.write(pickle.dumps(model))


def train_test_cnn_model(model_path, label_binarizer_path):
    vectorizer = KerasVectorizer()
    vectorizer.fit(X)
    with open(f"{model_path}/vectorizer.pkl", "wb") as f:
        f.write(pickle.dumps(vectorizer))

    X_vec = vectorizer.transform(X)

    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit(Y_mesh)
    with open(f"{label_binarizer_path}", "wb") as f:
        f.write(pickle.dumps(label_binarizer))
        f.seek(0)

    Y_vec = label_binarizer.transform(Y_mesh)

    model = CNNClassifier(multilabel=True)
    model.fit(X_vec, Y_vec)
    model.save(model_path)


def train_test_ensemble_model(scibert_path, tfidf_svm_path, label_binarizer_path):
    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit(Y)
    with open(f"{label_binarizer_path}", "wb") as f:
        f.write(pickle.dumps(label_binarizer))
        f.seek(0)

    Y_vec = label_binarizer.transform(Y)

    tfidf_svm = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svm', OneVsRestClassifier(SGDClassifier(loss='log')))
    ])
    
    tfidf_svm.fit(X, Y_vec)
    with open(tfidf_svm_path, "wb") as f:
        f.write(pickle.dumps(tfidf_svm))

    scibert = BertClassifier()
    scibert.fit(X, Y_vec)
    scibert.save(scibert_path)


def test_predict_mesh_tfidf_svm():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = f"{tmp_dir}/disease_mesh_tfidf_model/"
        os.mkdir(model_path)
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        train_test_tfidf_svm_model(model_path, label_binarizer_path)

        tags = predict_tags(X, model_path, label_binarizer_path)
        assert len(tags) == 5


def test_predict_mesh_tfidf_svm_threshold():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = f"{tmp_dir}/disease_mesh_tfidf_model/"
        os.mkdir(model_path)
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        train_test_tfidf_svm_model(model_path, label_binarizer_path)

        nb_examples = 5
        nb_labels = 5000
        Y = predict_tfidf_svm(X, model_path, nb_labels, threshold=0, return_probabilities=False)
        assert Y.sum() == nb_labels * nb_examples # all 1

        Y = predict_tfidf_svm(X, model_path, nb_labels, threshold=1, return_probabilities=False)
        assert Y.sum() == 0 # all 0


def test_predict_tags_ensemble():
    with tempfile.TemporaryDirectory() as tmp_dir:
        scibert_path = f"{tmp_dir}/scibert/"
        os.mkdir(scibert_path)
        tfidf_svm_path = f"{tmp_dir}/tfidf_svm.pkl"
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        train_test_ensemble_model(scibert_path, tfidf_svm_path, label_binarizer_path)

        tags = predict_tags(X, model_path=[scibert_path,tfidf_svm_path],
                            label_binarizer_path=label_binarizer_path)
        assert len(tags) == 5


def test_predict_tags_mesh():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = f"{tmp_dir}/disease_mesh_cnn_model/"
        os.mkdir(model_path)
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        train_test_cnn_model(model_path, label_binarizer_path)

        tags = predict_tags(X, model_path, label_binarizer_path)
        assert len(tags) == 5

def test_evaluate_model_predict_cnn():
    with patch('grants_tagger.predict.predict_cnn') as mock_predict_cnn:
        predict("X_test", "disease_mesh_cnn_model_path", "nb_labels", "threshold")
        mock_predict_cnn.assert_called()

def test_evaluate_model_predict_tfidf_svm():
    with patch('grants_tagger.predict.predict_tfidf_svm') as mock_predict_tfidf_svm:
        predict("X_test", "disease_mesh_tfidf_model_path", "nb_labels", "threshold")
        mock_predict_tfidf_svm.assert_called()

def test_evaluate_model_predict_ensemble():
    with patch('grants_tagger.predict.predict_proba_ensemble_tfidf_svm_bert') as mock_predict_ensemble:
        mock_predict_ensemble.return_value = np.random.randn(2, 24)
        predict("X_test", ["tfidf","bert_model_paths"], "nb_labels", 0.5)
        mock_predict_ensemble.assert_called()
