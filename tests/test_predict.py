from unittest.mock import patch
import tempfile
import shutil
import pickle
import os

# TODO: Remove when ScienceEnsemble implements fit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from wellcomeml.ml import BertClassifier

from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

from grants_tagger.models import MeshCNN, MeshTfidfSVM
from grants_tagger.predict import predict_tags, predict

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


def train_label_binarizer(Y, label_binarizer_path):
    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit(Y)
    with open(f"{label_binarizer_path}", "wb") as f:
        f.write(pickle.dumps(label_binarizer))
        f.seek(0)
    return label_binarizer

def train_test_tfidf_svm_model(model_path, label_binarizer_path):
    label_binarizer = train_label_binarizer(Y_mesh, label_binarizer_path)
    Y_vec = label_binarizer.transform(Y_mesh)

    model = MeshTfidfSVM(model_path=model_path)
    model.set_params(
        vec__min_df=1,
        vec__stop_words=None,
        clf__estimator__loss="log"
    )
    model.fit(X, Y_vec)
    model.save(model_path)


def train_test_cnn_model(model_path, label_binarizer_path):
    label_binarizer = train_label_binarizer(Y_mesh, label_binarizer_path)
    Y_vec = label_binarizer.transform(Y_mesh)

    model = MeshCNN()
    model.fit(X, Y_vec)
    model.save(model_path)


def train_test_ensemble_model(scibert_path, tfidf_svm_path, label_binarizer_path):
    label_binarizer = train_label_binarizer(Y, label_binarizer_path)
    Y_vec = label_binarizer.transform(Y)

    tfidf_svm = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svm', OneVsRestClassifier(SGDClassifier(loss='log')))
    ])
    
    tfidf_svm.fit(X, Y_vec)
    with open(tfidf_svm_path, "wb") as f:
        f.write(pickle.dumps(tfidf_svm))

    scibert = BertClassifier(
            epochs=1,
            pretrained="scibert")
    scibert.fit(X, Y_vec)
    scibert.save(scibert_path)


def test_predict_threshold():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = f"{tmp_dir}/disease_mesh_tfidf_model/"
        os.mkdir(model_path)
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        train_test_tfidf_svm_model(model_path, label_binarizer_path)

        nb_examples = 5
        nb_labels = 5000
        Y = predict(X, model_path, approach="mesh-tfidf-svm",
                threshold=0, return_probabilities=False)
        assert Y.sum() == nb_labels * nb_examples # all 1

        Y = predict(X, model_path, approach="mesh-tfidf-svm",
                threshold=1, return_probabilities=False)
        assert Y.sum() == 0 # all 0


def test_predict_tags_science_ensemble():
    with tempfile.TemporaryDirectory() as tmp_dir:
        scibert_path = f"{tmp_dir}/scibert/"
        os.mkdir(scibert_path)
        tfidf_svm_path = f"{tmp_dir}/tfidf_svm.pkl"
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        train_test_ensemble_model(scibert_path, tfidf_svm_path, label_binarizer_path)

        model_path = ",".join([scibert_path, tfidf_svm_path])
        tags = predict_tags(
                X, model_path=model_path,
                label_binarizer_path=label_binarizer_path,
                approach="science-ensemble")
        assert len(tags) == 5


def test_predict_tags_mesh_tfidf_svm():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = f"{tmp_dir}/disease_mesh_tfidf_model/"
        os.mkdir(model_path)
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        train_test_tfidf_svm_model(model_path, label_binarizer_path)

        tags = predict_tags(X, model_path, label_binarizer_path, 
                approach="mesh-tfidf-svm")
        assert len(tags) == 5


def test_predict_tags_mesh_cnn():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = f"{tmp_dir}/disease_mesh_cnn_model/"
        os.mkdir(model_path)
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        train_test_cnn_model(model_path, label_binarizer_path)

        tags = predict_tags(X, model_path, label_binarizer_path,
                approach="mesh-cnn")
        assert len(tags) == 5
