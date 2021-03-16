from unittest.mock import patch
import tempfile
import shutil
import pickle
import json
import os

import numpy as np
import pytest

from grants_tagger.train import create_label_binarizer, train_and_evaluate
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
    ["23"]
]

Y_mesh = [
    [str(i) for i in range(5000)],
    ["1", "2"],
    ["2"],
    ["200"],
    ["1000"]
]

def create_data(X, Y, data_path):
    with open(data_path, "w") as f:
        for x, y in zip(X, Y):
            f.write(json.dumps({"text": x, "tags": y, "meta": {}}))
            f.write("\n")

@pytest.fixture
def science_ensemble_path(tmp_path):
    data_path = os.path.join(tmp_path, "data.jsonl")
    create_data(X, Y, data_path)

    label_binarizer_path = os.path.join(tmp_path, "label_binarizer.pkl")
    label_binarizer = create_label_binarizer(data_path, label_binarizer_path)

    # TODO: Replace approach with science-ensemble when fit implemented
    tfidf_svm_path = os.path.join(tmp_path, "tfidf_svm.pkl")
    parameters = {
        'tfidf__min_df': 1,
        'tfidf__stop_words': None
    }
    train_and_evaluate(data_path, label_binarizer_path,
            approach="tfidf-svm", model_path=tfidf_svm_path,
            parameters=str(parameters), verbose=False)

    scibert_path = os.path.join(tmp_path, "scibert")
    parameters = {"epochs": 1}
    train_and_evaluate(data_path, label_binarizer_path,
            approach="scibert", model_path=scibert_path,
            parameters=str(parameters), verbose=False)
    
    science_ensemble_path = f"{tfidf_svm_path},{scibert_path}"
    return science_ensemble_path

@pytest.fixture
def mesh_tfidf_svm_path(tmp_path):
    mesh_data_path = os.path.join(tmp_path, "mesh_data.jsonl")
    create_data(X, Y_mesh, mesh_data_path)
    
    label_binarizer_path = os.path.join(tmp_path, "label_binarizer.pkl")

    model_path = os.path.join(tmp_path, "mesh_tfidf_svm")
    parameters = {
        'vec__min_df': 1,
        'vec__stop_words': None,
        'clf__estimator__loss': 'log',
        'model_path': model_path
    }
    train_and_evaluate(mesh_data_path, label_binarizer_path,
        approach="mesh-tfidf-svm", model_path=model_path,
        parameters=str(parameters), sparse_labels=True,
        verbose=False)
    return model_path

@pytest.fixture
def mesh_cnn_path(tmp_path):
    mesh_data_path = os.path.join(tmp_path, "mesh_data.jsonl")
    create_data(X, Y_mesh, mesh_data_path)
 
    label_binarizer_path = os.path.join(tmp_path, "label_binarizer.pkl")
    model_path = os.path.join(tmp_path, "mesh_cnn")
    train_and_evaluate(mesh_data_path, label_binarizer_path,
        approach="mesh-cnn", model_path=model_path,
        verbose=False)
    return model_path

@pytest.fixture
def label_binarizer_path(tmp_path):
    data_path = os.path.join(tmp_path, "mesh_data.jsonl")
    create_data(X, Y, data_path)
    
    label_binarizer_path = os.path.join(tmp_path, "label_binarizer.pkl")
    create_label_binarizer(data_path, label_binarizer_path)
    return label_binarizer_path

@pytest.fixture
def mesh_label_binarizer_path(tmp_path):
    mesh_data_path = os.path.join(tmp_path, "mesh_data.jsonl")
    create_data(X, Y_mesh, mesh_data_path)
    
    mesh_label_binarizer_path = os.path.join(tmp_path, "mesh_label_binarizer.pkl")
    create_label_binarizer(mesh_data_path, mesh_label_binarizer_path)
    return mesh_label_binarizer_path


def test_predict_tags_science_ensemble(science_ensemble_path, label_binarizer_path):
    tags = predict_tags(
        X, model_path=science_ensemble_path,
        label_binarizer_path=label_binarizer_path,
        approach="science-ensemble")
    assert len(tags) == 5
    tags = predict_tags(
        X, model_path=science_ensemble_path,
        label_binarizer_path=label_binarizer_path,
        approach="science-ensemble", probabilities=True)
    for tags_ in tags:
        for tag, prob in tags_.items():
            assert 0 <= prob <= 1.0
    tags = predict_tags(
        X, model_path=science_ensemble_path,
        label_binarizer_path=label_binarizer_path,
        approach="science-ensemble", threshold=0)
    for tags_ in tags:
        assert len(tags_) == 24
    tags = predict_tags(
        X, model_path=science_ensemble_path,
        label_binarizer_path=label_binarizer_path,
        approach="science-ensemble", threshold=1)
    for tags_ in tags:
        assert len(tags_) == 0
    

def test_predict_tags_mesh_tfidf_svm(mesh_tfidf_svm_path, mesh_label_binarizer_path):
    tags = predict_tags(
        X, mesh_tfidf_svm_path, mesh_label_binarizer_path, 
        approach="mesh-tfidf-svm")
    assert len(tags) == 5
    tags = predict_tags(
        X, mesh_tfidf_svm_path, mesh_label_binarizer_path, 
        approach="mesh-tfidf-svm", probabilities=True)
    for tags_ in tags:
        for tag, prob in tags_.items():
            assert 0 <= prob <= 1.0
    tags = predict_tags(
        X, mesh_tfidf_svm_path, mesh_label_binarizer_path, 
        approach="mesh-tfidf-svm", threshold=0)
    for tags_ in tags:
        assert len(tags_) == 5000
    tags = predict_tags(
        X, mesh_tfidf_svm_path, mesh_label_binarizer_path, 
        approach="mesh-tfidf-svm", threshold=1)
    for tags_ in tags:
        assert len(tags_) == 0


def test_predict_tags_mesh_cnn(mesh_cnn_path, mesh_label_binarizer_path):
    tags = predict_tags(
        X, mesh_cnn_path, mesh_label_binarizer_path,
        approach="mesh-cnn")
    assert len(tags) == 5
    tags = predict_tags(
        X, mesh_cnn_path, mesh_label_binarizer_path,
        approach="mesh-cnn", probabilities=True)
    for tags_ in tags:
        for tag, prob in tags_.items():
            assert 0 <= prob <= 1.0
    tags = predict_tags(
        X, mesh_cnn_path, mesh_label_binarizer_path,
        approach="mesh-cnn", threshold=0)
    for tags_ in tags:
        assert len(tags_) == 5000
    tags = predict_tags(
        X, mesh_cnn_path, mesh_label_binarizer_path,
        approach="mesh-cnn", threshold=1)
    for tags_ in tags:
        assert len(tags_) == 0

