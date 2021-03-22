import tempfile
import pickle
import json
import math
import os

from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
import numpy as np
import pytest

from grants_tagger.train import train_and_evaluate, create_label_binarizer


@pytest.fixture
def data_path(tmp_path, train_data_path, test_data_path):
    data_path = os.path.join(tmp_path, "data.jsonl")
    with open(data_path, "w") as data_f:
        with open(train_data_path, "r") as train_f:
            for line in train_f:
                data_f.write(line)
        with open(test_data_path, "r") as test_f:
            for line in test_f:
                data_f.write(line)
    return data_path


@pytest.fixture
def label_binarizer_path(tmp_path, data_path):
    label_binarizer_path = os.path.join(tmp_path, "label_binarizer.pkl") 
    create_label_binarizer(data_path, label_binarizer_path)
    return label_binarizer_path


@pytest.fixture
def train_data_path(tmp_path):
    texts = ["one", "one two"]
    tags = [["one"], ["one","two"]]
    meta = [{}, {}]
    test_data_path = os.path.join(tmp_path, "train_data.jsonl")

    with open(test_data_path, "w") as f:
        for text, tags_, meta_ in zip(texts, tags, meta):
            example = {"text": text, "tags": tags_, "meta": meta_}
            f.write(json.dumps(example)+"\n")
    return test_data_path


@pytest.fixture
def test_data_path(tmp_path):
    texts = ["two"]
    tags = [["two"]]
    meta = [{}]

    test_data_path = os.path.join(tmp_path, "test_data.jsonl")

    with open(test_data_path, "w") as f:
        for text, tags_, meta_ in zip(texts, tags, meta):
            example = {"text": text, "tags": tags_, "meta": meta_}
            f.write(json.dumps(example)+"\n")
    return test_data_path


def test_create_label_binarizer(tmp_path, data_path):
    label_binarizer_path = os.path.join(tmp_path, "label_binarizer.pkl")
    create_label_binarizer(data_path, label_binarizer_path)

    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    Y = label_binarizer.transform([["one"]])

    assert "one" in label_binarizer.classes_
    assert "two" in label_binarizer.classes_
    assert len(label_binarizer.classes_) == 2
    assert isinstance(Y, np.ndarray)


def test_create_label_binarizer_sparse(tmp_path, data_path):
    label_binarizer_path = os.path.join(tmp_path, "label_binarizer.pkl")

    create_label_binarizer(data_path, label_binarizer_path, sparse=True)

    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    Y = label_binarizer.transform([["one"]])

    assert "one" in label_binarizer.classes_
    assert "two" in label_binarizer.classes_
    assert len(label_binarizer.classes_) == 2
    assert isinstance(Y, csr_matrix)


def test_train_and_evaluate(data_path, label_binarizer_path):
    approach = "tfidf-svm"

    train_and_evaluate(data_path, label_binarizer_path, approach,
        parameters="{'tfidf__min_df': 1, 'tfidf__stop_words': None}")


def test_train_pickle_save(tmp_path, data_path, label_binarizer_path):
    approach = "tfidf-svm"

    model_path = os.path.join(tmp_path, "model.pkl")
    train_and_evaluate(data_path, label_binarizer_path, approach=approach,
        model_path=model_path, parameters="{'tfidf__min_df': 1, 'tfidf__stop_words': None}")
    assert os.path.exists(model_path)


def test_train_model_save(tmp_path, data_path, label_binarizer_path):
    approach = "mesh-cnn"

    train_and_evaluate(data_path, label_binarizer_path,
        approach, model_path=tmp_path)

    expected_vectorizer_path = os.path.join(tmp_path, "vectorizer.pkl")
    expected_model_variables_path = os.path.join(tmp_path, "variables")
    expected_model_assets_path = os.path.join(tmp_path, "assets")
    assert os.path.exists(expected_vectorizer_path)
    assert os.path.exists(expected_model_variables_path)
    assert os.path.exists(expected_model_assets_path)


def test_train_and_evaluate_generator(train_data_path, test_data_path, label_binarizer_path):
    approach = "mesh-cnn"

    train_and_evaluate(train_data_path, label_binarizer_path, approach,
        data_format="generator", sparse_labels=True, test_data_path=test_data_path)


def test_train_and_evaluate_generator_non_sparse_labels(train_data_path, test_data_path, label_binarizer_path):
    approach = "mesh-cnn"

    train_and_evaluate(train_data_path, label_binarizer_path, approach,
        data_format="generator", test_data_path=test_data_path)


def test_train_and_evaluate_threshold(data_path, label_binarizer_path):
    approach = "tfidf-svm"

    f1 = train_and_evaluate(data_path, label_binarizer_path, approach=approach,
        threshold=1, parameters="{'tfidf__min_df': 1, 'tfidf__stop_words': None}")
    assert f1 == 0
    f1 = train_and_evaluate(data_path, label_binarizer_path, approach=approach,
        threshold=0, parameters="{'tfidf__min_df': 1, 'tfidf__stop_words': None}")
    assert f1 == (2/3) # P: 0.5, R: 1.0, F: 0.66

