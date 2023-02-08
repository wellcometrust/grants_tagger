import pickle
import json
import os

from scipy.sparse import csr_matrix
import numpy as np
import pytest

from grants_tagger.train import train, create_label_binarizer


@pytest.fixture
def label_binarizer_path(tmp_path, data_path):
    label_binarizer_path = os.path.join(tmp_path, "label_binarizer.pkl")
    return label_binarizer_path


@pytest.fixture
def data_path(tmp_path):
    texts = [
        "this grant is about Malaria",
        "this one is about Malaria and Cholera",
        "mainly about Hepatitis",
        "both Cholera and Hepatitis and maybe about Arbovirus too",
        "new one Dengue",
        "both Dengue Malaria",
        "this grant is about Arbovirus",
    ]
    tags = [
        ["Malaria"],
        ["Malaria", "Cholera"],
        ["Hepatitis"],
        ["Cholera", "Hepatitis", "Arbovirus"],
        ["Dengue"],
        ["Dengue", "Malaria"],
        ["Arbovirus"],
    ]

    meta = [{}, {}, {}]
    data_path = os.path.join(tmp_path, "data.jsonl")

    with open(data_path, "w") as f:
        for text, tags_, meta_ in zip(texts, tags, meta):
            example = {"text": text, "tags": tags_, "meta": meta_}
            f.write(json.dumps(example) + "\n")

    return data_path


def test_create_label_binarizer(tmp_path, data_path):
    label_binarizer_path = os.path.join(tmp_path, "label_binarizer.pkl")
    create_label_binarizer(data_path, label_binarizer_path)

    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    Y = label_binarizer.transform([["Malaria"]])

    assert "Malaria" in label_binarizer.classes_
    assert "Cholera" in label_binarizer.classes_
    assert len(label_binarizer.classes_) == 3
    assert isinstance(Y, np.ndarray)


def test_create_label_binarizer_sparse(tmp_path, data_path):
    label_binarizer_path = os.path.join(tmp_path, "label_binarizer_sparse.pkl")

    create_label_binarizer(data_path, label_binarizer_path, sparse=True)

    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    Y = label_binarizer.transform([["Malaria"]])

    assert "Malaria" in label_binarizer.classes_
    assert "Cholera" in label_binarizer.classes_
    assert len(label_binarizer.classes_) == 3
    assert isinstance(Y, csr_matrix)


def test_train_and_evaluate(data_path, tmp_path, label_binarizer_path):
    approach = "mesh-xlinear"
    train(
        approach=approach,
        train_data_path=data_path,
        label_binarizer_path=label_binarizer_path,
        sparse_labels=True,
        parameters="{'vectorizer_library': 'sklearn', 'ngram_range': (1, 1),'beam_size': 30, 'threshold': 0.1, 'only_topk': 200, 'min_df':1, 'min_weight_value': 0.1, 'max_features':200}",
    )


def test_train_pickle_save(tmp_path, data_path, label_binarizer_path):
    approach = "mesh-xlinear"
    model_path = tmp_path
    os.makedirs(model_path, exist_ok=True)
    train(
        approach=approach,
        train_data_path=data_path,
        model_path=model_path,
        label_binarizer_path=label_binarizer_path,
        sparse_labels=True,
        parameters="{'vectorizer_library': 'sklearn', 'ngram_range': (1, 1),'beam_size': 30, 'threshold': 0.1, 'only_topk': 200, 'min_df':1, 'min_weight_value': 0.1, 'max_features':200}",
    )
    assert os.path.exists(model_path)


def test_train_model_save(tmp_path, data_path, label_binarizer_path):
    approach = "mesh-xlinear"
    model_path = tmp_path
    os.makedirs(model_path, exist_ok=True)
    train(
        approach=approach,
        train_data_path=data_path,
        model_path=model_path,
        label_binarizer_path=label_binarizer_path,
        sparse_labels=True,
        parameters="{'vectorizer_library': 'sklearn', 'ngram_range': (1, 1),'beam_size': 30, 'threshold': 0.1, 'only_topk': 200, 'min_df':1, 'min_weight_value': 0.1, 'max_features':200}",
    )

    expected_vectorizer_path = os.path.join(tmp_path, "vectorizer.pkl")
    expected_model_variables_path = os.path.join(tmp_path, "params.json")
    expected_model_ranker_path = os.path.join(tmp_path, "ranker")
    assert os.path.exists(expected_vectorizer_path)
    assert os.path.exists(expected_model_variables_path)
    assert os.path.exists(expected_model_ranker_path)
