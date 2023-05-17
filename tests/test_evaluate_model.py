from unittest.mock import patch
import tempfile
import pickle
import json
import os

from sklearn.preprocessing import MultiLabelBinarizer
import pytest

from grants_tagger.evaluate_model import evaluate_model
from grants_tagger.models.create_model_transformer import create_model
from grants_tagger.utils import load_pickle

X = [
    "this grant is about Malaria",
    "this one is about Malaria and Cholera",
    "mainly about Hepatitis",
    "both Cholera and Hepatitis and maybe about Arbovirus too",
    "new one Dengue",
    "both Dengue Malaria",
    "this grant is about Arbovirus",
]
Y = [
    ["Malaria"],
    ["Malaria", "Cholera"],
    ["Hepatitis"],
    ["Cholera", "Hepatitis", "Arbovirus"],
    ["Dengue"],
    ["Dengue", "Malaria"],
    ["Arbovirus"],
]


def binarize_Y(Y, label_binarizer_path):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())
    return label_binarizer.transform(Y)


@pytest.fixture
def label_binarizer_path(tmp_path):
    label_binarizer_path = os.path.join(tmp_path, "label_binarizer.pkl")

    label_binarizer = MultiLabelBinarizer(sparse_output=True)
    label_binarizer.fit(Y)
    with open(label_binarizer_path, "wb") as f:
        f.write(pickle.dumps(label_binarizer))
    return label_binarizer_path


@pytest.fixture
def data_path(tmp_path):
    data_path = os.path.join(tmp_path, "data.jsonl")
    with open(data_path, "w") as f:
        for x, y in zip(X, Y):
            item = json.dumps({"text": x, "tags": y, "meta": ""})
            f.write(item + "\n")
    return data_path


@pytest.fixture
def results_path(tmp_path):
    results_path = os.path.join(tmp_path, "results.json")
    return results_path


@pytest.fixture
def model_path(tmp_path):
    model = create_model()
    model.load("Wellcome/WellcomeBertMesh")

    model_path = os.path.join(tmp_path)
    model.save(model_path)

    return model_path


def test_evaluate_model(results_path, data_path, label_binarizer_path, model_path):
    evaluate_model(
        model_path,
        data_path,
        label_binarizer_path,
        0.5,
        results_path=results_path,
        sparse_y=False,
    )

    with open(results_path) as f:
        results = json.loads(f.read())

    assert len(results) == 1

    result = results[0]
    assert "f1" in result
    assert "precision" in result
    assert "recall" in result


def test_evaluate_model_multiple_thresholds(
    results_path, data_path, label_binarizer_path, model_path
):
    evaluate_model(
        model_path,
        data_path,
        label_binarizer_path,
        [0, 1, 0.5, 0.9],
        sparse_y=False,
        results_path=results_path,
    )

    with open(results_path) as f:
        results = json.loads(f.read())

    assert len(results) == 4
    result = results[0]
    assert "f1" in result
    assert "precision" in result
    assert "recall" in result


def test_evalaute_model_sparse_y():
    pass
