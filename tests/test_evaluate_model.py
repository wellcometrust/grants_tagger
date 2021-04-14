from unittest.mock import patch
import tempfile
import pickle
import json
import os

from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pytest

from grants_tagger.evaluate_model import evaluate_model, predict


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

def binarize_Y(Y, label_binarizer_path):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())
    return label_binarizer.transform(Y)

@pytest.fixture
def label_binarizer_path(tmp_path):
    label_binarizer_path = os.path.join(tmp_path, "label_binarizer.pkl")

    label_binarizer = MultiLabelBinarizer()
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
            f.write(item+"\n")
    return data_path

@pytest.fixture
def results_path(tmp_path):
    results_path = os.path.join(tmp_path, "results.json")
    return results_path

def test_evaluate_model(results_path, data_path, label_binarizer_path):
    with patch('grants_tagger.evaluate_model.predict') as mock_predict:
        Y_test = [["1"], ["2"]]
        mock_predict.return_value = binarize_Y(Y_test, label_binarizer_path)

        evaluate_model("mesh-cnn", "model_path", data_path, label_binarizer_path, 0.5, results_path=results_path)
        mock_predict.assert_called()
        
        with open(results_path) as f:
            results = json.loads(f.read())
        assert "f1" in results
        assert "precision" in results
        assert "recall" in results

def test_evaluate_model_multiple_thresholds(data_path, label_binarizer_path):
    with patch('grants_tagger.evaluate_model.predict') as mock_predict:
        Y_test = [["1"], ["2"]]
        mock_predict.return_value = binarize_Y(Y_test, label_binarizer_path)

        evaluate_model("mesh-cnn", "model_path", data_path, label_binarizer_path, [0,1,0.5,0.9])
        mock_predict.assert_called()
