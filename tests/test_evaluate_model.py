from unittest.mock import patch
import tempfile
import pickle
import json

from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

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

# TODO: Use fixtures 
def create_test_data(data_path):
    with open(data_path, "w") as f:
        for x, y in zip(X, Y):
            item = json.dumps({"text": x, "tags": y, "meta": ""})
            f.write(item+"\n")

# TODO: patch predict

def test_evaluate_model():
    with tempfile.TemporaryDirectory() as tmp_dir:
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        data_path = f"{tmp_dir}/data.jsonl"

        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit(Y)
        with open(label_binarizer_path, "wb") as f:
            f.write(pickle.dumps(label_binarizer))

        create_test_data(data_path)

        with patch('grants_tagger.evaluate_model.predict') as mock_predict:
            Y_test = [["1"], ["2"]]
            mock_predict.return_value = label_binarizer.transform(Y_test)

            evaluate_model("mesh-cnn", "model_path", data_path, label_binarizer_path, 0.5)
            mock_predict.assert_called()


def test_evaluate_model_multiple_thresholds():
    with tempfile.TemporaryDirectory() as tmp_dir:
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        data_path = f"{tmp_dir}/data.jsonl"

        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit(Y)
        with open(label_binarizer_path, "wb") as f:
            f.write(pickle.dumps(label_binarizer))

        create_test_data(data_path)

        with patch('grants_tagger.evaluate_model.predict') as mock_predict:
            Y_test = [["1"], ["2"]]
            mock_predict.return_value = label_binarizer.transform(Y_test)

            evaluate_model("mesh-cnn", "model_path", data_path, label_binarizer_path, [0,1,0.5,0.9])
            mock_predict.assert_called()
