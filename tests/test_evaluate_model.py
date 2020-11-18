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


def create_test_data(data_path):
    with open(data_path, "w") as f:
        for x, y in zip(X, Y):
            item = json.dumps({"text": x, "tags": y, "meta": ""})
            f.write(item+"\n")


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

            evaluate_model("model_path", data_path, label_binarizer_path, 0.5)
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

            evaluate_model("model_path", data_path, label_binarizer_path, [0,1,0.5,0.9])
            mock_predict.assert_called()

def test_evaluate_model_predict_cnn():
    with patch('grants_tagger.evaluate_model.predict_cnn') as mock_predict_cnn:
        predict("X_test", "disease_mesh_cnn_model_path", "nb_labels", "threshold")
        mock_predict_cnn.assert_called()

def test_evaluate_model_predict_tfidf_svm():
    with patch('grants_tagger.evaluate_model.predict_tfidf_svm') as mock_predict_tfidf_svm:
        predict("X_test", "disease_mesh_tfidf_model_path", "nb_labels", "threshold")
        mock_predict_tfidf_svm.assert_called()

def test_evaluate_model_predict_ensemble():
    with patch('grants_tagger.evaluate_model.predict_proba_ensemble_tfidf_svm_bert') as mock_predict_ensemble:
        mock_predict_ensemble.return_value = np.random.randn(2, 24)
        predict("X_test", "tfidf,bert_model_paths", "nb_labels", 0.5)
        mock_predict_ensemble.assert_called()
