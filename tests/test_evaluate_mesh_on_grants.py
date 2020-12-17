import tempfile
import os.path
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd

from grants_tagger.evaluate_mesh_on_grants import evaluate_mesh_on_grants


TRAIN_DATA = [
    {"text": "One", "tags": [1]},
    {"text": "One Two", "tags": [1, 2]},
    {"text": "Three", "tags": [3]}
]
VAL_DATA = [
    {"Title": "One", "Synopsis": " ", "Tags KW1": 1},
    {"Title": "Two", "Synopsis": " ", "Tags KW2": 2}
]


def test_evaluate_mesh_on_grants():
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = os.path.join(tmp_dir, "data.xlsx")
        model_path = os.path.join(tmp_dir, "model.pkl")
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")

        texts = [example["text"] for example in TRAIN_DATA]
        tags = [example["tags"] for example in TRAIN_DATA]

        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit(tags)
        with open(label_binarizer_path, "wb") as f:
            f.write(pickle.dumps(label_binarizer))

        model = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("svm", OneVsRestClassifier(SVC(probability=True)))
        ])
        X = texts
        Y = label_binarizer.transform(tags)
        model.fit(X, Y)
        with open(model_path, "wb") as f:
            f.write(pickle.dumps(model))

        data = pd.DataFrame.from_records(VAL_DATA)
        data.to_excel(data_path, engine="openpyxl")

        evaluate_mesh_on_grants(data_path, model_path, label_binarizer_path)
