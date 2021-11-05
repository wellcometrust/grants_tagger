import tempfile
import os.path
import pickle

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

from grants_tagger.models.mesh_cnn import MeshCNN
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
        model_path = os.path.join(tmp_dir, "model")
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")

        texts = [example["text"] for example in TRAIN_DATA]
        tags = [example["tags"] for example in TRAIN_DATA]

        label_binarizer = MultiLabelBinarizer(sparse_output=True)
        label_binarizer.fit(tags)
        with open(label_binarizer_path, "wb") as f:
            f.write(pickle.dumps(label_binarizer))

        X = texts
        Y = label_binarizer.transform(tags)

        model = MeshCNN()
        model.fit(X, Y)
        model.save(model_path)

        data = pd.DataFrame.from_records(VAL_DATA)
        data.to_excel(data_path, engine="openpyxl")

        evaluate_mesh_on_grants("mesh-cnn", data_path, model_path, label_binarizer_path)
