import tempfile
import os.path
import pickle
import pytest
import json

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

try:
    from grants_tagger.models.mesh_xlinear import MeshXLinear
    from grants_tagger.evaluate_mesh_on_grants import evaluate_mesh_on_grants

    pecos_installed = True
except ModuleNotFoundError as e:
    pecos_installed = False

TRAIN_DATA = [
    {
        "text": "this grant is about Malaria",
        "tags": ["Malaria"],
        "meta": {"Tagger1_tags": ["1"]},
    },
    {
        "text": "this one is about Malaria and Cholera",
        "tags": ["Malaria", "Cholera"],
        "meta": {"Tagger1_tags": ["2"]},
    },
    {
        "text": "mainly about Hepatitis",
        "tags": ["Hepatitis"],
        "meta": {"Tagger1_tags": ["3"]},
    },
    {
        "text": "both Cholera and Hepatitis and maybe about Arbovirus too",
        "tags": ["Cholera", "Hepatitis", "Arbovirus"],
        "meta": {"Tagger1_tags": ["4"]},
    },
    {"text": "new one Dengue", "tags": ["Dengue"], "meta": {"Tagger1_tags": ["5"]}},
    {
        "text": "both Dengue Malaria",
        "tags": ["Dengue", "Malaria"],
        "meta": {"Tagger1_tags": ["6"]},
    },
    {
        "text": "this grant is about Arbovirus",
        "tags": ["Arbovirus"],
        "meta": {"Tagger1_tags": ["7"]},
    },
]
VAL_DATA = [
    {"Title": "One", "Synopsis": " ", "Tags KW1": 1},
    {"Title": "Two", "Synopsis": " ", "Tags KW2": 2},
]


@pytest.fixture
def results_path(tmp_path):
    results_path = os.path.join(tmp_path, "results.json")
    return results_path


@pytest.mark.skipif(not pecos_installed, reason="Pecos not installed")
def test_evaluate_mesh_on_grants(results_path):
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

        model = MeshXLinear(min_df=1, max_df=10)
        model.fit(X, Y)
        model.save(tmp_dir)

        data = pd.DataFrame.from_records(VAL_DATA)
        data.to_excel(data_path, engine="openpyxl")

        evaluate_mesh_on_grants(data_path, tmp_dir, label_binarizer_path, results_path)

        with open(results_path) as f:
            results = json.loads(f.read())

        assert len(results) == 1
        assert "f1" in results
