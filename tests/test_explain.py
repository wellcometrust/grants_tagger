import json
import os

import pytest

from grants_tagger.train import train_and_evaluate, create_label_binarizer
from grants_tagger.explain import explain

X = [
    "all",
    "one two",
    "two",
    "four",
    "twenty four"
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
def mesh_cnn_path(tmp_path):
    mesh_data_path = os.path.join(tmp_path, "mesh_data.jsonl")
    create_data(X, Y_mesh, mesh_data_path)
 
    label_binarizer_path = os.path.join(tmp_path, "label_binarizer.pkl")
    model_path = os.path.join(tmp_path, "mesh_cnn")
    train_and_evaluate(mesh_data_path, label_binarizer_path,
        approach="mesh-cnn", model_path=model_path,
        sparse_labels=True, verbose=False,
        parameters="{'vec__tokenizer_library': 'transformers'}")
    return model_path

@pytest.fixture
def mesh_label_binarizer_path(tmp_path):
    mesh_data_path = os.path.join(tmp_path, "mesh_data.jsonl")
    create_data(X, Y_mesh, mesh_data_path)
    
    mesh_label_binarizer_path = os.path.join(tmp_path, "mesh_label_binarizer.pkl")
    create_label_binarizer(mesh_data_path, mesh_label_binarizer_path)
    return mesh_label_binarizer_path

@pytest.fixture
def texts_path(tmp_path):
    texts = [
        "HIV is a disease caused by a virus",
        "malaria is transmitted by mosquitoes"
    ]
    texts_path = os.path.join(tmp_path, "texts.txt")
    with open(texts_path, "w") as f:
        for text in texts:
            f.write(text)
            f.write("\n")
    return texts_path

@pytest.fixture
def explanations_path(tmp_path):
    return os.path.join(tmp_path, "explanations.html")

def test_explain(mesh_cnn_path, mesh_label_binarizer_path, texts_path,
        explanations_path):
    approach = "mesh-cnn"

    explain(approach, texts_path, mesh_cnn_path, mesh_label_binarizer_path,
            explanations_path, label="1")
    assert os.path.exists(explanations_path)
