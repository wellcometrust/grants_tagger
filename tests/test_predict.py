from unittest.mock import patch
import tempfile
import shutil
import pickle
import json
import os

from transformers import AutoTokenizer, AutoConfig
import numpy as np
import pytest
import torch

from grants_tagger.train import create_label_binarizer, train
from grants_tagger.predict import predict_tags
from grants_tagger.bertmesh.model import BertMesh

try:
    from grants_tagger.models.mesh_xlinear import MeshXLinear

    MESH_XLINEAR_IMPORTED = True
except ImportError:
    MESH_XLINEAR_IMPORTED = False

X = ["all", "one two", "two", "four", "twenty four"]

Y = [[str(i) for i in range(24)], ["1", "2"], ["2"], ["4"], ["23"]]

Y_mesh = [[str(i) for i in range(5000)], ["1", "2"], ["2"], ["200"], ["1000"]]


def create_data(X, Y, data_path):
    with open(data_path, "w") as f:
        for x, y in zip(X, Y):
            f.write(json.dumps({"text": x, "tags": y, "meta": {}}))
            f.write("\n")


@pytest.fixture
def mesh_xlinear_path(tmp_path):
    mesh_data_path = os.path.join(tmp_path, "mesh_data.jsonl")
    create_data(X, Y_mesh, mesh_data_path)

    label_binarizer_path = os.path.join(tmp_path, "label_binarizer.pkl")
    model_path = os.path.join(tmp_path, "mesh_xlinear")
    parameters = {"min_df": 1, "stop_words": None, "vectorizer_library": "sklearn"}
    train(
        mesh_data_path,
        label_binarizer_path,
        model_path=model_path,
        sparse_labels=True,
        verbose=False,
        parameters=str(parameters),
    )
    return model_path


@pytest.fixture
def label_binarizer_path(tmp_path):
    data_path = os.path.join(tmp_path, "mesh_data.jsonl")
    create_data(X, Y, data_path)

    label_binarizer_path = os.path.join(tmp_path, "label_binarizer.pkl")
    create_label_binarizer(data_path, label_binarizer_path)
    return label_binarizer_path


@pytest.fixture
def mesh_label_binarizer_path(tmp_path):
    mesh_data_path = os.path.join(tmp_path, "mesh_data.jsonl")
    create_data(X, Y_mesh, mesh_data_path)

    mesh_label_binarizer_path = os.path.join(tmp_path, "mesh_label_binarizer.pkl")
    create_label_binarizer(mesh_data_path, mesh_label_binarizer_path)
    return mesh_label_binarizer_path


# @pytest.mark.skipif(not MESH_XLINEAR_IMPORTED, reason="MeshXLinear missing")
# def test_predict_tags_mesh_xlinear(mesh_xlinear_path, mesh_label_binarizer_path):
#     # We need to pass parameters because the load function is different
#     # depending on the vectorizer library (pecos or sklearn)
#     parameters = str({"vectorizer_library": "sklearn"})
#     tags = predict_tags(
#         X,
#         mesh_xlinear_path,
#         mesh_label_binarizer_path,
#         parameters=parameters,
#     )
#     assert len(tags) == 5
#     tags = predict_tags(
#         X,
#         mesh_xlinear_path,
#         mesh_label_binarizer_path,
#         parameters=parameters,
#         probabilities=True,
#     )
#     for tags_ in tags:
#         for tag, prob in tags_.items():
#             assert 0 <= prob <= 1.0
#     tags = predict_tags(
#         X,
#         mesh_xlinear_path,
#         mesh_label_binarizer_path,
#         threshold=0,
#         parameters=parameters,
#     )
#     for tags_ in tags:
#         assert len(tags_) == 5000
#     tags = predict_tags(
#         X,
#         mesh_xlinear_path,
#         mesh_label_binarizer_path,
#         threshold=1,
#         parameters=parameters,
#     )
#     for tags_ in tags:
#         assert len(tags_) == 0
