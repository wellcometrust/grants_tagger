# from unittest.mock import patch
# import tempfile
# import shutil
# import pickle
# import json
# import os

# from transformers import AutoTokenizer, AutoConfig
# import numpy as np
# import pytest
# import torch

# from grants_tagger.train import create_label_binarizer, train
# from grants_tagger.predict import predict_tags
# from grants_tagger.bertmesh.model import BertMesh

# try:
#     from grants_tagger.models.mesh_xlinear import MeshXLinear

#     MESH_XLINEAR_IMPORTED = True
# except ImportError:
#     MESH_XLINEAR_IMPORTED = False

# X = ["all", "one two", "two", "four", "twenty four"]

# Y = [[str(i) for i in range(24)], ["1", "2"], ["2"], ["4"], ["23"]]

# Y_mesh = [[str(i) for i in range(5000)], ["1", "2"], ["2"], ["200"], ["1000"]]


# def create_data(X, Y, data_path):
#     with open(data_path, "w") as f:
#         for x, y in zip(X, Y):
#             f.write(json.dumps({"text": x, "tags": y, "meta": {}}))
#             f.write("\n")
