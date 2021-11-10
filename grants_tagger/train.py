# encoding: utf-8
"""
Train a spacy or sklearn model and pickle it
"""
from scipy import sparse as sp
import numpy as np

from pathlib import Path
import pickle
import os.path
import json
import ast

from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

from grants_tagger.models.create_model import create_model
from grants_tagger.utils import load_train_test_data, yield_tags

from tensorflow.random import set_seed

# TODO: Remove when WellcomeML implements setting random_seed inside models
# replace with param in configs then
set_seed(41)


def create_label_binarizer(data_path, label_binarizer_path, sparse=False):
    """Creates, saves and returns a multilabel binarizer for targets Y"""
    label_binarizer = MultiLabelBinarizer(sparse_output=sparse)
    # TODO: pass Y_train here which can be generator or list
    label_binarizer.fit(yield_tags(data_path))

    with open(label_binarizer_path, "wb") as f:
        f.write(pickle.dumps(label_binarizer))

    return label_binarizer


def train(
    train_data_path,
    label_binarizer_path,
    approach,
    parameters=None,
    model_path=None,
    threshold=None,
    sparse_labels=False,
    cache_path=None,
    data_format="list",
    verbose=True,
):
    """
    train_data_path: path. path to JSONL data that contains "text" and "tags" fields.
    label_binarizer_path: path. path to load or store label_binarizer.
    approach: str. approach to use for modelling e.g. tfidf-svm or bert.
    parameters: str. a stringified dict that contains params that get passed to the model.
    model_path: path. path to save the model.
    threshold: float, default 0.5. Probability on top of which a tag is assigned.
    sparse_labels: bool, default False. whether tags (labels) would be sparse for memory efficiency.
    cache_path: path, default None. path to use for caching data transformations for speed.
    data_format: str, default list. one of list, generator. generator used for memory efficiency.
    """

    if os.path.exists(label_binarizer_path):
        print(f"{label_binarizer_path} exists. Loading existing")
        with open(label_binarizer_path, "rb") as f:
            label_binarizer = pickle.loads(f.read())
    else:
        label_binarizer = create_label_binarizer(
            train_data_path, label_binarizer_path, sparse_labels
        )

    model = create_model(approach, parameters)

    # X can be (numpy arrays, lists) or generators
    X_train, _, Y_train, _ = load_train_test_data(
        train_data_path, label_binarizer, data_format=data_format
    )
    model.fit(X_train, Y_train)

    if model_path:
        if str(model_path).endswith("pkl") or str(model_path).endswith("pickle"):
            with open(model_path, "wb") as f:
                f.write(pickle.dumps(model))
        else:
            if not os.path.exists(model_path):
                Path(model_path).mkdir(parents=True, exist_ok=True)
            model.save(model_path)


if __name__ == "__main__":
    # Note that this CLI is purely for SageMaker so it is quite minimal
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=Path)
    argparser.add_argument("--model_path", type=Path)
    argparser.add_argument("--label_binarizer_path", type=Path)
    argparser.add_argument("--approach", type=str)
    argparser.add_argument("--parameters", type=str)
    argparser.add_argument("--threshold", type=float)
    argparser.add_argument("--data_format", type=str, default="list")
    argparser.add_argument("--sparse_labels", type=bool)
    argparser.add_argument("--cache_path", type=Path)
    args = argparser.parse_args()

    train(
        args.data_path,
        args.label_binarizer_path,
        args.approach,
        args.parameters,
        threshold=args.threshold,
        model_path=args.model_path,
        data_format=args.data_format,
        sparse_labels=args.sparse_labels,
    )
