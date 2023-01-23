# encoding: utf-8
"""
Train a spacy or sklearn model and pickle it
"""
from pathlib import Path
import pickle
import os.path
import json
import typer
import logging
import yaml
import configparser


logger = logging.getLogger(__name__)

import dvc.api
from typing import Optional

from grants_tagger.label_binarizer import create_label_binarizer
from grants_tagger.models.create_model_xlinear import create_model
from grants_tagger.utils import load_train_test_data, yield_tags

# from tensorflow.random import set_seed
from grants_tagger.utils import convert_dvc_to_sklearn_params


# TODO: Remove when WellcomeML implements setting random_seed inside models
# replace with param in configs then
# set_seed(41)


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


train_app = typer.Typer()


@train_app.command()
def train_cli(
    data_path: Optional[Path] = typer.Argument(
        None, help="path to processed JSON data to be used for training"
    ),
    label_binarizer_path: Optional[Path] = typer.Argument(
        None, help="path to label binarizer"
    ),
    model_path: Optional[Path] = typer.Argument(
        None, help="path to output model.pkl or dir to save model"
    ),
    approach: str = typer.Option("tfidf-svm", help="tfidf-svm, scibert, cnn, ..."),
    parameters: str = typer.Option(
        None, help="model params in sklearn format e.g. {'svm__kernel: linear'}"
    ),
    threshold: float = typer.Option(None, help="threshold to assign a tag"),
    data_format: str = typer.Option(
        "list",
        help="format that will be used when loading the data. One of list,generator",
    ),
    train_info: str = typer.Option(None, help="path to train times and instance"),
    sparse_labels: bool = typer.Option(
        False, help="flat about whether labels should be sparse when binarized"
    ),
    cache_path: Optional[Path] = typer.Option(
        None, help="path to cache data transformartions"
    ),
    config: Path = None,
):

    params = dvc.api.params_show()

    if params.get("train", {}).get(approach, {}).get("config"):
        config = os.path.join(
            os.path.dirname(__file__),
            "../configs",
            params["train"]["mesh-xlinear"]["config"],
        )
    # If parameters not provided from user we initialise from DVC
    if not parameters and not config:
        parameters = params["train"].get(approach)
        parameters = convert_dvc_to_sklearn_params(parameters)
        parameters = str(parameters)

    # Note that config overwrites parameters for backwards compatibility
    if config:
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.read(config)

        data_path = cfg["data"]["train_data_path"]
        label_binarizer_path = cfg["model"]["label_binarizer_path"]
        approach = cfg["model"]["approach"]
        parameters = cfg["model"]["parameters"]
        model_path = cfg["model"].get("model_path", None)
        threshold = cfg["model"].get("threshold", None)
        if threshold:
            threshold = float(threshold)
        data_format = cfg["data"].get("data_format", "list")
        sparse_labels = cfg["model"].get("sparse_labels", False)
        if sparse_labels:
            sparse_labels = bool(sparse_labels)
        cache_path = cfg["data"].get("cache_path")

    if model_path and os.path.exists(model_path):
        print(f"{model_path} exists. Remove if you want to rerun.")
    else:
        logger.info(parameters)
        train(
            data_path,
            label_binarizer_path,
            approach,
            parameters,
            model_path=model_path,
            threshold=threshold,
            data_format=data_format,
            sparse_labels=sparse_labels,
            cache_path=cache_path,
        )


if __name__ == "__main__":
    train_app()
