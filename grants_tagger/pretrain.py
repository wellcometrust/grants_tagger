"""
Pretrain a model on unlabeled data to improve feature extraction
"""
import configparser
import logging
import typer
import os

logger = logging.getLogger(__name__)

from typing import List, Optional
from pathlib import Path
from wellcomeml.ml.doc2vec_vectorizer import Doc2VecVectorizer
import pandas as pd


def pretrain(data_path, model_path, model_name):
    # TODO: Convert that to assume a JSONL with text field
    data = pd.read_csv(data_path)
    X = data["synopsis"].dropna().drop_duplicates()

    if model_name == "doc2vec":
        model = Doc2VecVectorizer(
            min_count=1,
            window_size=9,
            vector_size=300,
            negative=10,
            sample=1e-4,
            epochs=5,
        )
    else:
        raise NotImplementedError
    model.fit(X)

    model.save(model_path)


pretrain_app = typer.Typer()


@pretrain_app.command()
def pretrain_cli(
    data_path: Optional[Path] = typer.Argument(None, help="data to pretrain model on"),
    model_path: Optional[Path] = typer.Argument(None, help="path to save mode"),
    model_name: Optional[str] = typer.Option(
        "doc2vec", help="name of model to pretrain"
    ),
    config: Optional[Path] = typer.Option(
        None, help="config file with arguments for pretrain"
    ),
):

    if config:
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.read(config)

        try:
            cfg_pretrain = cfg["pretrain"]
        except KeyError:
            cfg_pretrain = {}
        data_path = cfg_pretrain.get("data_path")
        model_path = cfg_pretrain.get("model_path")
        model_name = cfg_pretrain.get("model_name")

    if not model_path:
        print("No pretraining defined. Skipping.")
    elif os.path.exists(model_path):
        print(f"{model_path} exists. Remove if you want to rerun.")
    else:
        pretrain(data_path, model_path, model_name)


if __name__ == "__main__":
    pretrain_app()
