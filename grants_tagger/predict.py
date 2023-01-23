"""
Predict function for disease part of mesh that optionally
exposes probabilities and that you can set the threshold
for making a prediction
"""
from pathlib import Path
import configparser
import pickle
import typer

import scipy.sparse as sp
import numpy as np

from grants_tagger.models.create_model_xlinear import load_model
from grants_tagger.models.utils import format_predictions
from typing import Optional


def predict_tags(
    X,
    model_path,
    label_binarizer_path,
    approach,
    probabilities=False,
    threshold=0.5,
    parameters=None,
    config=None,
):
    """
    X: list or numpy array of texts
    model_path: path to trained model
    label_binarizer_path: path to trained label_binarizer
    approach: approach used to train the model
    probabilities: bool, default False. When true probabilities are returned along with tags
    threshold: float, default 0.5. Probability threshold to be used to assign tags.
    parameters: any params required upon model creation
    config: Path to config file
    """
    if config:
        # For some models, it might be necessary to see the parameters before loading it

        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.read(config)
        parameters = cfg["model"]["parameters"]

    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    model = load_model(approach, model_path, parameters=parameters)
    Y_pred_proba = model.predict_proba(X)

    tags = format_predictions(
        Y_pred_proba, label_binarizer, threshold=threshold, probabilities=probabilities
    )

    return tags


predict_app = typer.Typer()


@predict_app.command()
def predict_cli(
    text: str,
    model_path: Path,
    label_binarizer_path: Path,
    approach: str,
    probabilities: Optional[bool] = typer.Option(False),
    threshold: Optional[float] = typer.Option(0.5),
):
    tags = predict_tags(
        [text], model_path, label_binarizer_path, approach, probabilities, threshold
    )
    print(tags[0])


if __name__ == "__main__":
    predict_app()
