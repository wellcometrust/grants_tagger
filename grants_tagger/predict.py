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

from grants_tagger.models.utils import format_predictions
from typing import Optional

import logging

logger = logging.getLogger(__name__)


def predict_tags(
    X,
    model_path,
    probabilities=False,
    threshold=0.5,
    parameters=None,
    config=None,
):
    """
    X: list or numpy array of texts
    model_path: path to trained model
    probabilities: bool, default False. When true probabilities are returned along with tags
    threshold: float, default 0.5. Probability threshold to be used to assign tags.
    parameters: any params required upon model creation
    config: Path to config file
    """
    from grants_tagger.models.create_model_transformer import load_model

    if config:
        # For some models, it might be necessary to see the parameters before loading it

        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.read(config)
        parameters = cfg["model"]["parameters"]

    model = load_model(model_path, parameters=parameters)

    classes = model.model.id2label.values()

    Y_pred_proba = model.predict_proba(X)
    Y_pred_proba = Y_pred_proba.toarray()

    tags = format_predictions(
        Y_pred_proba, classes, threshold=threshold, probabilities=probabilities
    )

    return tags


predict_app = typer.Typer()


@predict_app.command()
def predict_cli(
    text: str,
    model_path: Path,
    probabilities: Optional[bool] = typer.Option(False),
    threshold: Optional[float] = typer.Option(0.5),
):
    tags = predict_tags([text], model_path, probabilities, threshold)
    print(tags[0])


if __name__ == "__main__":
    predict_app()
