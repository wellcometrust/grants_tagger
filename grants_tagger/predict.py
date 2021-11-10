"""
Predict function for disease part of mesh that optionally
exposes probabilities and that you can set the threshold
for making a prediction
"""
from operator import itemgetter
from pathlib import Path
import argparse
import pickle
import os

import scipy.sparse as sp
import numpy as np

from grants_tagger.models.create_model import load_model


def predict_tags(
    X, model_path, label_binarizer_path, approach, probabilities=False, threshold=0.5
):
    """
    X: list or numpy array of texts
    model_path: path to trained model
    label_binarizer_path: path to trained label_binarizer
    approach: approach used to train the model
    probabilities: bool, default False. When true probabilities are returned along with tags
    threshold: float, default 0.5. Probability threshold to be used to assign tags.
    """
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    model = load_model(approach, model_path)
    Y_pred_proba = model.predict_proba(X)

    # TODO: Now that all models accept threshold, is that needed?
    tags = []
    for y_pred_proba in Y_pred_proba:
        if sp.issparse(y_pred_proba):
            y_pred_proba = np.asarray(y_pred_proba.todense()).ravel()
        if probabilities:
            tags_i = {
                tag: prob
                for tag, prob in zip(label_binarizer.classes_, y_pred_proba)
                if prob >= threshold
            }
        else:
            tags_i = [
                tag
                for tag, prob in zip(label_binarizer.classes_, y_pred_proba)
                if prob >= threshold
            ]
        tags.append(tags_i)
    return tags
