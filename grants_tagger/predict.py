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

import numpy as np

from wellcomeml.ml import BertClassifier
from grants_tagger.models import MeshCNN, MeshTfidfSVM, ScienceEnsemble 

FILEPATH = os.path.dirname(__file__)
DEFAULT_SCIBERT_PATH = os.path.join(FILEPATH, '../models/scibert-2020.05.5')
DEFAULT_TFIDF_SVM_PATH = os.path.join(FILEPATH, '../models/tfidf-svm-2020.05.2.pkl')
DEFAULT_LABELBINARIZER_PATH = os.path.join(FILEPATH, '../models/label_binarizer.pkl')


def predict(X_test, model_path, approach, threshold=0.5, return_probabilities=False):
    if approach == 'mesh-cnn':
        model = MeshCNN(
            threshold=threshold
        )
        model.load(model_path)
    elif approach == 'mesh-tfidf-svm':
        model = MeshTfidfSVM(
            threshold=threshold,
        )
        model.load(model_path)
    elif approach == 'science-ensemble':
        model = ScienceEnsemble()
        model.load(model_path)
    # part of science-ensemble
    elif approach == 'tfidf-svm':
        with open(model_path, "rb") as f:
            model = pickle.loads(f.read())
    # part of science-ensemble
    elif approach == 'scibert':
        model = BertClassifier(pretrained="scibert")
        model.load(model_path)
    else:
        raise NotImplementedError

    if return_probabilities:
        return model.predict_proba(X_test)
    else:
        return model.predict(X_test)

def predict_tags(
        X, model_path, label_binarizer_path,
        approach, probabilities=False,
        threshold=0.5):
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

    Y_pred_proba = predict(X, model_path, threshold=threshold,
        return_probabilities=True, approach=approach)

    # TODO: Now that all models accept threshold, is that needed?
    tags = []
    for y_pred_proba in Y_pred_proba:
        if probabilities:
            tags_i = {tag: prob for tag, prob in zip(label_binarizer.classes_, y_pred_proba)}
        else:
            tags_i = [tag for tag, prob in zip(label_binarizer.classes_, y_pred_proba) if prob > threshold]
        tags.append(tags_i)
    return tags
