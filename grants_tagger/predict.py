# coding: utf-8
"""
Predict labels for given text and pretrained model
"""
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from operator import itemgetter
from pathlib import Path
import argparse
import pickle
import os

from wellcomeml.ml.bert_classifier import BertClassifier

FILEPATH = os.path.dirname(__file__)
DEFAULT_MODEL_PATH = os.path.join(FILEPATH, 'models/scibert')
DEFAULT_LABELBINARIZER_PATH = os.path.join(FILEPATH, 'models/label_binarizer.pkl')

DEFAULT_MODEL = BertClassifier()
DEFAULT_MODEL.load(DEFAULT_MODEL_PATH)

with open(DEFAULT_LABELBINARIZER_PATH, "rb") as f:
    DEFAULT_LABELBINARIZER = pickle.load(f)


def sort_tags_probs(tags, probs, threshold):
    """
    Args:
        tags: list of science tags
        probs: list of probabilities
    Returns:
        sorted_tags_probs: list of tuple (tag, prob) sorted in ascending order
    """
    data = zip(tags.tolist()+['------ THRESHOLD ------'], probs.tolist()+[threshold])
    sorted_tags_probs = sorted(data, key = itemgetter(1), reverse=True)
    return sorted_tags_probs

def predict_tags(x, probabilities=False, model=DEFAULT_MODEL, label_binarizer=DEFAULT_LABELBINARIZER):
    '''Input example text when running the .py file in the terminal to return predicted grant tags - use format:'''
    if probabilities:
        Y_pred = model.predict_proba([x])
        tag_names = label_binarizer.classes_
        tags = {tag: prob for tag, prob in zip(tag_names, Y_pred[0])}
    else:
        Y_pred = model.predict([x])
        tags = label_binarizer.inverse_transform(Y_pred)[0]
    return tags

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="scikit pickled model"
    )
    argparser.add_argument(
        "--label_binarizer",
        type=Path,
        default=DEFAULT_LABELBINARIZER_PATH,
        help="label binarizer for Y"
    )
    argparser.add_argument(
        "--synopsis",
        type=str,
        help="synopsis of grant to tag"
    )
    argparser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="threshold value between 0 and 1"
    )

    args = argparser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    with open(args.label_binarizer, "rb") as f:
        label_binarizer = pickle.load(f)

    grant_tags = label_binarizer.classes_
    y_score = model.predict_proba([args.synopsis])[0]
    for tag, prob in sort_tags_probs(grant_tags, y_score, args.threshold):
        print(f"{tag:60s} - {prob:.4f}")
