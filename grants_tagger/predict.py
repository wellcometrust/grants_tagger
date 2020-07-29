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
DEFAULT_SCIBERT_PATH = os.path.join(FILEPATH, '../models/scibert-2020.05.5')
DEFAULT_TFIDF_SVM_PATH = os.path.join(FILEPATH, '../models/tfidf-svm-2020.05.2.pkl')
DEFAULT_LABELBINARIZER_PATH = os.path.join(FILEPATH, '../models/label_binarizer.pkl')

DEFAULT_SCIBERT = BertClassifier()
DEFAULT_SCIBERT.load(DEFAULT_SCIBERT_PATH)

with open(DEFAULT_TFIDF_SVM_PATH, "rb") as f:
    DEFAULT_TFIDF_SVM = pickle.loads(f.read())

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

def predict_tags(x, probabilities=False, threshold=0.5,
                 model=[DEFAULT_SCIBERT, DEFAULT_TFIDF_SVM],
                 label_binarizer=DEFAULT_LABELBINARIZER):
    '''Input example text when running the .py file in the terminal to return predicted grant tags - use format:'''
    if type(model) == list: # ensemble of models
        Y_pred_probs = np.zeros((1, len(label_binarizer.classes_)))
        for model_ in model:
            Y_pred_probs_model = model_.predict_proba([x])
            Y_pred_probs += Y_pred_probs_model

        Y_pred_probs /= len(model)
    else:
        Y_pred_probs = model.predict_proba([x])

    tag_names = label_binarizer.classes_

    if probabilities:
        tags = {tag: prob for tag, prob in zip(tag_names, Y_pred_probs[0])}
    else:
        tags = [tag for tag, prob in zip(tag_names, Y_pred_probs[0]) if prob > threshold]
    return tags

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_SCIBERT_PATH,
        help="scikit pickled model or path to model"
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
