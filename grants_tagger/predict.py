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


def predict_tags(X, probabilities=False, threshold=0.5,
                 scibert_path=DEFAULT_SCIBERT_PATH,
                 tfidf_svm_path=DEFAULT_TFIDF_SVM_PATH,
                 label_binarizer_path=DEFAULT_LABELBINARIZER_PATH):
    scibert = BertClassifier()
    scibert.load(scibert_path)

    with open(tfidf_svm_path, "rb") as f:
        tfidf_svm = pickle.loads(f.read())

    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.load(f)

    Y_pred_probs = np.zeros((len(X), len(label_binarizer.classes_)))
    for model in [scibert, tfidf_svm]:
        Y_pred_probs_model = model.predict_proba(X)
        Y_pred_probs += Y_pred_probs_model

        Y_pred_probs /= 2

    tag_names = label_binarizer.classes_

    if probabilities:
        tags = [
            {tag: prob for tag, prob in zip(tag_names, Y_pred_prob)}
            for Y_pred_prob in Y_pred_probs
        ]
    else:
        tags = [
            [tag for tag, prob in zip(tag_names, Y_pred_prob) if prob > threshold]
            for Y_pred_prob in Y_pred_probs
        ]
    return tags

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--scibert",
        type=Path,
        default=DEFAULT_SCIBERT_PATH,
        help="path to scibert model"
    )
    argparser.add_argument(
        "--tfidf_svm",
        type=Path,
        default=DEFAULT_TFIDF_SVM_PATH,
        help="path to scibert model"
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

    tags = predict_tags([args.synopsis])
    print(tags)
