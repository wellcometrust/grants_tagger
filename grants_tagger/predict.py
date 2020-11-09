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


def load_model(model_path):
    if 'pkl' in model_path[-4:]:
        with open(model_path, "rb") as f:
            model = pickle.loads(f.read())
            return model
    if 'scibert' in model_path:
        scibert = BertClassifier(pretrained="scibert")
        scibert.load(model_path)
        return scibert
    raise NotImplementedError


def predict_proba_ensemble_tfidf_svm_bert(X, model_paths):
    Y_pred_proba = []
    for model_path in model_paths:
        model = load_model(model_path)
        Y_pred_proba_model = model.predict_proba(X)
        Y_pred_proba.append(Y_pred_proba_model)

    Y_pred_proba = np.array(Y_pred_proba).sum(axis=0) / len(model_paths)
    return Y_pred_proba


def predict_tags(X, probabilities=False, threshold=0.5,
                 scibert_path=DEFAULT_SCIBERT_PATH,
                 tfidf_svm_path=DEFAULT_TFIDF_SVM_PATH,
                 label_binarizer_path=DEFAULT_LABELBINARIZER_PATH):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.load(f)

    Y_pred_probs = predict_proba_ensemble_tfidf_svm_bert(X, [tfidf_svm_path, scibert_path])

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
