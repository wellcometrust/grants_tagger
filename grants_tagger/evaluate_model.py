"""
Combine pretrained models and evaluate performance
"""
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
from wasabi import table
import numpy as np

from grants_tagger.utils import load_data
from grants_tagger.predict import predict


def evaluate_model(model_path, data_path, label_binarizer_path, threshold):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())
    nb_labels = len(label_binarizer.classes_)

    X, Y, _ = load_data(data_path, label_binarizer)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

    if type(threshold) == list:
        results = []
        for threshold_ in threshold:
            Y_pred = predict(X_test, model_path, nb_labels, threshold_)
            p = precision_score(Y_test, Y_pred, average='micro')
            r = recall_score(Y_test, Y_pred, average='micro')
            f1 = f1_score(Y_test, Y_pred, average='micro')
            results.append((threshold_, f"{p:.2f}", f"{r:.2f}", f"{f1:.2f}"))
        header = ["Threshold", "P", "R", "F1"]
        print(table(results, header, divider=True))
    else:
        Y_pred = predict(X_test, model_path, nb_labels, threshold)
        print(classification_report(Y_test, Y_pred))
 
if __name__ == '__main__':
    argparser = ArgumentParser(description=__file__)
    argparser.add_argument("--models", type=str, help="comma separated paths to pretrained models")
    argparser.add_argument("--data", type=Path, help="path to data that was used for training")
    argparser.add_argument("--label_binarizer", type=Path, help="path to label binarizer")
    argparser.add_argument("--threshold", type=str, default="0.5", help="threshold or comma separated thresholds used to assign tags")
    argparser.add_argument("--config", type=Path, help="path to config file that defines arguments")
    args = argparser.parse_args()

    if args.config:
        cfg = ConfigParser(allow_no_value=True)
        cfg.read(args.config)

        models = cfg["ensemble"]["models"]
        data = cfg["ensemble"]["data"]
        label_binarizer = cfg["ensemble"]["label_binarizer"]
        threshold = cfg["ensemble"]["threshold"]
    else:
        models = args.models
        data = args.data
        label_binarizer = args.label_binarizer
        threshold = args.threshold

    # comma indicates multiple threshold to evaluate against
    if "," in threshold:
        threshold = [float(t) for t in threshold.split(",")]
    else:
        threshold = float(threshold)
    if "," in models:
        models = models.split(",")
    evaluate_model(models, data, label_binarizer, threshold)
