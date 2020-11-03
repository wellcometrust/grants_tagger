"""
Combine pretrained models and evaluate performance
"""
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path
import pickle

from wellcomeml.ml import BertClassifier, CNNClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
from wasabi import table
import numpy as np

from grants_tagger.utils import load_data
from grants_tagger.predict_mesh import predict_proba_tfidf_svm, predict_proba_cnn
from grants_tagger.predict import predict_proba_ensemble_tfidf_svm_bert

# TODO: Save tfidf as vectorizer in disease_mesh
# TODO: Use Pipeline or class to explore predict for disease_mesh


def evaluate_model(model_path, data_path, label_binarizer_path, threshold):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    X, Y, _ = load_data(data_path, label_binarizer)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

    # comma indicates ensemble of more than one models
    if "," in model_path:
        model_paths = model_path.split(",")
        Y_pred_proba = predict_proba_ensemble_tfidf_svm_bert(X_test, model_paths)
    elif "disease_mesh_cnn" in model_path:
        Y_pred_proba = predict_proba_cnn(X_test, model_path)
    elif "disease_mesh_tfidf" in model_path:
        nb_labels = len(label_binarizer.classes_)
        Y_pred_proba = predict_proba_tfidf_svm(X_test, model_path, nb_labels)
    else:
        model = load_model(model_path)
        Y_pred_proba = model.predict_proba(X_test)

    # comma indicates multiple thresholds to evaluate against
    if "," in threshold:
        results = []
        for threshold in threshold.split(","):
            Y_pred = Y_pred_proba > float(threshold)
            p = round(precision_score(Y_test, Y_pred, average='micro'), 2)
            r = round(recall_score(Y_test, Y_pred, average='micro'), 2)
            f1 = round(f1_score(Y_test, Y_pred, average='micro'), 2)
            results.append((threshold, p, r, f1))
        header = ["Threshold", "P", "R", "F1"]
        print(table(results, header, divider=True))
    else:
        Y_pred = Y_pred_proba > float(threshold)
        print(classification_report(Y_test, Y_pred))
 
if __name__ == '__main__':
    argparser = ArgumentParser(description=__file__)
    argparser.add_argument("--models", type=str, help="comma separated paths to pretrained models")
    argparser.add_argument("--data", type=Path, help="path to data that was used for training")
    argparser.add_argument("--label_binarizer", type=Path, help="path to label binarizer")
    argparser.add_argument("--threshold", type=str, help="threshold or comma separated thresholds used to assign tags")
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

    evaluate_model(models, data, label_binarizer, threshold)
