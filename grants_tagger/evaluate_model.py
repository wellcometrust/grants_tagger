"""
Combine pretrained models and evaluate performance
"""
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path
import pickle

from wellcomeml.ml import BertClassifier, CNNClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import numpy as np

from grants_tagger.utils import load_data
from grants_tagger.predict_mesh import predict_proba_tfidf_svm, predict_proba_cnn

# TODO: Save tfidf as vectorizer in disease_mesh
# TODO: Use Pipeline or class to explore predict for disease_mesh

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

def evaluate_model(model_path, data_path, label_binarizer_path, threshold):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    X, Y, _ = load_data(data_path, label_binarizer)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

    # comma indicates ensemble of more than one models
    if "," in model_path:        
        Y_pred_proba = np.zeros(Y_test.shape)
        for model_path_ in model_path.split(","):
            model = load_model(model_path_)
            Y_pred_proba_model = model.predict_proba(X_test)
            Y_pred_proba += Y_pred_proba_model

        Y_pred_proba /= len(models)
    elif "disease_mesh_cnn" in model_path:
        Y_pred_proba = predict_proba_cnn(X_test, model_path)
    elif "disease_mesh_tfidf" in model_path:
        nb_labels = len(label_binarizer.classes_)
        Y_pred_proba = predict_proba_tfidf_svm(X_test, model_path, nb_labels)
    Y_pred = Y_pred_proba > threshold

    print(classification_report(Y_test, Y_pred))
 
if __name__ == '__main__':
    argparser = ArgumentParser(description=__file__)
    argparser.add_argument("--models", type=str, help="comma separated paths to pretrained models")
    argparser.add_argument("--data", type=Path, help="path to data that was used for training")
    argparser.add_argument("--label_binarizer", type=Path, help="path to label binarizer")
    argparser.add_argument("--threshold", type=float, help="threshold used to assign tags")
    argparser.add_argument("--config", type=Path, help="path to config file that defines arguments")
    args = argparser.parse_args()

    if args.config:
        cfg = ConfigParser(allow_no_value=True)
        cfg.read(args.config)

        models = cfg["ensemble"]["models"]
        data = cfg["ensemble"]["data"]
        label_binarizer = cfg["ensemble"]["label_binarizer"]
        threshold = float(cfg["ensemble"]["threshold"])
    else:
        models = args.models
        data = args.data
        label_binarizer = args.label_binarizer
        threshold = args.threshold

    evaluate_model(models, data, label_binarizer, threshold)
