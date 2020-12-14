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

from wellcomeml.ml.bert_classifier import BertClassifier
from wellcomeml.ml import CNNClassifier
from numpy import hstack, vstack
from scipy.sparse import hstack as sparse_hstack, csr_matrix
import numpy as np

FILEPATH = os.path.dirname(__file__)
DEFAULT_SCIBERT_PATH = os.path.join(FILEPATH, '../models/scibert-2020.05.5')
DEFAULT_TFIDF_SVM_PATH = os.path.join(FILEPATH, '../models/tfidf-svm-2020.05.2.pkl')
DEFAULT_LABELBINARIZER_PATH = os.path.join(FILEPATH, '../models/label_binarizer.pkl')


def load_model(model_path):
    if model_path.endswith('.pkl'):
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


def predict_tfidf_svm(X, model_path, nb_labels, threshold=0.5,
                      return_probabilities=True, y_batch_size=512):
    # TODO: generalise tfidf to vectorizer.pkl
    with open(f"{model_path}/tfidf.pkl", "rb") as f:
        vectorizer = pickle.loads(f.read())

    Y_pred = []
    for tag_i in range(0, nb_labels, y_batch_size):
        with open(f"{model_path}/{tag_i}.pkl", "rb") as f:
            classifier = pickle.loads(f.read())
        X_vec = vectorizer.transform(X)
        if return_probabilities:
            Y_pred_i = classifier.predict_proba(X_vec)
        elif threshold != 0.5:
            Y_pred_i = classifier.predict_proba(X_vec)
            Y_pred_i = csr_matrix(Y_pred_i > threshold)
        else:
            Y_pred_i = classifier.predict(X_vec)
        Y_pred.append(Y_pred_i)

    if return_probabilities:
        Y_pred = hstack(Y_pred)
    else:
        Y_pred = sparse_hstack(Y_pred)
    return Y_pred


def predict_cnn(X, model_path, threshold=0.5,
                return_probabilities=False, x_batch_size=512):
    with open(f"{model_path}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.loads(f.read())
    model = CNNClassifier(
        sparse_y=True, threshold=threshold, batch_size=x_batch_size
    )
    model.load(model_path)

    X_vec = vectorizer.transform(X)
    if return_probabilities:
        Y_pred_proba = []
        for i in range(0, X_vec.shape[0], x_batch_size):
            Y_pred_proba_batch = model.predict_proba(X_vec[i:i+x_batch_size])
            Y_pred_proba.append(Y_pred_proba_batch)
        Y_pred_proba = vstack(Y_pred_proba)
        return Y_pred_proba
    else:
        Y_pred = model.predict(X_vec)
        return Y_pred


def predict(X_test, model_path, nb_labels=None, threshold=0.5, return_probabilities=False):
    if type(model_path) == list:
        Y_pred_proba = predict_proba_ensemble_tfidf_svm_bert(X_test, model_path)
        if return_probabilities:
            Y_pred = Y_pred_proba
        else:
            Y_pred = Y_pred_proba > threshold
    elif "disease_mesh_cnn" in model_path:
        Y_pred = predict_cnn(X_test, model_path, threshold, return_probabilities)
    elif "disease_mesh_tfidf" in model_path:
        Y_pred = predict_tfidf_svm(X_test, model_path, nb_labels, threshold, return_probabilities)
    else:
        model = load_model(model_path)
        Y_pred_proba = model.predict_proba(X_test)
        if return_probabilities:
            Y_pred = Y_pred_proba
        else:
            Y_pred = Y_pred_proba > threshold
    return Y_pred


def predict_tags(X, model_path, label_binarizer_path,
                 probabilities=False, threshold=0.5,
                 y_batch_size=512):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    nb_labels = len(label_binarizer.classes_)

    Y_pred_proba = predict(X, model_path, threshold=threshold, return_probabilities=True, nb_labels=nb_labels)

    tags = []
    for y_pred_proba in Y_pred_proba:
        if probabilities:
            tags_i = {tag: prob for tag, prob in zip(label_binarizer.classes_, y_pred_proba)}
        else:
            tags_i = [tag for tag, prob in zip(label_binarizer.classes_, y_pred_proba) if prob > threshold]
        tags.append(tags_i)
    return tags
