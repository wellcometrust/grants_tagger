"""
Predict function for disease part of mesh that optionally
exposes probabilities and that you can set the threshold 
for making a prediction
"""
import pickle
import os

from wellcomeml.ml import CNNClassifier
from numpy import hstack, vstack
import numpy as np


def predict_proba_tfidf_svm(X, model_path, nb_labels, y_batch_size=512):
    # TODO: generalise tfidf to vectorizer.pkl
    with open(f"{model_path}/tfidf.pkl", "rb") as f:
        vectorizer = pickle.loads(f.read())

    Y_pred_proba = []
    for tag_i in range(0, nb_labels, y_batch_size):
        with open(f"{model_path}/{tag_i}.pkl", "rb") as f:
            classifier = pickle.loads(f.read())
        X_vec = vectorizer.transform(X)
        Y_pred_i = classifier.predict_proba(X_vec)
        Y_pred_proba.append(Y_pred_i)
    Y_pred_proba = hstack(Y_pred_proba)
    return Y_pred_proba

def predict_proba_cnn(X, model_path, x_batch_size=512):
    with open(f"{model_path}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.loads(f.read())
    model = CNNClassifier(sparse_y=True)
    model.load(model_path)

    X_vec = vectorizer.transform(X)
    Y_pred_proba = []
    for i in range(0, X_vec.shape[0], x_batch_size):
        Y_pred_proba_batch = model.predict_proba(X_vec[i:i+x_batch_size])
        Y_pred_proba.append(Y_pred_proba_batch)
    Y_pred_proba = vstack(Y_pred_proba)

def predict_mesh_tags(X, model_path, label_binarizer_path,
                      probabilities=False, threshold=0.5,
                      y_batch_size=512):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    nb_labels = len(label_binarizer.classes_)

    if "tfidf" in str(model_path):
        Y_pred_proba = predict_proba_tfidf_svm(X, model_path, nb_labels)
    elif "cnn" in str(model_path):
        Y_pred_proba = predict_proba_cnn(X, model_path)
    else:
        raise NotImplementedError

    tags = []
    for y_pred_proba in Y_pred_proba:
        if probabilities:
            tags_i = {tag: prob for tag, prob in zip(label_binarizer.classes_, y_pred_proba)}
        else:
            tags_i = [tag for tag, prob in zip(label_binarizer.classes_, y_pred_proba) if prob > threshold]
        tags.append(tags_i)
    return tags
