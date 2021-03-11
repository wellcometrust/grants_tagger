# encoding: utf-8
"""
Train a spacy or sklearn model and pickle it
"""
from scipy import sparse as sp
import numpy as np

from pathlib import Path
import pickle
import os.path
import json
import ast

from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

from grants_tagger.models import create_model
from grants_tagger.utils import load_train_test_data, yield_tags


def create_label_binarizer(data_path, label_binarizer_path, sparse=False):        
    label_binarizer = MultiLabelBinarizer(sparse_output=sparse)
    # TODO: pass Y_train here which can be generator or list
    label_binarizer.fit(yield_tags(data_path))

    with open(label_binarizer_path, 'wb') as f:
        f.write(pickle.dumps(label_binarizer))

    return label_binarizer

def train_and_evaluate(
        train_data_path, label_binarizer_path, approach,
        parameters=None, model_path=None, test_data_path=None,
        threshold=None, test_size=0.25, sparse_labels=False,
        cache_path=None, data_format="list", verbose=True):

    if os.path.exists(label_binarizer_path):
        print(f"{label_binarizer_path} exists. Loading existing")
        with open(label_binarizer_path, "rb") as f:
            label_binarizer = pickle.loads(f.read())
    else:
        label_binarizer = create_label_binarizer(train_data_path, label_binarizer_path, sparse_labels)

    model = create_model(approach, parameters)

    # X can be (numpy arrays, lists) or generators
    X_train, X_test, Y_train, Y_test = load_train_test_data(
        train_data_path, label_binarizer, test_data_path=test_data_path,
        test_size=test_size, data_format=data_format)
    model.fit(X_train, Y_train)

    # TODO: Can we handle it better?
    if data_format == "generator":
        Y_test_gen = Y_test()
        if sparse_labels:
            Y_test_batches = []
            for Y_test_batch in Y_test_gen:
                Y_test_batch = sp.csr_matrix(Y_test_batch)
                Y_test_batches.append(Y_test_batch)
            Y_test = sp.vstack(Y_test_batches)
        else:
            Y_test = list(Y_test_gen)
    
    if threshold:
        Y_pred_prob = model.predict_proba(X_test)
        Y_pred_test = Y_pred_prob > threshold
    else:
        Y_pred_test = model.predict(X_test)

    f1 = f1_score(Y_test, Y_pred_test, average='micro')
    if verbose:
        report = classification_report(Y_test, Y_pred_test, target_names=label_binarizer.classes_)
        print(report)

    if model_path:
        if str(model_path).endswith('pkl') or str(model_path).endswith('pickle'):
            with open(model_path, 'wb') as f:
                f.write(pickle.dumps(model))
        else:
            model.save(model_path)
    
    return f1
