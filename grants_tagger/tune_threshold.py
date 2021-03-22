"""
Tune threshold of multilabel classifier to maximise f1

based on paper "Threshold optimisation for multi-label classifiers"
by Pillai https://doi.org/10.1016/j.patcog.2013.01.012

There are currently two deviations from the paper
* fixed number of thresholds to be tried instead of all values from Y_pred_proba for efficiency
* initialisation of thresholds with 0.5 which seems to help convergence in large number of labels
"""
from pathlib import Path
import argparse
import pickle
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, multilabel_confusion_matrix, confusion_matrix
from scipy.sparse import csr_matrix, issparse
import numpy as np

from grants_tagger.evaluate_model import predict
from grants_tagger.utils import load_train_test_data


def argmaxf1(thresholds, Y_test, Y_pred_proba, label_i, nb_thresholds, iterations):
    candidate_thresholds = thresholds.copy()
    current_threshold_i = thresholds[label_i]
    max_f1 = 0
    optimal_threshold_i = current_threshold_i

    Y_pred = Y_pred_proba > candidate_thresholds
    Y_pred = csr_matrix(Y_pred)
    cm = multilabel_confusion_matrix(Y_test, Y_pred)

    if nb_thresholds:
        candidate_thresholds_i = [t/nb_thresholds for t in range(1, nb_thresholds)]
    else:
        candidate_thresholds_i = np.unique(Y_pred_proba[:, label_i])
    for candidate_threshold_i in candidate_thresholds_i:
        # no need to check in first iteration where first calibration happens
        if iterations >= 1:
            # paper proves that smaller thresholds will not produce better f1
            if candidate_threshold_i < current_threshold_i:
                continue
        candidate_thresholds[label_i] = candidate_threshold_i

        y_pred = Y_pred_proba[:, label_i] > candidate_threshold_i
        y_test = Y_test[:, label_i]
        if issparse(y_test):
            y_test = np.array(y_test.todense()).ravel()
        cm_i = confusion_matrix(y_test, y_pred)
        cm[label_i, :, :] = cm_i

        tn, fp, fn, tp = cm.sum(axis=0).ravel()
        f1 = tp / (tp + (fp+fn) / 2)

        if f1 > max_f1:
            max_f1 = f1
            optimal_threshold_i = candidate_threshold_i
    return optimal_threshold_i, max_f1


def optimise_threshold(Y_test, Y_pred_proba, nb_thresholds=None, init_threshold=None):
    if init_threshold:
        optimal_thresholds = [init_threshold] * Y_pred_proba.shape[1]
    else:
        # start with lowest posible threshold per label
        optimal_thresholds = Y_pred_proba.min(axis=0).ravel()
    updated = True

    Y_pred = Y_pred_proba > optimal_thresholds
    optimal_f1 = f1_score(Y_test, Y_pred, average="micro")
    print("---Starting f1---")
    print(f"{optimal_f1:.3f}\n")
    
    iterations = 0
    while updated:
        print(f"---Iteration {iterations}---")
        updated = False
        for label_i in range(Y_test.shape[1]):
            # find threshold for label that maximises overall f1
            optimal_threshold_i, max_f1 = argmaxf1(optimal_thresholds, Y_test, Y_pred_proba, label_i, nb_thresholds, iterations)
            if max_f1 > optimal_f1:
                print(f"Label: {label_i:4d} - f1: {max_f1:.6f} - Th: {optimal_threshold_i:.3f}")
                optimal_f1 = max_f1
                optimal_thresholds[label_i] = optimal_threshold_i
                updated = True
        iterations += 1

    return optimal_thresholds


def tune_threshold(approach, data_path, model_path, label_binarizer_path, thresholds_path, sample_size=None, nb_thresholds=None, init_threshold=None):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    _, X_test, _, Y_test = load_train_test_data(data_path, label_binarizer)
    if not sample_size:
        sample_size = Y_test.shape[0]

    sample_indices = random.sample(list(range(Y_test.shape[0])), sample_size)

    X_test = np.array(X_test)
    X_test_sample = X_test[sample_indices]
    Y_test_sample = Y_test[sample_indices, :]

    Y_pred_proba = predict(X_test_sample, model_path, approach, return_probabilities=True)

    optimal_thresholds = optimise_threshold(Y_test_sample, Y_pred_proba, nb_thresholds, init_threshold)

    Y_pred = predict(X_test, model_path, approach, threshold=optimal_thresholds)
    
    optimal_f1 = f1_score(Y_test, Y_pred, average="micro")
    print("---Optimal f1---")
    print(f"{optimal_f1:.3f}")

    with open(thresholds_path, "wb") as f:
        f.write(pickle.dumps(optimal_thresholds))
