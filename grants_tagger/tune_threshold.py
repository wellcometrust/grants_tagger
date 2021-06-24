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

from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, multilabel_confusion_matrix, confusion_matrix
from scipy.sparse import csr_matrix, issparse
import numpy as np
import pandas as pd

from grants_tagger.evaluate_model import predict
from grants_tagger.utils import load_train_test_data


def argmaxf1(thresholds, Y_test, Y_pred_proba, label_i, nb_thresholds, iterations, min_threshold=0, max_threshold=1):
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
    candidate_thresholds_i = [th for th in candidate_thresholds_i if min_threshold <= th <= max_threshold]

    if (iterations == 0) and (label_i == 0):
        print(f"Candidate thresholds: {candidate_thresholds_i}")

    for candidate_threshold_i in candidate_thresholds_i:
        # no need to check in first iteration where first calibration happens
        if iterations >= 1:
            # paper proves that smaller thresholds will not produce better f1
            if candidate_threshold_i < current_threshold_i:
                continue
            if (candidate_threshold_i > max_threshold) or (candidate_threshold_i < min_threshold):
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


def optimise_threshold(Y_val, Y_pred_proba, nb_thresholds=None, init_threshold=None, min_threshold=0, max_threshold=1):
    if init_threshold:
        optimal_thresholds = [init_threshold] * Y_pred_proba.shape[1]
    else:
        # start with lowest posible threshold per label
        optimal_thresholds = Y_pred_proba.min(axis=0).ravel()
    updated = True

    Y_pred = Y_pred_proba > optimal_thresholds
    optimal_f1 = f1_score(Y_val, Y_pred, average="micro")
    print("---Starting f1---")
    print(f"{optimal_f1:.3f}\n")

    labels_count = Y_val.sum(axis=0)

    previous_f1 = 0
    iterations = 0
    while updated:
        print(f"---Iteration {iterations}---")
        updated = False
        for label_i in range(Y_val.shape[1]):
            #if labels_count[label_i] < 15:
            #    print(f"Skipping {label_i} since {labels_count[label_i]} < 15 examples")
            #    continue
            # find threshold for label that maximises overall f1
            optimal_threshold_i, max_f1 = argmaxf1(optimal_thresholds, Y_val, Y_pred_proba, label_i, nb_thresholds, iterations, min_threshold, max_threshold)
            if max_f1 > optimal_f1:
                optimal_f1 = max_f1
                optimal_thresholds[label_i] = optimal_threshold_i 
                print(f"Label: {label_i:4d} - val f1: {max_f1:.6f} - Th: {optimal_threshold_i:.3f}")
                updated = True
        iterations += 1
        
    return optimal_thresholds

def val_test_split(X_test, Y_test, val_size, shuffle=True, stratify=True):
    if val_size > 1:
        val_size = val_size / Y_test.shape[0]
    
    if stratify:
        X_test = np.array(X_test).reshape(-1, 1)
        if shuffle:
            test_indices = list(range(Y_test.shape[0]))
            random.shuffle(test_indices)
            X_test = X_test[test_indices]
            Y_test = Y_test[test_indices]
        X_val, Y_val, X_test, Y_test = iterative_train_test_split(X_test, Y_test, test_size = 1-val_size)
        X_val = X_val.reshape(-1)
        X_test = X_test.reshape(-1)
    else:
        X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = 1-val_size, shuffle=True)
        X_val = np.array(X_val)
        X_test = np.array(X_test)
    return X_val, Y_val, X_test, Y_test

def tune_threshold(approach, data_path, model_path, label_binarizer_path, thresholds_path,
        val_size=None, nb_thresholds=None, init_threshold=None, split_data=True, n_splits=3,
        stratify=True, sample_size=None, min_threshold=0, max_threshold=1, verbose=False):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    if split_data:
        print("Warning: Data will be split in the same way as train. If you don't want that you set split_data=False")
        _, X_test, _, Y_test = load_train_test_data(data_path, label_binarizer)
    else:
        X_test, Y_test = load_data(data_path, label_binarizer)

    Y_pred = predict(X_test, model_path, approach)
    f1 = f1_score(Y_test, Y_pred, average="micro")
    print("---Current f1---")
    print(f"{f1:.3f}")
  
    print(f"Stratify: {stratify}")
    X_val, Y_val, X_test, Y_test = val_test_split(X_test, Y_test, val_size, stratify=stratify)

    print(f"Validation set: {Y_val.shape[0]} examples")
    print(f"Test set: {Y_test.shape[0]} examples")

    if n_splits > 1:
        optimal_thresholds = []

        def yield_splits(X_val, Y_val, n_splits, sample_size=None, stratify=False):
            if sample_size:
                for _ in range(n_splits):
                    X_val_split, Y_val_split, _, _ = val_test_split(X_val, Y_val, sample_size, stratify=stratify)
                    yield X_val_split, Y_val_split
            else:
                if stratify:
                    k_fold = IterativeStratification(n_splits=n_splits, order=2)
                    k_fold_iterator = k_fold.split(X_val, Y_val)
                else:
                    k_fold = KFold(n_splits=n_splits)
                    k_fold_iterator = k_fold.split(X_val)
                for i, (fold_indices, _) in enumerate(k_fold_iterator):
                    X_val_split = X_val[fold_indices]
                    Y_val_split = Y_val[fold_indices, :]
                    yield X_val_split, Y_val_split

        for i, (X_val_split, Y_val_split) in enumerate(yield_splits(X_val, Y_val, n_splits, sample_size, stratify)):
            print("---------------------")
            print(f"Cross validation set {i}: {Y_val_split.shape[0]} examples")
            
            Y_pred_proba = predict(X_val_split, model_path, approach, return_probabilities=True)

            optimal_thresholds_cv = optimise_threshold(Y_val_split, Y_pred_proba, nb_thresholds, init_threshold, min_threshold, max_threshold)
            optimal_thresholds.append(optimal_thresholds_cv)
        
            Y_pred = predict(X_test, model_path, approach, threshold=optimal_thresholds_cv)

            optimal_f1 = f1_score(Y_test, Y_pred, average="micro")
            print("---Optimal f1 in cross vall---")
            print(f"{optimal_f1:.3f}")
            print("---------------------")
        
        optimal_thresholds = np.array(optimal_thresholds)
        optimal_thresholds_std = np.round(optimal_thresholds.std(axis=0),4)
        optimal_thresholds = optimal_thresholds.mean(axis=0)
    else:
        Y_pred_proba = predict(X_val, model_path, approach, return_probabilities=True)
        optimal_thresholds = optimise_threshold(Y_val, Y_pred_proba, nb_thresholds, init_threshold)
        optimal_thresholds_std = np.zeros(optimal_thresholds.shape)

    Y_pred = predict(X_test, model_path, approach, threshold=optimal_thresholds)

    optimal_f1 = f1_score(Y_test, Y_pred, average="micro")
    print("---Optimal f1---")
    print(f"{optimal_f1:.3f}")

    verbose = True
    if verbose:
        print(pd.DataFrame(list(zip(label_binarizer.classes_, optimal_thresholds, optimal_thresholds_std, Y_val.sum(axis=0))), columns=["Label", "Threshold", "Std", "Count"]))

    with open(thresholds_path, "wb") as f:
        f.write(pickle.dumps(optimal_thresholds))
