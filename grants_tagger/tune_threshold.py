"""
Tune threshold of multilabel classifier to maximise f1

based on paper "Threshold optimisation for multi-label classifiers"
by Pillai https://doi.org/10.1016/j.patcog.2013.01.012

There are currently two deviations from the paper
* fixed number of thresholds to be tried instead of all values from Y_pred_proba for efficiency
* initialisation of thresholds with 0.5 which seems to help convergence in large number of labels
"""
from functools import partial
import multiprocessing
import pickle
import random
import logging
import time
import os

from sklearn.metrics import f1_score
from scipy.sparse import issparse, csc_matrix
from tqdm import tqdm
import numpy as np
import typer
from typing import List, Optional
from pathlib import Path


from grants_tagger.models.create_model_xlinear import load_model
from grants_tagger.utils import load_train_test_data, load_data

logger = logging.getLogger(__name__)
logger.setLevel(
    os.environ.get("LOGGING_LEVEL", logging.INFO)
)  # this should inherit from root

random.seed(41)


def confusion_matrix(y_true, y_pred):
    """
    Args:
        y_true: 1D numpy array or sparse matrix (n_examples, 1) of bool or int (0,1)
        y_pred: 1D numpy array or sparse matrix (n_examples, 1) of bool or int (0,1)
    """
    tp = y_true.dot(y_pred)
    fp = y_pred.sum() - tp
    fn = y_true.sum() - tp
    tn = y_true.shape[0] - tp - fp - fn
    # to make it interchangeable with sklearn
    return np.array([[tn, fp], [fn, tp]])


def multilabel_confusion_matrix(Y_true, Y_pred):
    """
    Args:
        Y_true: sparse csr_matrix of bool or int (0,1)
        Y_pred: sparse csr_matrix of bool or int (0,1)
    """
    if issparse(Y_true):
        tp = Y_true.multiply(Y_pred).sum(axis=0)
    else:
        tp = np.multiply(Y_true, Y_pred).sum(axis=0)
    fp = Y_pred.sum(axis=0) - tp
    fn = Y_true.sum(axis=0) - tp
    tn = Y_true.shape[0] - tp - fp - fn
    # to make it interchangeable with sklearn
    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)


def val_test_split(X_test, Y_test, val_size):
    test_size = Y_test.shape[0]
    indices = list(range(test_size))
    random.shuffle(indices)
    val_indices = indices[:val_size]
    test_indices = indices[val_size:]

    X_test = np.array(X_test)
    X_val = X_test[val_indices]
    Y_val = Y_test[val_indices, :]
    X_test = X_test[test_indices]
    Y_test = Y_test[test_indices, :]
    return X_test, Y_test, X_val, Y_val


def argmaxf1(
    Y_test, Y_pred_proba, optimal_thresholds, label_i, cm, nb_thresholds, iterations
):
    y_pred_proba = Y_pred_proba[:, label_i]
    if issparse(y_pred_proba):
        y_pred_proba = np.array(y_pred_proba.todense()).ravel()
    y_test = Y_test[:, label_i]
    if issparse(y_test):
        y_test = np.array(y_test.todense()).ravel()

    label_threshold = optimal_thresholds[label_i]
    optimal_threshold = label_threshold

    if nb_thresholds:
        candidate_thresholds = [t / nb_thresholds for t in range(1, nb_thresholds)]
    else:
        candidate_thresholds = np.unique(y_pred_proba)

    y_pred = y_pred_proba > label_threshold

    tn, fp, fn, tp = cm
    max_f1 = tp / (tp + (fp + fn) / 2)

    label_cm = confusion_matrix(y_test, y_pred)
    label_tn, label_fp, label_fn, label_tp = label_cm.ravel()

    for candidate_threshold in candidate_thresholds:
        # no need to check in first iteration where first calibration happens
        if iterations >= 1:
            # paper proves that smaller thresholds will not produce better f1
            if candidate_threshold < label_threshold:
                continue

        y_pred = y_pred_proba > candidate_threshold

        candidate_cm = confusion_matrix(y_test, y_pred)
        candidate_tn, candidate_fp, candidate_fn, candidate_tp = candidate_cm.ravel()

        new_tp = tp + candidate_tp - label_tp
        new_fp = fp + candidate_fp - label_fp
        new_fn = fn + candidate_fn - label_fn
        new_f1 = new_tp / (new_tp + (new_fp + new_fn) / 2)

        if new_f1 > max_f1:
            max_f1 = new_f1
            optimal_threshold = candidate_threshold
            logger.debug(
                f"Label: {label_i:4d} - f1: {max_f1:.6f} - Th: {optimal_threshold:.3f}"
            )

    return optimal_threshold


def optimise_threshold(Y_test, Y_pred_proba, nb_thresholds=None):
    # start with lowest posible threshold per label
    optimal_thresholds = Y_pred_proba.min(axis=0)

    if issparse(optimal_thresholds):
        optimal_thresholds = np.array(optimal_thresholds.todense())

    optimal_thresholds = optimal_thresholds.ravel()

    updated = True

    # Convert to CSC for fast column retrieval when asking for labels
    if issparse(Y_pred_proba):
        Y_pred_proba = Y_pred_proba.tocsc()
    Y_pred = Y_pred_proba > optimal_thresholds
    if issparse(Y_test):
        Y_test = Y_test.tocsc()
        Y_pred = csc_matrix(Y_pred)

    optimal_f1 = f1_score(Y_test, Y_pred, average="micro")
    cm = multilabel_confusion_matrix(Y_test, Y_pred)
    cm = cm.sum(axis=0).ravel()
    print("---Min f1---")
    print(f"{optimal_f1:.3f}\n")

    iterations = 0
    while updated:
        start = time.time()
        logger.debug(f"---Iteration {iterations}---")
        updated = False

        nb_labels = Y_test.shape[1]
        argmaxf1_partial = partial(
            argmaxf1,
            Y_test,
            Y_pred_proba,
            optimal_thresholds,
            cm=cm,
            nb_thresholds=nb_thresholds,
            iterations=iterations,
        )
        with multiprocessing.Pool() as pool:
            new_optimal_thresholds = pool.map(argmaxf1_partial, range(nb_labels))

        for i in range(nb_labels):
            if optimal_thresholds[i] != new_optimal_thresholds[i]:
                updated = True
                break

        optimal_thresholds = np.array(new_optimal_thresholds)

        Y_pred = Y_pred_proba > optimal_thresholds
        if issparse(Y_test):
            Y_pred = csc_matrix(Y_pred)
        cm = multilabel_confusion_matrix(Y_test, Y_pred)
        cm = cm.sum(axis=0).ravel()

        tn, fp, fn, tp = cm
        optimal_f1 = tp / (tp + (fp + fn) / 2)
        time_spent = time.time() - start
        print(
            f"Iteration {iterations} - f1 {optimal_f1:.3f} - time spent {time_spent}s"
        )

        iterations += 1

    print("---Optimal f1 in val set---")
    print(f"{optimal_f1:.3f}\n")

    return optimal_thresholds


def tune_threshold(
    approach,
    data_path,
    model_path,
    label_binarizer_path,
    thresholds_path,
    val_size: float = 0.8,
    nb_thresholds: int = None,
    init_threshold: float = 0.2,
    split_data: bool = False,
):

    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    if split_data:
        # To split in the same way train split in case split was done during train
        _, X_test, _, Y_test = load_train_test_data(data_path, label_binarizer)
    else:
        X_test, Y_test, _ = load_data(data_path, label_binarizer)

    test_size = Y_test.shape[0]
    if val_size < 1:
        val_size = int(test_size * val_size)

    X_val, Y_val, X_test, Y_test = val_test_split(X_test, Y_test, val_size)

    model = load_model(approach, model_path)
    Y_pred_proba = model.predict_proba(X_val)
    Y_pred = Y_pred_proba > init_threshold

    f1 = f1_score(Y_val, Y_pred, average="micro")
    print("---Starting f1---")
    print(f"{f1:.3f}\n")

    optimal_thresholds = optimise_threshold(Y_val, Y_pred_proba, nb_thresholds)

    Y_pred_proba = model.predict_proba(X_test)
    Y_pred = Y_pred_proba > optimal_thresholds

    optimal_f1 = f1_score(Y_test, Y_pred, average="micro")
    print("---Optimal f1 in test set---")
    print(f"{optimal_f1:.3f}")

    with open(thresholds_path, "wb") as f:
        f.write(pickle.dumps(optimal_thresholds))


tune_threshold_app = typer.Typer()


@tune_threshold_app.command()
def tune_threshold_cli(
    approach: str = typer.Argument(..., help="modelling approach e.g. mesh-cnn"),
    data_path: Path = typer.Argument(
        ..., help="path to data in jsonl to train and test model"
    ),
    model_path: Path = typer.Argument(
        ..., help="path to data in jsonl to train and test model"
    ),
    label_binarizer_path: Path = typer.Argument(..., help="path to label binarizer"),
    thresholds_path: Path = typer.Argument(..., help="path to save threshold values"),
    val_size: Optional[float] = typer.Option(
        0.8, help="validation size of text data to use for tuning"
    ),
    nb_thresholds: Optional[int] = typer.Option(
        None, help="number of thresholds to be tried divided evenly between 0 and 1"
    ),
    init_threshold: Optional[float] = typer.Option(
        0.2, help="initial threshold value to compare against"
    ),
    split_data: bool = typer.Option(
        False, help="flag on whether to split data as was done for train"
    ),
):

    tune_threshold(
        approach,
        data_path,
        model_path,
        label_binarizer_path,
        thresholds_path,
        val_size,
        nb_thresholds,
        init_threshold,
        split_data,
    )


if __name__ == "__main__":
    tune_threshold_app()
