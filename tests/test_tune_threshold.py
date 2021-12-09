from sklearn.metrics import confusion_matrix as sk_confusion_matrix, multilabel_confusion_matrix as sk_multilabel_confusion_matrix
import  scipy.sparse as sp
import numpy as np

from grants_tagger.tune_threshold import optimise_threshold, confusion_matrix, multilabel_confusion_matrix, val_test_split


def test_random_Y():
    Y_test = np.random.rand(2, 3) > 0.5
    Y_pred_proba = np.random.rand(2, 3)

    thresholds = optimise_threshold(Y_test, Y_pred_proba)
    assert len(thresholds) == 3


def test_ones_Y():
    Y_test = np.ones((2, 3))
    Y_pred_proba = np.array([
        [0.3, 0.5, 0.7],
        [0.1, 0.8, 0.2]
    ])
    optimal_thresholds = optimise_threshold(Y_test, Y_pred_proba)
    expected_thresholds = [0.1, 0.5, 0.2]
    assert np.array_equal(optimal_thresholds, expected_thresholds)


def test_realistic_Y():
    Y_test = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    Y_pred_proba = np.array([
        [0.3, 0.5, 0.7],
        [0.6, 0.7, 0.3],
        [0.1, 0.8, 0.2]
    ])
    optimal_thresholds = optimise_threshold(Y_test, Y_pred_proba)
    expected_thresholds = [0.3, 0.7, 0.3]
    assert np.array_equal(optimal_thresholds, expected_thresholds)


def test_confusion_matrix():
    y_test = np.random.randint(0, 2, size=(100,))
    y_pred = np.random.randint(0, 2, size=(100,))

    custom_cm = confusion_matrix(y_test, y_pred)
    sk_cm = sk_confusion_matrix(y_test, y_pred)
    assert np.array_equal(custom_cm, sk_cm)


def test_multilabel_confusion_matrix():
    Y_test = np.random.randint(0, 2, size=(100, 1000))
    Y_pred = np.random.randint(0, 2, size=(100, 1000))

    custom_cm = multilabel_confusion_matrix(Y_test, Y_pred)
    sk_cm = sk_multilabel_confusion_matrix(Y_test, Y_pred)
    assert np.array_equal(custom_cm, sk_cm)


def test_multilabel_confusion_matrix_sparse():
    Y_test = sp.csr_matrix(np.random.randint(0, 2, size=(100, 1000)))
    Y_pred = sp.csr_matrix(np.random.randint(0, 2, size=(100, 1000)))

    custom_cm = multilabel_confusion_matrix(Y_test, Y_pred)
    sk_cm = sk_multilabel_confusion_matrix(Y_test, Y_pred)
    assert np.array_equal(custom_cm, sk_cm)
def test_val_test_split():
    pass
