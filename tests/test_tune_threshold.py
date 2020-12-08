import numpy as np

from grants_tagger.tune_threshold import optimise_threshold


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
