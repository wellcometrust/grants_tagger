"""
Evaluate model performance on test set
"""
import pickle
import json

from sklearn.metrics import precision_recall_fscore_support
from wasabi import table, row
import scipy.sparse as sp

from grants_tagger.utils import load_train_test_data, load_data
from grants_tagger.models.create_model import load_model


def predict_sparse_probs(model, X_test, batch_size=256, cutoff_prob=0.01):
    Y_pred_proba = []
    for i in range(0, X_test.shape[0], batch_size):
        Y_pred_proba_batch = model.predict_proba(X_test)
        Y_pred_proba_batch[Y_pred_proba_batch < cutoff_prob] = 0
        Y_pred_proba_batch = sp.csr_matrix(Y_pred_proba_batch)
        Y_pred_proba.append(Y_pred_proba_batch)
    Y_pred_proba = sp.vstack(Y_pred_proba_batch)
    return Y_pred_proba


def evaluate_model(
    approach,
    model_path,
    data_path,
    label_binarizer_path,
    threshold,
    split_data=True,
    results_path=None,
    sparse_y=False,
):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    if split_data:
        print(
            "Warning: Data will be split in the same way as train. If you don't want that you set split_data=False"
        )
        _, X_test, _, Y_test = load_train_test_data(data_path, label_binarizer)
    else:
        X_test, Y_test, _ = load_data(data_path, label_binarizer)

    model = load_model(approach, model_path)

    if sparse_y:
        predict_sparse_probs(model, X_test)
    else:
        Y_pred_proba = model.predict_proba(X_test)

    if type(threshold) != list:
        threshold = [threshold]

    widths = (12, 5, 5, 5)
    header = ["Threshold", "P", "R", "F1"]
    print(table([], header, divider=True, widths=widths))

    results = []
    for th in threshold:
        Y_pred = Y_pred_proba > th
        p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average="micro")
        result = {
            "threshold": f"{th:.2f}",
            "precision": f"{p:.2f}",
            "recall": f"{r:.2f}",
            "f1": f"{f1:.2f}",
        }
        results.append(result)

        row_data = (
            result["threshold"],
            result["precision"],
            result["recall"],
            result["f1"],
        )
        print(row(row_data, widths=widths))

    if results_path:
        with open(results_path, "w") as f:
            f.write(json.dumps(results))
