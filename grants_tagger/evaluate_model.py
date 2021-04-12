"""
Combine pretrained models and evaluate performance
"""
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
from wasabi import table

from grants_tagger.utils import load_train_test_data, load_data
from grants_tagger.predict import predict


def evaluate_model(approach, model_path, data_path, label_binarizer_path,
        threshold, split_data=True, results_path="results.json"):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    if split_data:
        print("Warning: Data will be split in the same way as train. If you don't want that you set split_data=False")
        _, X_test, _, Y_test = load_train_test_data(data_path, label_binarizer)   
    else:
        X_test, Y_test, _ = load_data(data_path, label_binarizer)

    # TODO: Combine the two approaches. In default just print one row.
    if type(threshold) == list:
        results = []
        for threshold_ in threshold:
            # TODO: predict_proba can run once and then apply threshold only
            Y_pred = predict(X_test, model_path, approach, threshold_)
            p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average="micro")
            results.append((threshold_, f"{p:.2f}", f"{r:.2f}", f"{f1:.2f}"))
        header = ["Threshold", "P", "R", "F1"]
        print(table(results, header, divider=True))
    else:
        Y_pred = predict(X_test, model_path, approach, threshold)
        print(classification_report(Y_test, Y_pred))        

        p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average="micro")
        with open(results_path, "w") as f:
            results = {
                "precision": p,
                "recall": r,
                "f1": f1
            }
            f.write(json.dumps(results))
