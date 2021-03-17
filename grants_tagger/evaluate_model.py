"""
Combine pretrained models and evaluate performance
"""
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
from wasabi import table

from grants_tagger.utils import load_train_test_data, load_data
from grants_tagger.predict import predict


def evaluate_model(approach, model_path, data_path, label_binarizer_path,
        threshold, split_data=True):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    if split_data:
        print("Warning: Data will be split in the same way as train. If you don't want that you set split_data=False")
        _, X_test, _, Y_test = load_train_test_data(data_path, label_binarizer)   
    else:
        X_test, Y_test, _ = load_data(data_path, label_binarizer)

    if type(threshold) == list:
        results = []
        for threshold_ in threshold:
            Y_pred = predict(X_test, model_path, approach, threshold_)
            p = precision_score(Y_test, Y_pred, average='micro')
            r = recall_score(Y_test, Y_pred, average='micro')
            f1 = f1_score(Y_test, Y_pred, average='micro')
            results.append((threshold_, f"{p:.2f}", f"{r:.2f}", f"{f1:.2f}"))
        header = ["Threshold", "P", "R", "F1"]
        print(table(results, header, divider=True))
    else:
        Y_pred = predict(X_test, model_path, approach, threshold)
        print(classification_report(Y_test, Y_pred))
