# encoding: utf-8
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

def load_data(data_path, label_binarizer=None):
    """Load data from the dataset."""
    print("Loading data...")

    texts = []
    tags = []
    meta = []
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)

            texts.append(data["text"])
            tags.append(data["tags"])
            meta.append(data["meta"])

    if label_binarizer:
        tags = label_binarizer.transform(tags)

    return texts, tags, meta

def load_train_test_data(
        train_data_path, label_binarizer,
        test_data_path=None, from_same_distribution=False,
        test_size=None):

    if test_data_path:
        X_train, Y_train, _ = load_data(train_data_path, label_binarizer)
        X_test, Y_test, _ = load_data(test_data_path, label_binarizer)

        if from_same_distribution:
            X_train, _, Y_train, _ = train_test_split(
                X_train, Y_train, random_state=42
            )
            _, X_test, _, Y_test = train_test_split(
                X_test, Y_test, random_state=42
            )
    else:
        X, Y, _ = load_data(train_data_path, label_binarizer)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, random_state=42, test_size=test_size
        )
           
    return X_train, X_test, Y_train, Y_test

def yield_train_data(data_path, label_binarizer, batch_size=100):
    with open(data_path) as f:
        i = 0
        X = []
        Y = []
        for line in f:
            i += 1
            item = json.loads(line)
            X.append(item['text'])
            tags = item['tags']
            Y.append(label_binarizer.transform([tags])[0])
            if i % batch_size == 0:
                Y = np.array(Y)
                yield X, Y
                X = []
                Y = []
        if X:
            yield X, Y

def load_test_data(data_path, label_binarizer):
    X = []
    Y = []
    for x, y in yield_train_data(data_path, label_binarizer):
        X.extend(x)
        Y.extend(y)
    return X, np.array(Y)

# TODO: Move to common for cases where Y is a matrix
def calc_performance_per_tag(Y_true, Y_pred, tags):
    metrics = []
    for tag_index in range(Y_true.shape[1]):
        y_true_tag = Y_true[:,tag_index]
        y_pred_tag = Y_pred[:,tag_index]
        metrics.append({
            'Tag': tags[tag_index],
            'f1': f1_score(y_true_tag, y_pred_tag)
        })
    return pd.DataFrame(metrics)
