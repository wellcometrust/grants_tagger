# encoding: utf-8
import pickle
import json

import requests
from sklearn.metrics import f1_score
import tensorflow as tf
import pandas as pd
import numpy as np


# TODO: Move to common for cases where Y is a matrix
def calc_performance_per_tag(Y_true, Y_pred, tags):
    metrics = []
    for tag_index in range(Y_true.shape[1]):
        y_true_tag = Y_true[:, tag_index]
        y_pred_tag = Y_pred[:, tag_index]
        metrics.append({"Tag": tags[tag_index], "f1": f1_score(y_true_tag, y_pred_tag)})
    return pd.DataFrame(metrics)


def get_ec2_instance_type():
    """Utility function to get ec2 instance name, or empty string if not possible to get name"""

    try:
        instance_type_request = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-type", timeout=5
        )
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        return ""

    if instance_type_request.status_code == 200:
        return instance_type_request.content.decode()
    else:
        return ""


def load_pickle(obj_path):
    with open(obj_path, "rb") as f:
        return pickle.loads(f.read())


def save_pickle(obj_path, obj):
    with open(obj_path, "wb") as f:
        f.write(pickle.dumps(obj))


def write_jsonl(f, data):
    for item in data:
        f.write(json.dumps(item))
        f.write("\n")
