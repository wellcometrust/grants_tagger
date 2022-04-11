import tempfile
import json
import io
import os

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import pytest

from grants_tagger.utils import (
    load_data,
    calc_performance_per_tag,
    load_train_test_data,
    load_pickle,
    save_pickle,
)


@pytest.fixture
def train_data_path(tmp_path):
    test_data = [
        {"text": "A", "tags": ["T1", "T2"], "meta": {"Grant_ID": 1, "Title": "A"}},
        {"text": "B", "tags": ["T1"], "meta": {"Grant_ID": 2, "Title": "B"}},
    ]
    test_data_path = os.path.join(tmp_path, "train_data.jsonl")
    with open(test_data_path, "w") as f:
        for line in test_data:
            f.write(json.dumps(line))
            f.write("\n")

    return test_data_path


@pytest.fixture
def test_data_path(tmp_path):
    test_data = [{"text": "C", "tags": ["T2"], "meta": {"Grant_ID": 3, "Title": "C"}}]
    test_data_path = os.path.join(tmp_path, "test_data.jsonl")
    with open(test_data_path, "w") as f:
        for line in test_data:
            f.write(json.dumps(line))
            f.write("\n")

    return test_data_path


@pytest.fixture
def data_path(tmp_path, train_data_path, test_data_path):
    data_path = os.path.join(tmp_path, "data.jsonl")
    with open(data_path, "w") as data_f:
        with open(train_data_path, "r") as train_f:
            for line in train_f:
                data_f.write(line)
        with open(test_data_path, "r") as test_f:
            for line in test_f:
                data_f.write(line)
    return data_path


@pytest.fixture
def label_binarizer():
    label_binarizer = MultiLabelBinarizer(classes=["T1", "T2"])
    label_binarizer.fit([["T1", "T2"]])
    return label_binarizer


def test_load_data(data_path):
    texts, tags, meta = load_data(data_path)
    assert len(texts) == 3
    assert len(tags) == 3
    assert len(meta) == 3
    assert texts[0] == "A"
    assert tags[0] == ["T1", "T2"]
    assert meta[0] == {"Grant_ID": 1, "Title": "A"}


def test_load_data_with_label_binarizer(data_path, label_binarizer):
    texts, tags, meta = load_data(data_path, label_binarizer)
    assert np.array_equal(tags[0], [1, 1])
    assert np.array_equal(tags[1], [1, 0])
    assert np.array_equal(tags[2], [0, 1])


def test_calc_performance_per_tag():
    Y_true = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    Y_pred = np.array([[0, 0, 1], [1, 0, 1], [0, 0, 1]])
    tags = ["T1", "T2", "T3"]

    performance_per_tag_test = pd.DataFrame(
        {
            "Tag": tags,
            "f1": [
                f1_score(Y_true[:, i], Y_pred[:, i]) for i in range(Y_true.shape[1])
            ],
        }
    )
    performance_per_tag = calc_performance_per_tag(Y_true, Y_pred, tags)
    print(performance_per_tag)
    assert performance_per_tag.equals(performance_per_tag_test)


def test_load_train_test_data(data_path, label_binarizer):
    texts_train, texts_test, tags_train, tags_test = load_train_test_data(
        data_path, label_binarizer
    )
    assert np.array_equal(tags_train[0], [1, 0])
    assert np.array_equal(tags_train[1], [0, 1])
    assert np.array_equal(tags_test[0], [1, 1])


def test_load_train_test_data_generator(
    train_data_path, test_data_path, label_binarizer
):
    X_train, X_test, Y_train, Y_test = load_train_test_data(
        train_data_path,
        label_binarizer,
        test_data_path=test_data_path,
        data_format="generator",
    )

    tags_train = list(Y_train())
    tags_test = list(Y_test())
    assert np.array_equal(tags_train[0], [1, 1])
    assert np.array_equal(tags_train[1], [1, 0])
    assert np.array_equal(tags_test[0], [0, 1])


def test_save_load_pickle(tmp_path):
    obj = {"data": [1, 2]}
    obj_path = os.path.join(tmp_path, "obj.pkl")
    save_pickle(obj_path, obj)
    assert os.path.exists(obj_path)
    obj_loaded = load_pickle(obj_path)
    assert obj_loaded == obj
