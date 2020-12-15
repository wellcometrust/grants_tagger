import tempfile
import os.path
import pickle
import json

from sklearn.preprocessing import MultiLabelBinarizer

from grants_tagger.evaluate_human import evaluate_human


def test_evaluate_human():
    data = [
        {
            "text": "Lorem",
            "tags": ["T1", "T2"],
            "meta": {
                "Tagger1_tags": ["T1"]
            }
        },
        {
            "text": "Ipsum",
            "tags": ["T2"],
            "meta": {
                "Tagger1_tags": ["T1"]
            }
        }
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = os.path.join(tmp_dir, "data.jsonl")
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer_path")

        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit([line["tags"] for line in data])

        with open(data_path, "w") as f:
            for line in data:
                f.write(json.dumps(line)+"\n")
        with open(label_binarizer_path, "wb") as f:
            f.write(pickle.dumps(label_binarizer))
        f1 = evaluate_human(data_path, label_binarizer_path)
        assert 0 < f1 < 1


def test_evaluate_human_zero_f1():
    data = [
        {
            "text": "Lorem",
            "tags": ["T2"],
            "meta": {
                "Tagger1_tags": ["T1"]
            }
        },
        {
            "text": "Ipsum",
            "tags": ["T2"],
            "meta": {
                "Tagger1_tags": ["T1"]
            }
        }
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = os.path.join(tmp_dir, "data.jsonl")
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer_path")

        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit([line["tags"] for line in data])

        with open(data_path, "w") as f:
            for line in data:
                f.write(json.dumps(line)+"\n")
        with open(label_binarizer_path, "wb") as f:
            f.write(pickle.dumps(label_binarizer))
        f1 = evaluate_human(data_path, label_binarizer_path)
        assert f1 == 0


def test_evaluate_human_perfect_f1():
    data = [
        {
            "text": "Lorem",
            "tags": ["T1", "T2"],
            "meta": {
                "Tagger1_tags": ["T1", "T2"]
            }
        },
        {
            "text": "Ipsum",
            "tags": ["T2"],
            "meta": {
                "Tagger1_tags": ["T2"]
            }
        }
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = os.path.join(tmp_dir, "data.jsonl")
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer_path")

        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit([line["tags"] for line in data])

        with open(data_path, "w") as f:
            for line in data:
                f.write(json.dumps(line)+"\n")
        with open(label_binarizer_path, "wb") as f:
            f.write(pickle.dumps(label_binarizer))
        f1 = evaluate_human(data_path, label_binarizer_path)
        assert f1 == 1
