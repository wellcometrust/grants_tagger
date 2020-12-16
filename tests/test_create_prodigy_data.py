import tempfile
import os.path
import pickle
import json

from sklearn.preprocessing import MultiLabelBinarizer

from grants_tagger.create_prodigy_data import create_prodigy_data


DATA = [
    {"text": "One", "tags": ["T1", "T2"], "meta": {"Grant_ID": 1}}
]


def test_create_prodigy_data_teach():
    mode = "teach"

    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = os.path.join(tmp_dir, "data.jsonl")
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")
        output_path = os.path.join(tmp_dir, "out.jsonl")

        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit([example["tags"] for example in DATA])
        with open(label_binarizer_path, "wb") as f:
            f.write(pickle.dumps(label_binarizer))

        with open(data_path, "w") as f:
            for line in DATA:
                f.write(json.dumps(line)+"\n")

        create_prodigy_data(data_path, label_binarizer_path, output_path, mode)

        out_data = []
        with open(output_path) as f:
            for line in f:
                out_data.append(json.loads(line))

        assert len(out_data) == 1
        assert out_data[0]["text"] == "One"
        assert out_data[0]["meta"]["grant_id"] == 1
        assert "tags" not in out_data[0]


def test_create_prodigy_data_train():
    mode = "train"

    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = os.path.join(tmp_dir, "data.jsonl")
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")
        output_path = os.path.join(tmp_dir, "out.jsonl")

        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit([example["tags"] for example in DATA])
        with open(label_binarizer_path, "wb") as f:
            f.write(pickle.dumps(label_binarizer))

        with open(data_path, "w") as f:
            for line in DATA:
                f.write(json.dumps(line)+"\n")

        create_prodigy_data(data_path, label_binarizer_path, output_path, mode)

        out_data = []
        with open(output_path) as f:
            for line in f:
                out_data.append(json.loads(line))

        assert len(out_data) == 2
        assert out_data[0]["text"] == "One"
        assert out_data[0]["answer"] == "accept"
        assert out_data[1]["text"] == "One"
        assert out_data[1]["answer"] == "accept"
        assert "T1" in [out_data[0]["label"], out_data[1]["label"]]
        assert "T2" in [out_data[0]["label"], out_data[1]["label"]]
