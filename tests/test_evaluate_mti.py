import os.path
import tempfile
import pickle
import json

from sklearn.preprocessing import MultiLabelBinarizer

from grants_tagger.evaluate_mti import evaluate_mti


MESH_DATA = [
    {"text": "One Two", "tags": ["T1", "T2"]},
    {"text": "Three", "tags": ["T3"]},
    {"text": "Two", "tags": ["T2"]}
]
MTI_DATA = [
    ("0", "T1", "", ""),
    ("0", "T2", "", ""),
    ("1", "T1", "", ""),
    ("2", "T2", "", ""),
    ("2", "T3", "", "")
]


def test_evaluate_mti():
    with tempfile.TemporaryDirectory() as tmp_dir:
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")
        mesh_data_path = os.path.join(tmp_dir, "mesh_data.jsonl")
        mti_output_path = os.path.join(tmp_dir, "mti.csv")

        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit([example["tags"]  for example in MESH_DATA])
        with open(label_binarizer_path, "wb") as f:
            f.write(pickle.dumps(label_binarizer))

        with open(mesh_data_path, "w") as f:
            for line in MESH_DATA:
                f.write(json.dumps(line)+"\n")

        with open(mti_output_path, "w") as f:
            for line in MTI_DATA:
                f.write("|".join(line)+"\n")

        f1 = evaluate_mti(label_binarizer_path, mesh_data_path, mti_output_path)
        assert 0 < f1 < 1
