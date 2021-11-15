import tempfile
import os.path
import pickle
import json
import csv

from sklearn.preprocessing import MultiLabelBinarizer
import pytest

from grants_tagger.evaluate_scispacy_meshtagger import evaluate_scispacy_meshtagger
try:
    import scispacy
    SCISPACY_INSTALLED = True
except ModuleNotFoundError:
    SCISPACY_INSTALLED = False

MESH_TAGS = [
    {
        "DescriptorUI": "D008288",
        "DescriptorName": "Malaria"
    },
    {
        "DescriptorUI": "D006678",
        "DescriptorName": "HIV"
    }
]
DATA = [
    {
        "text": "Malaria is a mosquito-borne infectious disease that affects humans and other animals. Malaria causes symptoms that typically include fever, tiredness, vomiting, and headaches. In severe cases it can cause yellow skin, seizures, coma, or death",
        "tags": ["Malaria"],
        "meta": {}
    },
    {
        "text": "The human immunodeficiency viruses are two species of Lentivirus that infect humans. Without treatment, average survival time after infection with HIV is estimated to be 9 to 11 years, depending on the HIV subtype.",
        "tags": ["HIV"],
        "meta": {}
    },
]


@pytest.mark.skipif(not SCISPACY_INSTALLED, reason="scispacy missing")
def test_evaluate_scispacy_meshtagger():
    with tempfile.TemporaryDirectory() as tmp_dir:
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")
        mesh_tags_path = os.path.join(tmp_dir, "mesh_tags.csv")
        data_path = os.path.join(tmp_dir, "data.jsonl")

        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit([example["tags"] for example in DATA])
        with open(label_binarizer_path, "wb") as f:
            f.write(pickle.dumps(label_binarizer))

        with open(mesh_tags_path, "w") as f:
            csvwriter = csv.DictWriter(f, fieldnames=["DescriptorUI", "DescriptorName"])
            csvwriter.writeheader()
            for line in MESH_TAGS:
                csvwriter.writerow(line)

        with open(data_path, "w") as f:
            for line in DATA:
                f.write(json.dumps(line)+"\n")

        evaluate_scispacy_meshtagger(label_binarizer_path, mesh_tags_path, data_path)
