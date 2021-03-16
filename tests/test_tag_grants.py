from unittest.mock import patch
import tempfile
import shutil
import pickle
import csv
import os

import pytest

from grants_tagger.tag_grants import tag_grants


@pytest.fixture
def grants_path(tmp_path):
    grants_path = os.path.join(tmp_path, "grants.csv")
    with open(grants_path, "w") as tmp_grants:
        csvwriter = csv.DictWriter(tmp_grants, fieldnames=["title", "synopsis", "grant_id", "grant_no", "reference"])
        csvwriter.writeheader()

        X = [str(i) for i in range(5)]
        for i, x in enumerate(X):
            csvwriter.writerow({
                "title": "",
                "synopsis": x,
                "grant_id": i,
                "reference": i,
                "grant_no": i
            })
    return grants_path

@pytest.fixture
def tagged_grants_path(tmp_path):
    return os.path.join(tmp_path, "tagged_grants.csv")

def test_tag_grants(grants_path, tagged_grants_path):
    with patch('grants_tagger.tag_grants.predict_tags') as mock_predict:
        mock_predict.return_value = [[f"tag #{i}"] for i in range(5)]
        tag_grants(
            grants_path,
            tagged_grants_path,
            model_path="tfidf_svm_path",
            label_binarizer_path="label_binarizer_path",
            approach="science-ensemble"
        )
        tagged_grants = []
        with open(tagged_grants_path) as f:
            for line in f:
                tagged_grants.append(line)
        assert len(tagged_grants) == 6 # 5 + header
