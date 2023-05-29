import os
import json
import tempfile
import pytest
from grants_tagger.evaluate_pubs import evaluate_pubs


@pytest.fixture
def test_data():
    return [
        {
            "abstract": "This is a test abstract about Malaria",
            "mesh_terms": ["Malaria"],
        },
        {
            "abstract": "This is a test abstract about Cholera",
            "mesh_terms": ["Cholera"],
        },
        {
            "abstract": "This is a test abstract about Malaria and Cholera",
            "mesh_terms": ["Malaria", "Cholera"],
        },
        {
            "abstract": "This is a test abstract without any mesh terms",
            "mesh_terms": [],
        },
    ]


def test_evaluate_pubs(test_data):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        json.dump(test_data, f)
        data_path = f.name

    try:
        evaluate_pubs(data_path, threshold=0.5, batch_size=2, device="cpu")
    finally:
        os.unlink(data_path)
