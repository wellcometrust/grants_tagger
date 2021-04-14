import tempfile
import json
import os

from grants_tagger.preprocess_mesh import process_data, preprocess_mesh


def test_preprocess_mesh():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"]
    }
    with tempfile.NamedTemporaryFile(mode="r+") as input_tmp:
        input_tmp.write("\n"+json.dumps(item)+", ")
        input_tmp.seek(0)
        output_tmp = tempfile.NamedTemporaryFile(mode="r+")
        preprocess_mesh(input_tmp.name, output_tmp.name)
    output_tmp.seek(0)
    expected_processed_item = {
        "text": "This is an abstract",
        "tags": ["T1", "T2"],
        "meta": {}
    }
    assert output_tmp.read() == json.dumps(expected_processed_item) + "\n"

def test_preprocess_mesh_test_split():
    items = [
        {
            "abstractText": "This is an abstract",
            "meshMajor": [f"T{i}", "T20"]
        }
        for i in range(10)
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, "data.json")
        output_path = os.path.join(tmp_dir, "processed_data.jsonl")
        
        with open(input_path, "w") as f:
            f.write("\n")
            for item in items:
                f.write(json.dumps(item)+",")
                f.write("\n")

        preprocess_mesh(input_path, output_path, test_split=0.1)
        
        train_output_path = os.path.join(tmp_dir, "train_processed_data.jsonl")
        test_output_path = os.path.join(tmp_dir, "test_processed_data.jsonl")
        assert os.path.exists(train_output_path)
        assert os.path.exists(test_output_path)

        examples = 0
        with open(train_output_path) as f:
            for line in f:
                examples += 1
        assert examples == 9

        examples = 0
        with open(test_output_path) as f:
            for line in f:
                examples += 1
        assert examples == 1   

def test_process_data():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"]
    }
    expected_processed_item = {
        "text": "This is an abstract",
        "tags": ["T1", "T2"],
        "meta": {}
    }
    processed_item = process_data(item)
    assert processed_item == expected_processed_item

def test_process_data_with_filter_tags():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"]
    }
    expected_processed_item = {
        "text": "This is an abstract",
        "tags": ["T1"],
        "meta": {}
    }
    processed_item = process_data(item, filter_tags=["T1"])
    assert processed_item == expected_processed_item

def test_process_data_with_missing_filter_tag():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"]
    }
    processed_item = process_data(item, filter_tags=["T3"])
    assert processed_item == None
