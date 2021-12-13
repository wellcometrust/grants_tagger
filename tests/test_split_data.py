import tempfile
import json
import os

from grants_tagger.split_data import split_data

def test_split_data():
    items = [
        {
            "text": "This is an abstract",
            "tags": ["T1", "T2"],
            "meta": {}
        }
        for i in range(10)
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, "processed_data.jsonl")
        train_output_path = os.path.join(tmp_dir, "train_data.jsonl")
        test_output_path = os.path.join(tmp_dir, "test_data.jsonl")
        
        with open(input_path, "w") as f:
            for item in items:
                f.write(json.dumps(item))
                f.write("\n")

        split_data(input_path, train_output_path, test_output_path, 0.1)
        
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