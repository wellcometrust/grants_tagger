import tempfile
import pickle
import json
import math
import os

from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
import numpy as np

from grants_tagger.train import train_and_evaluate, create_label_binarizer


# TODO: Use fixtures to reduce duplication

def test_create_label_binarizer():
    texts = ["one", "one two", "two"]
    tags = [["one"], ["one","two"], ["two"]]

    with tempfile.TemporaryDirectory() as tmp_dir:
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")
        data_path = os.path.join(tmp_dir, "data.jsonl")

        with open(data_path, "w") as f:
            for text, tags_ in zip(texts, tags):
                example = {"text": text, "tags": tags_}
                f.write(json.dumps(example)+"\n")

        create_label_binarizer(data_path, label_binarizer_path)

        with open(label_binarizer_path, "rb") as f:
            label_binarizer = pickle.loads(f.read())

        Y = label_binarizer.transform([["one"]])

        assert "one" in label_binarizer.classes_
        assert "two" in label_binarizer.classes_
        assert len(label_binarizer.classes_) == 2
        assert isinstance(Y, np.ndarray)


def test_create_label_binarizer_sparse():
    texts = ["one", "one two", "two"]
    tags = [["one"], ["one","two"], ["two"]]

    with tempfile.TemporaryDirectory() as tmp_dir:
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")
        data_path = os.path.join(tmp_dir, "data.jsonl")

        with open(data_path, "w") as f:
            for text, tags_ in zip(texts, tags):
                example = {"text": text, "tags": tags_}
                f.write(json.dumps(example)+"\n")

        create_label_binarizer(data_path, label_binarizer_path, sparse=True)

        with open(label_binarizer_path, "rb") as f:
            label_binarizer = pickle.loads(f.read())

        Y = label_binarizer.transform([["one"]])

        assert "one" in label_binarizer.classes_
        assert "two" in label_binarizer.classes_
        assert len(label_binarizer.classes_) == 2
        assert isinstance(Y, csr_matrix)


def test_train_and_evaluate():
    approach = "tfidf-svm"

    texts = ["one", "one two", "two"]
    tags = [["one"], ["one", "two"], ["two"]]
    with tempfile.NamedTemporaryFile("r+") as train_data_tmp:
        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit(tags)
        with tempfile.NamedTemporaryFile() as label_binarizer_tmp:
            label_binarizer_tmp.write(pickle.dumps(label_binarizer))

            for text, tags_ in zip(texts, tags):
                train_data_tmp.write(json.dumps({"text": text, "tags": tags_, "meta": {}}))
                train_data_tmp.write("\n")

            train_data_tmp.seek(0)
            label_binarizer_tmp.seek(0)

            train_and_evaluate(train_data_tmp.name, label_binarizer_tmp.name, approach,
                               parameters="{'tfidf__min_df': 1, 'tfidf__stop_words': None}")


def test_train_pickle_save():
    pass

def test_train_model_save():
    approach = "mesh-cnn"

    texts = ["one", "one two", "two"]
    tags = [["one"], ["one","two"], ["two"]]

    with tempfile.TemporaryDirectory() as tmp_dir:
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")
        train_data_path = os.path.join(tmp_dir, "train_data.jsonl")

        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit(tags)
        with open(label_binarizer_path, "wb") as label_binarizer_tmp:
            label_binarizer_tmp.write(pickle.dumps(label_binarizer))

        with open(train_data_path, "w") as train_data_tmp:
            for text, tags_ in zip(texts, tags):
                train_data_tmp.write(json.dumps({"text": text, "tags": tags_, "meta": {}}))
                train_data_tmp.write("\n")

        train_and_evaluate(train_data_tmp.name, label_binarizer_tmp.name,
                           approach, model_path=tmp_dir)

        expected_vectorizer_path = os.path.join(tmp_dir, "vectorizer.pkl")
        expected_model_variables_path = os.path.join(tmp_dir, "variables")
        expected_model_assets_path = os.path.join(tmp_dir, "assets")
        assert os.path.exists(expected_vectorizer_path)
        assert os.path.exists(expected_model_variables_path)
        assert os.path.exists(expected_model_assets_path)


def test_train_and_evaluate_generator():
    approach = "mesh-cnn"

    train_texts = ["one", "one two"]
    train_tags = [["one"], ["one", "two"]]
    
    test_texts = ["two"]
    test_tags = [["two"]]

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_data_path = os.path.join(tmp_dir, "train_data.jsonl")
        test_data_path = os.path.join(tmp_dir, "test_data.jsonl")
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")

        with open(train_data_path, "w") as f:
            for text, tags_ in zip(train_texts, train_tags):
                f.write(json.dumps({"text": text, "tags": tags_, "meta": {}}))
                f.write("\n")

        with open(test_data_path, "w") as f:
            for text, tags_ in zip(test_texts, test_tags):
                f.write(json.dumps({"text": text, "tags": tags_, "meta": {}}))
                f.write("\n")

        train_and_evaluate(train_data_path, label_binarizer_path, approach,
                           data_format="generator", sparse_labels=True, test_data_path=test_data_path)


def test_train_and_evaluate_generator_non_sparse_labels():
    approach = "mesh-cnn"

    train_texts = ["one", "one two"]
    train_tags = [["one"], ["one", "two"]]
    
    test_texts = ["two"]
    test_tags = [["two"]]

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_data_path = os.path.join(tmp_dir, "train_data.jsonl")
        test_data_path = os.path.join(tmp_dir, "test_data.jsonl")
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")

        with open(train_data_path, "w") as f:
            for text, tags_ in zip(train_texts, train_tags):
                f.write(json.dumps({"text": text, "tags": tags_, "meta": {}}))
                f.write("\n")

        with open(test_data_path, "w") as f:
            for text, tags_ in zip(test_texts, test_tags):
                f.write(json.dumps({"text": text, "tags": tags_, "meta": {}}))
                f.write("\n")
        
        train_and_evaluate(train_data_path, label_binarizer_path, approach,
                           data_format="generator", test_data_path=test_data_path)


def test_train_and_evaluate_threshold():
    pass


# TODO: Move to models
def test_train_and_evaluate_y_batch_size():
    approach = "mesh-tfidf-svm"

    texts = ["one", "one two", "all"]
    tags = [["1"], ["1", "2"], [str(i) for i in range(5000)]]

    with tempfile.TemporaryDirectory() as tmp_dir:
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")
        train_data_path = os.path.join(tmp_dir, "train_data.jsonl")
        model_path = os.path.join(tmp_dir, "model")

        with open(train_data_path, "w") as train_data_tmp:
            for text, tags_ in zip(texts, tags):
                train_data_tmp.write(json.dumps({"text": text, "tags": tags_, "meta": {}}))
                train_data_tmp.write("\n")
        
        parameters = {
            'vec__min_df': 1,
            'vec__stop_words': None,
            'y_batch_size': 512,
            'model_path': model_path
        }
        train_and_evaluate(
            train_data_path, label_binarizer_path, approach,
            parameters=str(parameters),
            model_path=model_path, sparse_labels=True)

        model_artifacts = os.listdir(model_path)
        assert len(model_artifacts) == math.ceil(5000 / 512) + 2 # (vectorizer, meta)

