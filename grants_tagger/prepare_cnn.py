import pickle
import json

from sklearn.preprocessing import MultiLabelBinarizer
import scipy.sparse as sp
import numpy as np
import typer

from wellcomeml.ml import KerasVectorizer, CNNClassifier


def write_pickle(pickle_path, obj):
    with open(pickle_path, "wb") as f:
        f.write(pickle.dumps(obj))

def write_txt(data_path, data):
    with open(data_path, "w") as f:
        for line in data:
            f.write(line)
            f.write("\n")

def load_data(data_path):
    with open(data_path) as f:
        texts = []
        tags = []
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            tags.append(item["tags"])
    return texts, tags

def prepare_cnn(data_path, vectorizer_path, label_binarizer_path):
    print("Loading data")
    X, Y = load_data(data_path)

    print("Training vectorizer")
    vectorizer = KerasVectorizer(tokenizer_library="transformers", vocab_size=30_000, sequence_length=200)
    vectorizer.fit(X)
    write_pickle(vectorizer_path, vectorizer)

    print("Training label binarizer")
    label_binarizer = MultiLabelBinarizer(sparse_output=True)
    label_binarizer.fit(Y)
    write_pickle(label_binarizer_path, label_binarizer)

if __name__ == "__main__":
    typer.run(prepare_cnn)
