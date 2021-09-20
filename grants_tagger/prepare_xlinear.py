import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import scipy as sp
import numpy as np
import typer

from grants_tagger.utils import load_data


def write_txt(txt_path, data):
    with open(txt_path, "w") as f:
        for item in data:
            f.write(item.replace("\n", " "))
            f.write("\n")

def load_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.loads(f.read())

def write_pickle(pickle_path, obj):
    with open(pickle_path, "wb") as f:
        f.write(pickle.dumps(obj))

def prepare_xlinear(data_path, x_path, y_path, tfidf_vectorizer_path, label_binarizer_path, x_txt_path=None):
    print("Loading data")
    X, Y, _ = load_data(data_path)

    print("Creating tfidf features")
    if os.path.exists(tfidf_vectorizer_path):
        print("   loading existing")
        tfidf = load_pickle(tfidf_vectorizer_path)
    else:
        print("   training vectorizer")
        tfidf = TfidfVectorizer(max_features=400_000, min_df=10)
        tfidf.fit(X)
        write_pickle(tfidf_vectorizer_path, tfidf)
    X_vec = tfidf.transform(X).astype("float32")

    print("Binarizing labels")
    if os.path.exists(label_binarizer_path):
        print("   loading existing")
        label_binarizer = load_pickle(label_binarizer_path)
    else:
        print("   training binarizer")
        label_binarizer = MultiLabelBinarizer(sparse_output=True)
        label_binarizer.fit(Y)
        write_pickle(label_binarizer_path, label_binarizer)
    Y_vec = label_binarizer.transform(Y)

    print("Saving")
    sp.sparse.save_npz(x_path, X_vec)
    sp.sparse.save_npz(y_path, Y_vec)
    if x_txt_path:
        write_txt(x_txt_path, X)

if __name__ == "__main__":
    typer.run(prepare_xlinear)


