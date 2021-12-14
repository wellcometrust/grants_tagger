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


def prepare_xlinear(
    data_path,
    x_path,
    y_path,
    tfidf_vectorizer_path,
    label_binarizer_path,
    dense_vectorizer_path=None,
    x_txt_path=None,
    sequence_length=None,
    sample_size=None,
    years=None,
):
    print("Loading data")
    X, Y, meta = load_data(data_path)

    if sample_size or years:
        print("Sampling")
        if sample_size:
            sample_size = int(sample_size)
            X = X[:sample_size]
            Y = Y[:sample_size]
        else:
            min_year, max_year = [int(year) for year in years.split(",")]
            print("   calculating sample size")
            sample_indices = [
                i
                for i, m in enumerate(meta)
                if m["year"] and (min_year <= int(m["year"]) <= max_year)
            ]
            print(f"   sample_size {len(sample_indices)}")
            X = [X[i] for i in sample_indices]
            Y = [Y[i] for i in sample_indices]

    print("Creating tfidf features")
    if os.path.exists(tfidf_vectorizer_path):
        print("   loading existing tfidf")
        tfidf = load_pickle(tfidf_vectorizer_path)
    else:
        print("   training tfidf")
        tfidf = TfidfVectorizer(max_features=400_000, min_df=5, stop_words="english")
        tfidf.fit(X)
        write_pickle(tfidf_vectorizer_path, tfidf)

    print("   transforming X")
    X_vec = tfidf.transform(X).astype("float32")
    print(f"   number of features {X_vec.shape[1]}")

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
    print(f"   number of labels {Y_vec.shape[1]}")

    print("Saving")
    sp.sparse.save_npz(x_path, X_vec)
    sp.sparse.save_npz(y_path, Y_vec)
    if x_txt_path:
        write_txt(x_txt_path, X)


if __name__ == "__main__":
    typer.run(prepare_xlinear)
