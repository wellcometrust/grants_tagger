import pickle
import os

from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizerFast
import scipy.sparse as sp
import numpy as np
import typer

from grants_tagger.utils import load_data


def load_pickle(obj_path):
    with open(obj_path, "rb") as f:
        return pickle.loads(f.read())


def save_pickle(obj_path, obj):
    with open(obj_path, "wb") as f:
        f.write(pickle.dumps(obj))


def prepare_bertmesh(
    data_path,
    x_path,
    y_path,
    label_binarizer_path,
    pretrained_model="bert-base-uncased",
    tokenizer_path=None,
    years=None,
):
    X, Y, meta = load_data(data_path)

    if years:
        print("filtering data")
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

    if os.path.exists(label_binarizer_path):
        print("loading existing")
        label_binarizer = load_pickle(label_binarizer_path)
    else:
        print("training label binarizer")
        label_binarizer = MultiLabelBinarizer(sparse_output=True)
        label_binarizer.fit(Y)

    print("transforming data")
    Y_vec = label_binarizer.transform(Y)
    print(f"   number of labels {Y_vec.shape[1]}")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    X_vec = tokenizer(X, truncation=True, padding=True)["input_ids"]
    X_vec = np.array(X_vec)
    print(f"   number of features {X_vec.shape[1]}")

    print("saving")
    np.save(x_path, X_vec)
    sp.save_npz(y_path, Y_vec)
    save_pickle(label_binarizer_path, label_binarizer)
    if tokenizer_path:
        save_pickle(tokenizer_path, tokenizer)


if __name__ == "__main__":
    typer.run(prepare_bertmesh)
