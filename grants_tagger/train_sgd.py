import pickle
import time
import sys

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
import scipy.sparse as sp
import numpy as np
import typer


def train_sgd(x_path, y_path, model_path=None):
    X = sp.load_npz(x_path)
    Y = sp.load_npz(y_path)

    print(f"X size {X.data.nbytes/1_000_000_000:.2f}GB")
    print(f"Y size {Y.data.nbytes/1_000_000_000:.2f}GB")
    model = OneVsRestClassifier(SGDClassifier(loss="log"), n_jobs=4)
    start = time.time()
    model.fit(X, Y[:,0:4])
    train_time = time.time() - start

    print(f"Model took {train_time:.2f}s to train")
    obj = pickle.dumps(model)
    model_size = sys.getsizeof(obj)
    print(f"Model is {model_size / 1_000_000:.2f}MB before sparsifying")
        
    for i, est in enumerate(model.estimators_):
        est.coef_[np.abs(est.coef_) < 0.001] = 0
        est.sparsify()
        model.estimators_[i] = est
    obj = pickle.dumps(model)
    model_size = sys.getsizeof(obj)
    print(f"Model is {model_size / 1_000_000:.2f}MB after sparsifying")

    if model_path:
        with open(model_path, "wb") as f:
            f.write(obj)


if __name__ == "__main__":
    typer.run(train_sgd)
