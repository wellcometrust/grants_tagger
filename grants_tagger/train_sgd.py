import pickle
import sys

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
import scipy.sparse as sp
import typer


def train_sgd(x_path, y_path, model_path=None):
    X = sp.load_npz(x_path)
    Y = sp.load_npz(y_path)

    model = SGDClassifier(loss="log")
    model.fit(X, Y[:, 0])

    model_size = sys.getsizeof(pickle.dumps(model))
    print(model_size)


if __name__ == "__main__":
    typer.run(train_sgd)
