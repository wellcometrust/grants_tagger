import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, minmax_scale
import scipy as sp
import numpy as np
import typer

from wellcomeml.ml import Sent2VecVectorizer
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

class DenseVectorizer():
    def __init__(self, pretrained_vectors="glove-wiki-gigaword-300"):
        self.embeddings_index = api.load(pretrained_vectors)
        self.embedding_size = len(self.embeddings_index.values()[0])

    def _vectorize(self, x):
        word_embeddings = np.array([embeddings_index[w] for w in x.split() if w in embeddings_index])
        if words_embeddings.size:
            return np.mean(word_embeddings, axis=0)
    
    def transform(self, X):
        X_dense = np.random.uniform(-0.25, 0.25, (X.shape[0], self.embedding_size))
        for i, x in enumerate(X):
            x_den = self.vectorize(x)
            if x_den:
                X_dense[i, :] = self.vectorize(x)
        return X_dense

def prepare_xlinear(data_path, x_path, y_path, tfidf_vectorizer_path, label_binarizer_path, dense_vectorizer_path=None, x_txt_path=None):
    print("Loading data")
    X, Y, _ = load_data(data_path)

    print("Creating tfidf features")
    if os.path.exists(tfidf_vectorizer_path):
        print("   loading existing tfidf")
        tfidf = load_pickle(tfidf_vectorizer_path)
    else:
        print("   training tfidf")
        tfidf = TfidfVectorizer(max_features=400_000, min_df=10)
        tfidf.fit(X)
        write_pickle(tfidf_vectorizer_path, tfidf)

    if dense_vectorizer_path:
        if dense_vectorizer_path == "sent2vec":
            # sent2vec is not picklable
            dense = Sent2VecVectorizer()
            pass
        elif os.path.exists(dense_vectorizer_path):
            print("   loading existing dense")
            dense = load_pickle(dense_vectorizer_path)
        else:
            print("   initializing new dense")
            dense = DenseVectorizer()

        print("   transforming X")
        X_vec_dense = minmax_scale(dense.transform(X).astype("float32"))
        X_vec_tfidf = minmax_scale(tfidf.transform(X).astype("float32"))
        X_vec = sp.sparse.vstack([X_vec_dense, X_vec_tfidf]).tocsr() 
    else:
        print("   transforming X")
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


