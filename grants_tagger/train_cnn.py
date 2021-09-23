import pickle
import time
import os

import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import typer

from grants_tagger.utils import load_data
from wellcomeml.ml import CNNClassifier


def load_pickle(obj_path):
    with open(obj_path, "rb") as f:
        obj = pickle.loads(f.read())
    return obj

def train_cnn(data_path, vectorizer_path, label_binarizer_path, model_path):
    # TODO: Try keeping only X and Y for rows that some of Ys are 1
    X, Y, _ = load_data(data_path)

    vectorizer = load_pickle(vectorizer_path)
    label_binarizer = load_pickle(label_binarizer_path)

    def yield_data():
        buffer_size=512
        for i in range(0, len(X), buffer_size):
            X_vec = vectorizer.transform(X[i:i+buffer_size])
            Y_vec = label_binarizer.transform(Y[i:i+buffer_size])
            Y_vec = Y_vec[:,:300].todense()        
            for j in range(X_vec.shape[0]):
                x = X_vec[j,:]
                y = Y_vec[j,:]
                y = np.squeeze(np.asarray(y))
                yield x, y

    data = tf.data.Dataset.from_generator(yield_data, output_types=(tf.int32, tf.int32))
    data = data.shuffle(1000)
    
    start = time.time()
    model = CNNClassifier(multilabel=True, sparse_y=True, batch_size=256, learning_rate=1e-4, l2=1e-8)
    model.fit(data)
    train_time = time.time() - start
    print(f"It took {train_time:.2f}s to train the model")

    model.save(model_path)
    # Note that this does not work
    model_size = sum(os.path.getsize(f) for f in os.listdir('.') if os.path.isfile(f)) / 1_000_000
    print(f"Model size is {model_size:.2f}MB")

    with open(model_path, "wb") as f:
        f.write(obj)

if __name__ == "__main__":
    typer.run(train_cnn)
