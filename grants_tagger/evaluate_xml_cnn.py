import pickle
import math
import os

from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf
import typer

from grants_tagger.train_xml_cnn import load_data


def load_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        obj = pickle.loads(f.read())
    return obj

def precision_at_k(Y_test, Y_pred_proba, max_k):
    print("- sorting probs")
    ind = np.argsort(Y_pred_proba, axis=1)[:,-max_k:]
    print("- getting Y_pred")
    Y_pred = sp.csr_matrix(Y_pred_proba > 0.5)

    p = []
    for k in range(1, max_k+1):
        print(f"- k: {k}")
        k_ind = ind[:,-k:]    
        Y_test_k = np.take_along_axis(Y_test, k_ind, axis=1)
        Y_pred_k = np.take_along_axis(Y_pred, k_ind, axis=1)

        pk = np.sum(Y_test_k.multiply(Y_pred_k), axis=1) / k
        p.append(np.mean(pk))
    return p

def evaluate_xml_cnn(test_data_path, model_path):
    label_binarizer = load_pickle(os.path.join(model_path, "label_binarizer.pkl"))
    tokenizer = load_pickle(os.path.join(model_path, "tokenizer.pkl"))
    model = tf.keras.models.load_model(os.path.join(model_path, "model"))

    print("Loading data")
    X, Y = load_data(test_data_path)
    
    print("Vectorising data")
    X_vec = tokenizer.texts_to_sequences(X)
    X_vec = tf.keras.preprocessing.sequence.pad_sequences(X_vec, maxlen=500)
    Y_vec = label_binarizer.transform(Y)

    print("Predicting")
    batch_x = None
    if batch_x:
        Y_pred = []
        for i in tqdm(range(0, X_vec.shape[0], batch_x)):
            Y_pred_batch = model.predict(X_vec[i:i+batch_x,:]) > 0.5
            Y_pred_batch = sp.csr_matrix(Y_pred_batch)
            Y_pred.append(Y_pred_batch)
        Y_pred = sp.vstack(Y_pred)
    else:
        Y_pred_proba = model.predict(X_vec)
        Y_pred = sp.csr_matrix(Y_pred_proba > 0.5)

    print("Calculating metrics")
    p, r, f1, _ = precision_recall_fscore_support(Y_vec, Y_pred, average="micro")
    print(f"P: {p} R: {r} f1: {f1}")

    print("Using xclib")
    import xclib.evaluation.xc_metrics as xc_metrics
    pk = xc_metrics.precision(Y_pred_proba, Y_vec)
    print(" ".join([f"P@{k}: {pk[k-1]}" for k in [1, 3, 5]]))

    print("Using custom")
    pk = precision_at_k(Y_vec, Y_pred_proba, 5)
    print(" ".join([f"P@{k}: {pk[k-1]}" for k in [1, 3, 5]]))

if __name__ == "__main__":
    typer.run(evaluate_xml_cnn)
