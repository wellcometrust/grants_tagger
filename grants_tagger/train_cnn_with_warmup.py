import pickle

from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.cluster import KMeans
import scipy.sparse as sp
import tensorflow as tf
import numpy as np
import typer

from wellcomeml.ml import CNNClassifier
from grants_tagger.utils import load_data

tf.get_logger().setLevel('ERROR')


def load_pickle(obj_path):
    with open(obj_path, "rb") as f:
        return pickle.loads(f.read())

def train_cnn_with_warmup(data_path, tfidf_vectorizer_path, keras_vectorizer_path, label_binarizer_path, model_path):
    X, Y, _ = load_data(data_path)

    print("Transforming data")
    vectorizer = load_pickle(tfidf_vectorizer_path)
    X_vec = vectorizer.transform(X)

    label_binarizer = load_pickle(label_binarizer_path)
    Y_vec = label_binarizer.transform(Y)

    # Note that X_vec might not be the best representation for clustering
    L_vec = normalize(Y_vec.T.dot(X_vec))
    print(f"L_vec shape: {L_vec.shape}")
    sp.save_npz("data/processed/L_vec.npz", L_vec)

    vectorizer = load_pickle(keras_vectorizer_path)
    X_vec = vectorizer.transform(X)

    # sample
    X_vec = X_vec[:100_000,:]
    Y_vec = Y_vec[:100_000,:]
    L_vec = L_vec[:,:100_000]

    for i, cluster_size in enumerate([16, 256, 4096, Y_vec.shape[1]]):
        print(f"I:{i} Cluster size {cluster_size}")
        
        print("   clustering")
        kmeans = KMeans(n_clusters=cluster_size, max_iter=5, n_init=1)
        K = kmeans.fit_predict(L_vec).reshape(-1, 1)
        print(f"K shape: {K.shape}")

        print("   encoding clusters")
        cluster_encoder = OneHotEncoder(sparse=True)
        K_vec = cluster_encoder.fit_transform(K)

        M = np.dot(Y_vec, K_vec) >= 1
        print(f"M shape: {M.shape}")

        print("   training model")
        l2 = 1e-8
        if i == 0:
            cnn = CNNClassifier(multilabel=True, sparse_y=True, batch_size=256, learning_rate=1e-4, l2=l2, nb_epochs=1, hidden_size=200, dense_size=512)
        else:
            inputs = cnn.model.inputs
            model = tf.keras.Model(inputs, cnn.model.layers[-2].output)
            out = model(inputs)
            out = tf.keras.layers.Dense(cluster_size, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(l2))(out)
            cnn.model = tf.keras.Model(inputs, out)
            print(cnn.model.summary())

        cnn.fit(X_vec, M)



if __name__ == "__main__":
    typer.run(train_cnn_with_warmup)
