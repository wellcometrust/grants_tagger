import math

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from scipy.sparse import vstack, csr_matrix
import tensorflow as tf
import numpy as np


class CustomBiLSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_size=None, attention=None, multilabel=None, l2=None,
                 dropout=None, learning_rate=None, sparse_y=None, batch_size=None,
                 dense_size=None, nb_epochs=None):
        pass

    def _yield_data(self, X, Y, batch_size, shuffle=False):
        while True:
            if shuffle:
                randomize = np.arange(len(X))
                np.random.shuffle(randomize)
                X = X[randomize]
                Y = Y[randomize]
            for i in range(0, X.shape[0], batch_size):
                yield X[i:i+batch_size], Y[i:i+batch_size,:].todense()

    def _build_model(self, vocabulary_size, sequence_length, nb_tags):
                
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocabulary_size, 250),
            tf.keras.layers.Dropout(0.1, noise_shape=(None, sequence_length, 1)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, kernel_regularizer=tf.keras.regularizers.l2(1e-7), return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, kernel_regularizer=tf.keras.regularizers.l2(1e-7))),
            tf.keras.layers.Dense(10_000, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(nb_tags, activation='sigmoid')
        ])
        optimizer = tf.keras.optimizers.Adam(lr=1e-5)
        precision = tf.keras.metrics.Precision(name='precision')
        recall = tf.keras.metrics.Recall(name='recall')
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics = [precision, recall])
        print(model.summary())
        return model

    def fit(self, X, Y):
        nb_tags = Y.shape[1]
        vocabulary_size = X.max() + 1
        sequence_length = X.shape[1]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=min(10_000, int(0.2*X.shape[0])))
        batch_size = 256
        train_data_gen = self._yield_data(X_train, Y_train, batch_size, shuffle=True)
        test_data_gen = self._yield_data(X_test, Y_test, batch_size)
        train_steps_per_epoch = math.ceil(X_train.shape[0] / batch_size)
        val_steps_per_epoch = math.ceil(X_test.shape[0] / batch_size)

        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = self._build_model(vocabulary_size, sequence_length, nb_tags)
        model.fit(x=train_data_gen, steps_per_epoch=train_steps_per_epoch,
                  validation_data=test_data_gen, validation_steps=val_steps_per_epoch,
                  epochs=5)
        self.model = model
        return self

    def predict(self, X):
        batch_size = 256
        Y = []
        for i in range(0, X.shape[0], 256):
            X_batch = X[i:i+batch_size]
            Y_batch = self.model(X_batch) > 0.5
            Y.append(csr_matrix(Y_batch))
        Y = vstack(Y)
        return Y
