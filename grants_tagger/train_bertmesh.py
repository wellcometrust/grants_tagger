import random
import math

from transformers import TFBertModel
import tensorflow_addons as tfa
import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import typer


class MultiLabelAttention(tf.keras.layers.Layer):
    def __init__(self, labels):
        super(MultiLabelAttention, self).__init__()
        self.labels = labels

    def build(self, input_shape):
        # input_shape (batch_size, sequence_length, emb_dim)
        self.A = self.add_weights(
            shape=(input_shape[-1], labels),
            initializer="random",
            trainable=True,
            name="multilabel_attention"
        )

    def call(self, inputs):
        attention_weights = tf.nn.softmax(tf.math.tanh(tf.matmul(inputs, self.A), axis=1)) # softmax over seq
        # attention_weights (batch_size, seq_len, labels)
        # inputs (batch_size, seq_len, emb_dim)
        return tf.matmul(attention_weights, inputs, tranpose_a=True) # (batch_size, labels, emb_dim)

def train_bertmesh(x_path, y_path, model_path, multilabel_attention=False):
    X = np.load(x_path)
    Y = sp.load_npz(y_path)

    inputs = tf.keras.Input(shape=X.shape[1], dtype=X.dtype)
    bert = TFBertModel.from_pretrained("bert-base-uncased")

    # TODO: Optionally concatenate vectors from 4 last layers
    cls = bert(inputs)[0]
    if multilabel_attention:
        attention_outs = MultiLabelAttention(Y.shape[1])(cls)
        dense = tf.keras.layers.Dense(512, activation="relu")(attention_outs) # this will return one dense per attention label
        out = tf.keras.layers.Dense(1, activation="sigmoid")(dense) # 1 output per label
        out = tf.keras.layers.Flatten()(out)
    else:
        cls = tf.keras.layers.GlobalMaxPooling1D()(cls)
        dense = tf.keras.layers.Dense(512, activation="relu")(cls)
        out = tf.keras.layers.Dense(Y.shape[1], activation="sigmoid")(dense)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    print(model.summary())

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    metrics = [tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"),
        tfa.metrics.F1Score(Y.shape[1], average="micro", threshold=0.5, name="f1")]
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)
    
    def yield_data(X, Y, batch_size):
        indices = list(range(X.shape[0]))
        while True:
            random.shuffle(indices)
            X = X[indices,:]
            Y = Y[indices,:]
            for i in range(0, X.shape[0], batch_size):
                yield X[i:i+batch_size,:], Y[i:i+batch_size,:].todense()
    
    batch_size = 256
    steps_per_epoch = math.ceil(X.shape[0]/batch_size)
    
    train_data = yield_data(X, Y, batch_size)
    print("iterate over data")
    batches = sum([1 for _ in train_data])
    print(f"found {batches} batches")
    model.fit(train_data, epochs=5, steps_per_epoch=steps_per_epoch)
    model.save(model_path)


if __name__ == "__main__":
    typer.run(train_bertmesh)


