import pickle
import json
import math
import os

from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow_addons as tfa
import tensorflow as tf
import typer


def load_data(data_path):
    X = []
    Y = []
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line)
            X.append(item["text"])
            Y.append(item["tags"])
    return X, Y


def create_model(vocab_size, sequence_length, nb_labels, emb_size, nb_filters, hidden_dim):
    inputs = tf.keras.Input(shape=(sequence_length,))

    # Init weights with pretrain vectors
    x = tf.keras.layers.Embedding(vocab_size, emb_size, input_length=sequence_length)(inputs)
    x = tf.keras.layers.Dropout(0.25)(x)

    conv_outs = []
    for filter_size in [2, 4, 8]:
        conv_x = tf.keras.layers.Conv1D(nb_filters, filter_size, strides=2, activation="relu")(x)
        conv_x = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)(conv_x)
        conv_x = tf.keras.layers.Flatten()(conv_x)
        conv_outs.append(conv_x)
    x = tf.keras.layers.concatenate(conv_outs)

    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(nb_labels, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_xml_cnn(data_path, model_path):
    vocab_size = 30_000
    sequence_length = 500
    emb_size = 300
    nb_filters = 32
    hidden_dim = 512

    print("Loading data")
    X, Y = load_data(data_path)

    print("Fitting label binarizer")
    label_binarizer = MultiLabelBinarizer(sparse_output=True)
    Y_vec = label_binarizer.fit_transform(Y)

    print("Fitting tokenizer")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(vocab_size)
    tokenizer.fit_on_texts(X)
    X_vec = tokenizer.texts_to_sequences(X)
    X_vec = tf.keras.preprocessing.sequence.pad_sequences(X_vec, maxlen=sequence_length)

    nb_labels = Y_vec.shape[1]
    model = create_model(vocab_size, sequence_length, nb_labels, emb_size, nb_filters, hidden_dim)
    print(model.summary())

    print("Fitting model")
    epochs = 50
    batch_size = 256

    metrics = [tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tfa.metrics.F1Score(nb_labels, average="micro", name="f1")]
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=metrics)
    
    def yield_train_data(X_vec, Y_vec, batch_size):
        while True:
            for i in range(0, X_vec.shape[0], batch_size):
                X_batch = X_vec[i:i+batch_size,:]
                Y_batch = Y_vec[i:i+batch_size,:].todense()
                yield X_batch, Y_batch
            print("Ran out of data")

    train_data = yield_train_data(X_vec, Y_vec, batch_size)
    model.fit(train_data, epochs=epochs, steps_per_epoch=math.ceil(X_vec.shape[0]/batch_size))

    model.save(model_path)

    label_binarizer_path = os.path.join(model_path, "label_binarizer.pkl")
    with open(label_binarizer_path, "wb") as f:
        f.write(pickle.dumps(label_binarizer))

    tokenizer_path = os.path.join(model_path, "tokenizer.pkl")
    with open(tokenizer_path, "wb") as f:
        f.write(pickle.dumps(tokenizer))


if __name__ == "__main__":
    typer.run(train_xml_cnn)
