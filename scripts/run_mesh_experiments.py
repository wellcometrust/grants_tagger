"""
Experiments on disease mesh tags to understand role of params
"""
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json
import math

from wellcomeml.ml import KerasVectorizer, CNNClassifier, BiLSTMClassifier
from tensorboard.plugins.hparams import api as hp
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from scipy.sparse import vstack, csr_matrix
from tqdm import tqdm
import tensorflow as tf
import numpy as np


def create_dataset(data_path, nb_tags, nb_examples_per_tag):
    texts = []
    tags = []

    tags_count = {}
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            item_tags = []
            for tag in item["tags"]:
                if tag not in tags_count:
                    if len(tags_count) < nb_tags:
                        tags_count[tag] = 1
                        item_tags.append(tag)
                    else:
                        # tags full
                        pass
                else: # tag in tags count
                    tags_count[tag] += 1
                    item_tags.append(tag)
            if item_tags:
                texts.append(item["text"])
                tags.append(item_tags)
            if all([c > nb_examples_per_tag for t, c in tags_count.items()]):
                break
        return texts, tags

def vectorize_data(train_texts, train_tags, test_texts, test_tags,
                   vocabulary_size, sequence_length):
    # fit vectorizer and transform
    vectorizer = KerasVectorizer(vocabulary_size, sequence_length)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # vectorize tags
    label_binarizer = MultiLabelBinarizer(sparse_output=True)
    label_binarizer.fit(train_tags)
    Y_train = label_binarizer.transform(train_tags)
    Y_test = label_binarizer.transform(test_tags)
    return X_train, X_test, Y_train, Y_test

def build_cnn_model(learning_rate=0.01, batch_size=256, attention=True,
                    vocabulary_size=400_000, sequence_length=400, nb_tags=512,
                    l2=1e-6, dropout=0.1, hidden_size=100, dense_size=100_000):
    model = CNNClassifier(
        attention=attention, multilabel=True,
        learning_rate=learning_rate, batch_size=batch_size,
        nb_layers=4, dense_size=dense_size, hidden_size=hidden_size,
        dropout=dropout, l2=l2
    )
    model = model._build_model(vocab_size=vocabulary_size, sequence_length=sequence_length, nb_outputs=nb_tags)
    print(model.summary())
    return model

def build_bilstm_model(learning_rate=0.01, batch_size=256, attention=True,
                       vocabulary_size=400_000, sequence_length=400, nb_tags=512,
                       l2=1e-6, dropout=0.1, hidden_size=100, dense_size=100_000):
    model = BiLSTMClassifier(
        attention=attention, multilabel=True,
        learning_rate=learning_rate, batch_size=batch_size,
        dense_size=dense_size, l2=l2, hidden_size=hidden_size,
        dropout=dropout
    )
    model = model._build_model(vocab_size=vocabulary_size, sequence_length=sequence_length, nb_outputs=nb_tags)
    print(model.summary())
    return model

def build_custom_bilstm_model(learning_rate=0.01, batch_size=256, attention=True,
                              vocabulary_size=400_000, sequence_length=400, nb_tags=512,
                              l2=1e-6, dropout=0.1, hidden_size=100, dense_size=100_000):
    bias_init = 0 #np.log(Y_train.sum(axis=0) / (Y_train.shape[0] - Y_train.sum(axis=0)))
    
    # define model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocabulary_size, hidden_size, input_length=sequence_length,
            mask_zero=False, embeddings_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dropout(dropout, noise_shape=(None, sequence_length, 1)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(hidden_size/2), kernel_regularizer=tf.keras.regularizers.l2(l2), recurrent_regularizer=tf.keras.regularizers.l2(l2), dropout=dropout, recurrent_dropout=0, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(hidden_size/2), kernel_regularizer=tf.keras.regularizers.l2(l2), recurrent_regularizer=tf.keras.regularizers.l2(l2), dropout=dropout, recurrent_dropout=0)),
        tf.keras.layers.Dense(dense_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(nb_tags, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2), bias_initializer=tf.keras.initializers.Constant(bias_init))
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[])
    print(model.summary())
    return model

def train(X_train, X_test, Y_train, Y_test, params):
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    l2 = params["l2"]
    dropout = params["dropout"]
    nb_tags = params["nb_tags"]
    hidden_size = params["hidden_size"]
    dense_size = params["dense_size"]
    attention = params["attention"]
    architecture = params["architecture"]
    epochs = params["epochs"]
    vocabulary_size = params["vocabulary_size"]
    sequence_length = params["sequence_length"]

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{learning_rate}-{batch_size}"
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    def yield_data(X, Y, batch_size, shuffle=True):
        while True:
            if shuffle:
                randomize = np.arange(len(X))
                np.random.shuffle(randomize)
                X = X[randomize]
                Y = Y[randomize]
            for i in range(0, X.shape[0], batch_size):
                yield X[i:i+batch_size, :], Y[i:i+batch_size, :].todense()

    train_data = yield_data(X_train, Y_train, batch_size)
    test_data = yield_data(X_test, Y_test, batch_size)
    steps_per_epoch = math.ceil(X_train.shape[0]/batch_size)
    validation_steps = math.ceil(X_test.shape[0]/batch_size)

    if architecture == 'cnn':
        build_model_f = build_cnn_model
    elif architecture == 'bilstm':
        build_model_f = build_bilstm_model
    else:
        build_model_f = build_custom_bilstm_model
    model = build_model_f(
        learning_rate=learning_rate, nb_tags=nb_tags, attention=attention,
        l2=l2, dropout=dropout, hidden_size=hidden_size, dense_size=dense_size,
        vocabulary_size=vocabulary_size, sequence_length=sequence_length)
    history = model.fit(x=train_data, steps_per_epoch=steps_per_epoch,
                        validation_data=test_data, validation_steps=validation_steps,
                        epochs=epochs, callbacks=[tensorboard, early_stopping])

    steps = len(history.history["loss"]) * steps_per_epoch * batch_size

    # evaluate                                                                                                                 
    def predict(model, data_gen, steps):
        Y_pred = []
        for _ in tqdm(range(steps)):
            X_batch, _ = next(data_gen)
            Y_pred_batch = model(X_batch) > 0.5
            Y_pred.append(csr_matrix(Y_pred_batch))
        Y_pred = vstack(Y_pred)
        return Y_pred
 
    test_data = yield_data(X_test, Y_test, batch_size, shuffle=False)

    Y_pred_test = predict(model, test_data, validation_steps)
    f1_test = f1_score(Y_test, Y_pred_test, average='micro')
    return f1_test, steps

def experiment(data_path, params):
    nb_tags = params["nb_tags"]
    nb_examples_per_tag = params["nb_examples_per_tag"]
    vocabulary_size = params["vocabulary_size"]
    sequence_length = params["sequence_length"]
    texts, tags = create_dataset(
        data_path, nb_tags, nb_examples_per_tag)

    nb_train = len(texts) - min(int(0.2*len(texts)), 10_000)
    train_texts = texts[:nb_train]
    train_tags = tags[:nb_train]
    test_texts = texts[nb_train:]
    test_tags = tags[nb_train:]

    X_train, X_test, Y_train, Y_test = vectorize_data(
        train_texts, train_tags, test_texts, test_tags,
        vocabulary_size, sequence_length)

    f1, steps = train(X_train, X_test, Y_train, Y_test, params)

    run_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    with tf.summary.create_file_writer(f"logs/hparam_tuning/run-{run_datetime}".format()).as_default():
        hp.hparams(params)
        tf.summary.scalar('f1', f1, step=1)
        tf.summary.scalar('steps', steps, step=1)
    return f1, steps

def run_mesh_experiments(data_path, learning_rate, batch_size, nb_tags, nb_examples_per_tag,
                         dropout, l2, hidden_size, dense_size, attention, architecture,
                         epochs, vocabulary_size, sequence_length):
    params = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "nb_tags": nb_tags,
        "nb_examples_per_tag": nb_examples_per_tag,
        "dropout": dropout,
        "l2": l2,
        "hidden_size": hidden_size,
        "dense_size": dense_size,
        "attention": attention,
        "architecture": architecture,
        "epochs": epochs,
        "vocabulary_size": vocabulary_size,
        "sequence_length": sequence_length
    }
    if type(learning_rate) is str and ',' in learning_rate:
        for learning_rate in learning_rate.split(','):
            params["learning_rate"] = float(learning_rate)
            experiment(data_path, params)

if __name__ == '__main__':
    argparser = ArgumentParser(description=__doc__.strip())
    argparser.add_argument('--data_path', type=Path)
    argparser.add_argument('--learning_rate', default=0.01)
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--nb_tags', type=int, default=32)
    argparser.add_argument('--nb_examples_per_tag', type=int, default=100)
    argparser.add_argument('--dropout', type=float, default=0.1)
    argparser.add_argument('--l2', type=float, default=1e-7)
    argparser.add_argument('--hidden_size', type=int, default=100)
    argparser.add_argument('--dense_size', type=int, default=10_000)
    argparser.add_argument('--attention', type=bool, default=True)
    argparser.add_argument('--architecture', default="cnn")
    argparser.add_argument('--epochs', type=int, default=500)
    argparser.add_argument('--vocabulary_size', type=int, default=400_000)
    argparser.add_argument('--sequence_length', type=int, default=400)
    argparser.add_argument('--config', type=Path)
    args = argparser.parse_args()

    if args.config:
        cfg = ConfigParser()
        cfg.read(args.config)

        data_path = cfg["data"]["data_path"]
        learning_rate = cfg["learning_rate"]["learning_rate"]
        batch_size = cfg["model"]["batch_size"]
        nb_tags = cfg["data"]["nb_tags"]
        nb_examples_per_tag = cfg["data"]["nb_examples_per_tag"]
        dropout = cfg["model"]["dropout"]
        l2 = cfg["model"]["l2"]
        hidden_size = cfg["model"]["hidden_size"]
        dense_size = cfg["model"]["dense_size"]
        attention = cfg["model"]["attention"]
        architecture = cfg["model"]["architecture"]
        epochs = cfg["model"]["epochs"]
        vocabulary_size = cfg["vectorizer"]["vocabulary_size"]
        sequence_length = cfg["vectorizer"]["sequence_length"]
    else:
        data_path = args.data_path
        learning_rate = args.learning_rate
        batch_size = args.batch_size
        nb_tags = args.nb_tags
        nb_examples_per_tag = args.nb_examples_per_tag
        dropout = args.dropout
        l2 = args.l2
        hidden_size = args.hidden_size
        dense_size = args.dense_size
        attention = args.attention
        architecture = args.architecture
        epochs = args.epochs
        vocabulary_size = args.vocabulary_size
        sequence_length = args.sequence_length
    
    run_mesh_experiments(
        data_path,
        learning_rate,
        batch_size,
        nb_tags,
        nb_examples_per_tag,
        dropout,
        l2,
        hidden_size,
        dense_size,
        attention,
        architecture,
        epochs,
        vocabulary_size,
        sequence_length
    )
