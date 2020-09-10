import json
import math

from wellcomeml.ml import KerasVectorizer, CNNClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import numpy as np

DEFAULT_VOCABULARY_SIZE = 400_000
DEFAULT_SEQUENCE_LENGTH = 400
DEFAULT_NB_TAGS = 512

def create_dataset(data_path, nb_tags=DEFAULT_NB_TAGS, nb_examples_per_tag=10_000):
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
                   vocabulary_size=DEFAULT_VOCABULARY_SIZE,
                   sequence_length=DEFAULT_SEQUENCE_LENGTH):
    # fit vectorizer and transform
    vectorizer = KerasVectorizer(vocabulary_size, sequence_length)
    X_train = vectorizer.transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # vectorize tags
    label_binarizer = MultiLabelBinarizer(sparse=True)
    label_binarizer.fit(train_tags)
    Y_train = label_binarizer.transform(train_tags)
    Y_test = label_binarizer.transform(test_tags)
    return X_train, X_test, Y_train, Y_test

def build_model(learning_rate=0.01, batch_size=256, attention=True,
                vocabulary_size=DEFAULT_VOCABULARY_SIZE,
                sequence_length=DEFAULT_SEQUENCE_LENGTH, nb_tags=512):
    model = CNNClassifier(
        attention=attention, multilabel=True,
        learning_rate=learning_rate, batch_size=batch_size
    )
    model = model._build_model(vocabulary_size, sequence_length, nb_tags)
    return model

def train(X_train, X_test, Y_train, Y_test, learning_rate=0.01, batch_size=256, nb_epochs=5):
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)

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

    model = build_model(learning_rate=learning_rate)
    model.fit(x=train_data, steps_per_epoch=steps_per_epoch,
              validation_data=test_data, validation_steps=validation_steps,
              epochs=nb_epochs, callbacks=[tensorboard])

def learning_rate_experiment(train_data_path, test_data_path):
    # create dataset
    train_texts, train_tags = create_dataset(train_data_path, 32, 100)
    test_texts, test_tags = create_dataset(test_data_path, 32, 100)

    # vectorize data
    X_train, X_test, Y_train, Y_test = vectorize_data(train_texts, train_tags, test_texts, test_tags)

    # for param train
    for learning_rate in [0.1, 0.01]:
        # build model with params
        train(learning_rate)
        # note we want to stop training when loss converges (possibly stop improving for some iterations)

        # evaluate on best model?

learning_rate_experiment("train_data_path", "test_data_path")
