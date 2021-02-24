# encoding: utf-8
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
import pandas as pd
import numpy as np

def load_data(data_path, label_binarizer=None, X_format="List"):
    """Load data from the dataset."""
    print("Loading data...")

    texts = []
    tags = []
    meta = []
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)

            texts.append(data["text"])
            tags.append(data["tags"])
            meta.append(data["meta"])

    if label_binarizer:
        tags = label_binarizer.transform(tags)

    if X_format == "DataFrame":
        X = pd.DataFrame(meta)
        X["text"] = texts
        return X, tags, meta

    return texts, tags, meta

def load_train_test_data(
        train_data_path, label_binarizer,
        test_data_path=None, from_same_distribution=False,
        X_format="List", test_size=None):

    if test_data_path:
        X_train, Y_train, _ = load_data(train_data_path, label_binarizer, X_format)
        X_test, Y_test, _ = load_data(test_data_path, label_binarizer, X_format)

        if from_same_distribution:
            X_train, _, Y_train, _ = train_test_split(
                X_train, Y_train, random_state=42
            )
            _, X_test, _, Y_test = train_test_split(
                X_test, Y_test, random_state=42
            )
    else:
        X, Y, _ = load_data(train_data_path, label_binarizer, X_format)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, random_state=42, test_size=test_size
        )
           
    return X_train, X_test, Y_train, Y_test

def load_test_data(data_path, label_binarizer):
    X = []
    Y = []
    for x, y in yield_train_data(data_path, label_binarizer):
        X.extend(x)
        Y.extend(y)
    return X, np.array(Y)

# TODO: Move to common for cases where Y is a matrix
def calc_performance_per_tag(Y_true, Y_pred, tags):
    metrics = []
    for tag_index in range(Y_true.shape[1]):
        y_true_tag = Y_true[:,tag_index]
        y_pred_tag = Y_pred[:,tag_index]
        metrics.append({
            'Tag': tags[tag_index],
            'f1': f1_score(y_true_tag, y_pred_tag)
        })
    return pd.DataFrame(metrics)

def yield_texts(data_path):
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            yield item["text"]

def yield_tags(data_path):
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            yield item["tags"]

def load_dataset(data_path, tokenizer, label_binarizer, sparse_labels=False, data_cache=None, random_seed=42, shuffle=True, shuffle_buffer=1000):
    def data_gen():
        with open(data_path) as f:
            texts = []
            tags = []
            for line in f:
                item = json.loads(line)
                texts.append(item["text"])
                tags.append(item["tags"])

                if len(texts) >= 1000: 
                    text_encoded = tokenizer.transform(texts)
                    tags_encoded = label_binarizer.transform(tags)

                    if sparse_labels:
                        tags_encoded = tags_encoded.todense() # returns matrix
                        tags_encoded = np.asarray(tags_encoded)
                    
                    for i in range(len(texts)):
                        yield text_encoded[i], tags_encoded[i]

                    texts = []
                    tags = []
            
            # TODO: Refactor
            if texts:
                text_encoded = tokenizer.transform(texts)
                tags_encoded = label_binarizer.transform(tags)

                if sparse_labels:
                    tags_encoded = tags_encoded.todense() # returns matrix
                    tags_encoded = np.squeeze(np.asarray(tags_encoded))
                    
                for i in range(len(texts)):
                    yield text_encoded[i], tags_encoded[i]

    data = tf.data.Dataset.from_generator(data_gen, output_types=(tf.int32, tf.int32))

    if shuffle:
        data = data.shuffle(shuffle_buffer, seed=random_seed, reshuffle_each_iteration=False)
    if data_cache:
        data = data.cache(data_cache)
    return data

def load_train_test_dataset(data_path, tokenizer, label_binarizer, test_data_path=None, test_size=0.1, sparse_labels=False, data_cache=None, random_seed=42, shuffle=True, shuffle_buffer=1000):
    data = load_dataset(data_path, tokenizer, label_binarizer, sparse_labels=sparse_labels,
                        shuffle_buffer=shuffle_buffer, data_cache=data_cache, random_seed=random_seed)

    if test_data_path:
        test_data = load_dataset(data_path, tokenizer, label_binarizer, sparse_labels=sparse_labels,
                                 shuffle_buffer=shuffle_buffer, random_seed=random_seed) # cache will load train data if it has the same name
        train_data = data
    else:
        print("Splitting train and test. This might take a while.")
        steps = 0
        for _ in data:
            steps += 1

        train_steps = int((1-test_size) * steps)
    
        train_data = data.take(train_steps)
        test_data = data.skip(train_steps)
        print(f"Splitted. Train data size {train_steps}. Test data size {steps-train_steps}")

    if shuffle:
        train_data = train_data.shuffle(shuffle_buffer, seed=random_seed)
        test_data = test_data.shuffle(shuffle_buffer, seed=random_seed)

    return train_data, test_data
