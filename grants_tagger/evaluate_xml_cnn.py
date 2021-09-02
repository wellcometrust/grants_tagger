import pickle
import os

from sklearn.metrics import precision_recall_fscore_support
import tensorflow_addons as tfa
import tensorflow as tf
import typer

from grants_tagger.train_xml_cnn import load_data


def load_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        obj = pickle.loads(f.read())
    return obj

def evaluate_xml_cnn(test_data_path, model_path):
    label_binarizer = load_pickle(os.path.join(model_path, "label_binarizer.pkl"))
    tokenizer = load_pickle(os.path.join(model_path, "tokenizer.pkl"))
    model = tf.keras.models.load_model(os.path.join(model_path, "model"))

    X, Y = load_data(test_data_path)
    X_vec = tokenizer.texts_to_sequences(X)
    X_vec = tf.keras.preprocessing.sequence.pad_sequences(X_vec, maxlen=500)
    Y_vec = label_binarizer.transform(Y)

    Y_pred = model.predict(X_vec)
    p, r, f1, _ = precision_recall_fscore_support(Y_vec, Y_pred, average="micro")
    print(f"P: {p} R: {r} f1: {f1}")


if __name__ == "__main__":
    typer.run(evaluate_xml_cnn)
