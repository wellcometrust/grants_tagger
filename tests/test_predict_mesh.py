import tempfile
import shutil
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from wellcomeml.ml import CNNClassifier, KerasVectorizer

from grants_tagger.predict_mesh import predict_mesh_tags, predict_tfidf_svm

X = [
    "all",
    "one two",
    "two",
    "two hundred",
    "one thousand"
]
Y = [
    [str(i) for i in range(5000)],
    ["1", "2"],
    ["2"],
    ["200"],
    ["1000"]
]

def train_test_tfidf_svm_model(model_path, label_binarizer_path):
    tfidf = TfidfVectorizer()
    tfidf.fit(X)
    with open(f"{model_path}/tfidf.pkl", "wb") as f:
        f.write(pickle.dumps(tfidf))

    X_vec = tfidf.transform(X)

    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit(Y)
    with open(f"{label_binarizer_path}", "wb") as f:
        f.write(pickle.dumps(label_binarizer))
        f.seek(0)

    Y_vec = label_binarizer.transform(Y)

    for tag_i in range(0, Y_vec.shape[1], 512):
        model = OneVsRestClassifier(SGDClassifier(loss='log'))
        model.fit(X_vec, Y_vec[:, tag_i:tag_i+512])
        with open(f"{model_path}/{tag_i}.pkl", "wb") as f:
            f.write(pickle.dumps(model))

def train_test_cnn_model(model_path, label_binarizer_path):
    vectorizer = KerasVectorizer()
    vectorizer.fit(X)
    with open(f"{model_path}/vectorizer.pkl", "wb") as f:
        f.write(pickle.dumps(vectorizer))

    X_vec = vectorizer.transform(X)

    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit(Y)
    with open(f"{label_binarizer_path}", "wb") as f:
        f.write(pickle.dumps(label_binarizer))
        f.seek(0)

    Y_vec = label_binarizer.transform(Y)

    model = CNNClassifier(multilabel=True)
    model.fit(X_vec, Y_vec)
    model.save(model_path)


def test_predict_mesh_tfidf_svm():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = f"{tmp_dir}/tfidf_model/"
        os.mkdir(model_path)
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        train_test_tfidf_svm_model(model_path, label_binarizer_path)

        tags = predict_mesh_tags(X, model_path, label_binarizer_path)
        assert len(tags) == 5


def test_predict_mesh_tfidf_svm_threshold():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = f"{tmp_dir}/tfidf_model/"
        os.mkdir(model_path)
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        train_test_tfidf_svm_model(model_path, label_binarizer_path)

        nb_examples = 5
        nb_labels = 5000
        Y = predict_tfidf_svm(X, model_path, nb_labels, threshold=0, return_probabilities=False)
        assert Y.sum() == nb_labels * nb_examples # all 1

        Y = predict_tfidf_svm(X, model_path, nb_labels, threshold=1, return_probabilities=False)
        assert Y.sum() == 0 # all 0


def test_predict_mesh_cnn():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = f"{tmp_dir}/cnn_model/"
        os.mkdir(model_path)
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        train_test_cnn_model(model_path, label_binarizer_path)

        tags = predict_mesh_tags(X, model_path, label_binarizer_path)
        assert len(tags) == 5
