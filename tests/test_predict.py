import tempfile
import shutil
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from wellcomeml.ml import BertClassifier

from grants_tagger.predict import predict_tags

X = [
    "all",
    "one two",
    "two",
    "four",
    "twenty four"
]
Y = [
    [str(i) for i in range(24)],
    ["1", "2"],
    ["2"],
    ["4"],
    ["24"]
]

def train_test_model(scibert_path, tfidf_svm_path, label_binarizer_path):
    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit(Y)
    with open(f"{label_binarizer_path}", "wb") as f:
        f.write(pickle.dumps(label_binarizer))
        f.seek(0)

    Y_vec = label_binarizer.transform(Y)

    tfidf_svm = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svm', OneVsRestClassifier(SGDClassifier(loss='log')))
    ])
    
    tfidf_svm.fit(X, Y_vec)
    with open(tfidf_svm_path, "wb") as f:
        f.write(pickle.dumps(tfidf_svm))

    scibert = BertClassifier()
    scibert.fit(X, Y_vec)
    scibert.save(scibert_path)

def test_predict_mesh():
    with tempfile.TemporaryDirectory() as tmp_dir:
        scibert_path = f"{tmp_dir}/model/"
        os.mkdir(scibert_path)
        tfidf_svm_path = f"{tmp_dir}/tfidf_svm.pkl"
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        train_test_model(scibert_path, tfidf_svm_path, label_binarizer_path)

        tags = predict_tags(X, scibert_path=scibert_path, tfidf_svm_path=tfidf_svm_path,
                            label_binarizer_path=label_binarizer_path)
        assert len(tags) == 5
