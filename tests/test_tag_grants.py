import tempfile
import shutil
import pickle
import csv
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from wellcomeml.ml import BertClassifier

from grants_tagger.tag_grants import tag_grants

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

def test_tag_grants_with_mesh():
    with tempfile.TemporaryDirectory() as tmp_dir:
        scibert_path = f"{tmp_dir}/model/"
        os.mkdir(scibert_path)
        tfidf_svm_path = f"{tmp_dir}/tfidf_svm.pkl"
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        train_test_model(scibert_path, tfidf_svm_path, label_binarizer_path)
        
        tagged_grants_path = f"{tmp_dir}/tagged_grants.csv"
        grants_path = f"{tmp_dir}/grants.csv"
        with open(grants_path, "w") as tmp_grants:
            csvwriter = csv.DictWriter(tmp_grants, fieldnames=["title", "synopsis", "grant_id", "grant_no", "reference"])
            csvwriter.writeheader()

            for i, x in enumerate(X):
                csvwriter.writerow({
                    "title": "",
                    "synopsis": x,
                    "grant_id": i,
                    "reference": i,
                    "grant_no": i
                })

            tmp_grants.seek(0)

            tag_grants(grants_path, tagged_grants_path, scibert_path=scibert_path, tfidf_svm_path=tfidf_svm_path, label_binarizer_path=label_binarizer_path)
