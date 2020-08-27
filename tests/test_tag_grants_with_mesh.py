import tempfile
import shutil
import pickle
import csv
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

from grants_tagger.tag_grants_with_mesh import tag_grants_with_mesh

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

def train_test_model(model_path, label_binarizer_path):
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

def test_tag_grants_with_mesh():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = f"{tmp_dir}/model/"
        os.mkdir(model_path)
        label_binarizer_path = f"{tmp_dir}/label_binarizer.pkl"
        train_test_model(model_path, label_binarizer_path)

        with tempfile.NamedTemporaryFile(mode="w+") as tmp_grants:
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
            grants_path = tmp_grants.name

            tmp_tagged_grants = tempfile.NamedTemporaryFile()
            tagged_grants_path = tmp_tagged_grants.name

            tag_grants_with_mesh(grants_path, tagged_grants_path, model_path=model_path, label_binarizer_path=label_binarizer_path)
