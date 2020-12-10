"""
Pretrain a model on unlabeled data to improve feature extraction
"""
from wellcomeml.ml import Doc2VecVectorizer
import pandas as pd


def pretrain(data_path, model_path, model_name):
    # TODO: Convert that to assume a JSONL with text field
    data = pd.read_csv(data_path)
    X = data["synopsis"].dropna().drop_duplicates()

    if model_name == "doc2vec":
        model = Doc2VecVectorizer(
            min_count=1,
            window_size=9,
            vector_size=300,
            negative=10,
            sample=1e-4,
            epochs=5
        )
    else:
        raise NotImplementedError
    model.fit(X)

    model.save(model_path)
