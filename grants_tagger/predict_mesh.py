"""
"""
import pickle
import os

from scipy.sparse import hstack


def predict_mesh_tags(X, model_path, label_binarizer_path,
                      probabilities=False, threshold=None):
    # TODO: generalise tfidf to vectorizer.pkl
    with open(f"{model_path}/tfidf.pkl", "rb") as f:
        vectorizer = pickle.loads(f.read())
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    Y_pred_proba = []
    for tag_i in os.listdir(model_path):
        if tag_i == "vectorizer.pkl":
            continue
        with open(f"{model_path}/{tag_i}.pkl", "rb") as f:
            classifier = pickle.loads(f.read())
        X_vec = vectorizer.transform(X)
        Y_pred_i = classifier.predict_proba(X_vec)
        Y_pred_proba.append(Y_pred_i)
    Y_pred_proba = hstack(Y_pred_proba)
    
    tags = []
    for y_pred_proba in Y_pred_proba:
        tags_i = [tag for tag, prob in zip(label_binarizer.classes_, y_pred_proba) if prob > 0.5]
        tags.append(tags_i)
    return tags

if __name__ == '__main__':
    X = ["malaria"]
    tags = predict_mesh_tags(X, "disease_mesh_tfidf-svm-2020.07.0/", "models/")
    print(tags)
