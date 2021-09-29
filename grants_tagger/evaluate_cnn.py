import pickle
import json

from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import typer

from wellcomeml.ml import CNNClassifier


def load_data(data_path):
    with open(data_path) as f:
        texts = []
        tags = []
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            tags.append(item["tags"])
    return texts, tags

def get_disease_indices(mesh_tree_path, labels):
    mesh_tree = ET.parse(mesh_tree_path)

    for mesh in tqdm(mesh_tree.getroot()):
        try:
            # TreeNumberList e.g. A11.118.637.555.567.550.500.100
            mesh_tree = mesh[-2][0].text
            # DescriptorUI e.g. M000616943
            mesh_code = mesh[0].text
            # DescriptorName e.g. Mucosal-Associated Invariant T Cells
            mesh_name = mesh[1][0].text
        except IndexError:
            pass
        
        label2index = {l: i for i, l in enumerate(labels)}

        disease_indices = []
        if mesh_tree.startswith('C') and not mesh_tree.startswith('C22') or mesh_tree.startswith('F03'):
            label_index = label2index.get(mesh_name)
            if label_index:
                disease_indices.append(label_index)
        return disease_indices

def evaluate_cnn(data_path, model_path, vectorizer_path, label_binarizer_path,
        figure_path=None, figure_data_path=None, mesh_tree_path=None, precision_k=False):
    X, Y = load_data(data_path)
    
    print("loading label binarizer")
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())
    
    print("loading vectorizer")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.loads(f.read())

    if mesh_tree_path:
        print("getting disease indices")
        disease_indices = get_disease_indices(mesh_tree_path, label_binarizer.classes_)
        print(f"found {len(disease_indices)} disease codes")

    print("transforming X")
    X_vec = vectorizer.transform(X)

    print("transforming Y")
    Y_vec = label_binarizer.transform(Y)
    if mesh_tree_path:
        Y_vec = Y_vec[:, disease_indices]

    print("predicting probs")
    model = CNNClassifier(sparse_y=True)
    model.load(model_path)

    batch_size = 512
    Y_pred_proba = []
    for i in tqdm(range(0, X_vec.shape[0], batch_size)):
        y_pred_proba_batch = model.predict_proba(X_vec[i:i+batch_size,:])
        y_pred_proba_batch[y_pred_proba_batch < 0.01] = 0
        Y_pred_proba.append(sp.csr_matrix(y_pred_proba_batch))
    Y_pred_proba = sp.vstack(Y_pred_proba)
   
    print(Y_pred_proba.shape)
    print(Y_vec.shape)

    print("evaluating")
    for th in [0.1, 0.2, 0.3, 0.4, 0.5]:
        if mesh_tree_path:
            Y_pred = Y_pred[:, disease_indices]

        Y_pred = Y_pred_proba > th
        p, r, f1, _ = precision_recall_fscore_support(Y_vec, Y_pred, average="micro")
        print(f"Th: {th} P: {p}, R: {r} f1: {f1}")

    if precision_k:
        import xclib.evaluation.xc_metrics as xc_metrics
        pk = xc_metrics.precision(Y_pred_proba, Y_vec)
        print(" ".join([f"P@{k}: {pk[k-1]}" for k in [1, 3, 5]]))
    
    if figure_path:
        Y_pred = Y_pred_proba > 0.5
        cm = multilabel_confusion_matrix(Y_vec, Y_pred)

        plot_data = []
        for i in tqdm(range(Y_vec.shape[1])):
            tn, fp, fn, tp = cm[i, :, :].ravel()
            denominator = tp + (fp + fn) / 2
            if denominator:
                f1 = tp / denominator
            else:
                f1 = 0
            nb_examples = Y_vec[:,i].sum()
            plot_data.append((f1, nb_examples))
        
        x, y = zip(*plot_data)
        plt.scatter(x, y)
        plt.savefig(figure_path)

    if figure_data_path:
        with open(figure_data_path, "w") as f:
            for x, y in plot_data:
                f.write(f"{x},{y}\n")
    
if __name__ == "__main__":
    typer.run(evaluate_cnn)
