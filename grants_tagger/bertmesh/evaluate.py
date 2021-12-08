import json

from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.sparse as sp
import torch
import typer

from grants_tagger.bertmesh.train_torch import BertMesh, MultiLabelAttention, MeshDataset


def evaluate(x_path, y_path, model_path, batch_size:int=256, threshold:float=0.5, results_path="results.json"):
    test_dataset = MeshDataset(x_path, y_path)
    test_data = DataLoader(test_dataset, batch_size)

    model = torch.load(model_path)
    model.eval()
    
    Y_pred = []
    with torch.no_grad():
        for data in tqdm(test_data):
            inputs, _ = data

            outs = model(inputs)

            Y_pred.append(sp.csr_matrix(outs.cpu().numpy() > threshold))

    Y_pred = sp.vstack(Y_pred)
    Y_test = test_dataset.Y
    
    p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average="micro")
    print(f"P {p:.2f} R {r:.2f} f1 {f1:.2f}")

    with open(results_path, "w") as f:
        f.write(json.dumps({"p": p, "r": r, "f1": f1}))


if __name__ == "__main__":
    typer.run(evaluate)
