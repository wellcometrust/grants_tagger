from datetime import datetime
import json

from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoConfig
from tqdm import tqdm
import scipy.sparse as sp
import torch
import typer
import wandb

from grants_tagger.bertmesh.model import BertMesh, MultiLabelAttention
from grants_tagger.bertmesh.data import MeshDataset


def evaluate(
    x_path,
    y_path,
    model_path,
    batch_size: int = 256,
    threshold: float = 0.5,
    cutoff_prob: float = 0.1,
    results_path="results.json",
    pr_curve_path="pr_curve.json",
    experiment_name=datetime.now().strftime("%d/%m/%Y"),
    dry_run: bool = False,
):
    if not dry_run:
        wandb.init(project="bertmesh", group=experiment_name, job_type="eval")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = MeshDataset(x_path, y_path)
    test_data = DataLoader(test_dataset, batch_size)

    model = BertMesh.from_pretrained(model_path)
    model.to(device)
    model.eval()

    Y_pred_proba = []
    with torch.no_grad():
        for data in tqdm(test_data):
            inputs = data[0].to(device)

            outs = model(inputs)

            Y_pred_proba_batch = outs.cpu().numpy()
            Y_pred_proba_batch[Y_pred_proba_batch < cutoff_prob] = 0
            Y_pred_proba.append(sp.csr_matrix(Y_pred_proba_batch))

            if dry_run:
                break

    Y_pred_proba = sp.vstack(Y_pred_proba)
    Y_test = test_dataset.Y

    if dry_run:
        Y_test = Y_test[: Y_pred_proba.shape[0]]

    pr_curve = []
    for th in [0.1, 0.2, 0.3, 0.4, 0.5]:
        Y_pred = Y_pred_proba > th
        p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average="micro")
        metrics = {"p": p, "r": r, "f1": f1, "th": th}
        pr_curve.append(metrics)
        print(f"P {p:.2f} R {r:.2f} f1 {f1:.2f}")

    with open(results_path, "w") as f:
        f.write(json.dumps(metrics))

    with open(pr_curve_path, "w") as f:
        f.write(json.dumps({"pr_curve": pr_curve}))

    if not dry_run:
        wandb.log(metrics)
        table = wandb.Table(
            data=[[point["p"], point["r"]] for point in pr_curve],
            columns=["precision", "recall"],
        )
        wandb.log(
            {
                "pr_curve": wandb.plot.line(
                    table, "precision", "recall", title="PR curve"
                )
            }
        )


if __name__ == "__main__":
    typer.run(evaluate)
