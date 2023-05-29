import json
import os
import torch
import numpy as np
import scipy
import math
import typer

from argparse import ArgumentParser
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


@torch.no_grad()
def evaluate_pubs(
    data_path: str, threshold: float = 0.5, batch_size: int = 4, device: str = "cpu"
):
    with open(data_path, "r") as f:
        data = json.load(f)

    # Filter publications without tags
    data = [pub for pub in data if pub["mesh_terms"]]

    tokenizer = AutoTokenizer.from_pretrained("Wellcome/WellcomeBertMesh")
    model = AutoModel.from_pretrained(
        "Wellcome/WellcomeBertMesh", trust_remote_code=True
    )
    model.to(device)
    model.eval()

    id2label = model.id2label
    label2id = {v: k for k, v in id2label.items()}
    mlb = MultiLabelBinarizer()
    mlb.fit([list(label2id.values())])

    # Dump labels to a file
    with open("labels.txt", "w") as f:
        for label in label2id:
            f.write(label + "\n")

    # Generate Y_true
    Y_true = []

    ignored_tags = []

    for pub in data:
        id_list = []
        for label in pub["mesh_terms"]:
            if label not in label2id:
                ignored_tags.append(label)
                continue
            id_list.append(label2id[label])
        Y_true.append(id_list)

    # Generate Y_pred
    pbar = tqdm(total=len(data))
    Y_pred = []

    for idx in range(0, len(data), batch_size):
        batch = data[idx : idx + batch_size]
        batch_abstracts = [pub["abstract"] for pub in batch]
        inputs = tokenizer(
            batch_abstracts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        inputs = inputs.to(device)

        outs = model(**inputs, return_labels=False)

        # Manually return the labels with custom threshold
        outs = [
            [
                label_id
                for label_id, label_prob in enumerate(out)
                if label_prob > threshold
            ]
            for out in outs
        ]

        Y_pred.extend(outs)

        pbar.update(batch_size)

    pbar.close()

    Y_true = mlb.transform(Y_true)
    Y_pred = mlb.transform(Y_pred)

    report = classification_report(Y_true, Y_pred, output_dict=True)

    report = {
        "micro avg": report["micro avg"],
        "macro avg": report["macro avg"],
        "weighted avg": report["weighted avg"],
        "samples avg": report["samples avg"],
    }

    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=str, help="Path to data")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    evaluate_pubs(args.data_path, batch_size=args.batch_size, device=args.device)
