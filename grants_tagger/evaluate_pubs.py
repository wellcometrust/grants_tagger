import json
import typer
import torch

from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from rich import print

evaluate_pubs_app = typer.Typer()


@torch.no_grad()
def evaluate_pubs(
    data_path: str, threshold: float = 0.5, batch_size: int = 4, device: str = "cpu"
):
    if not torch.cuda.is_available() and device == "cuda":
        print("[bold red]CUDA is not available, using CPU")
        device = "cpu"

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

    num_ignored_tags = 0

    for pub in data:
        id_list = []
        for label in pub["mesh_terms"]:
            if "/" in label:
                label = label.split("/", 1)[0]

            if label.endswith("*"):
                label = label[:-1]

            if label not in label2id:
                num_ignored_tags += 1
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


@evaluate_pubs_app.command()
def evaluate_pubs_cli(
    data_path: str = typer.Argument(..., help="Path to data"),
    threshold: float = typer.Option(0.5, help="Threshold for classification"),
    batch_size: int = typer.Option(4, help="Batch size"),
    device: str = typer.Option("cpu", help="Device to run on. CPU or CUDA"),
):
    evaluate_pubs(data_path, threshold, batch_size, device)


if __name__ == "__main__":
    evaluate_pubs_app()
