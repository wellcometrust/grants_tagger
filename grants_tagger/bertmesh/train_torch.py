from typing import Optional
import yaml
import json

from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
import torch
import typer
import wandb

from grants_tagger.bertmesh.data import MeshDataset
from grants_tagger.bertmesh.model import BertMesh, MultiLabelAttention

with open("params.yaml") as f:
    params = yaml.safe_load(f)
wandb.init(project="bertmesh", config=params["train"])


def train_bertmesh(
    x_path,
    y_path,
    model_path,
    hidden_size: int = 512,
    dropout: float = 0,
    multilabel_attention: bool = False,
    batch_size: int = 64,
    learning_rate: float = 1e-5,
    epochs: int = 5,
    pretrained_model="bert-base-uncased",
    warmup_steps: float = 1000,
    clip_norm: Optional[float] = None,
    train_metrics_path: Optional[str] = None,
    val_x_path: Optional[str] = None,
    val_y_path: Optional[str] = None,
    log_interval: int = 100,
    dry_run: bool = False,
):
    wandb.config.update({"multilabel_attention": multilabel_attention})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MeshDataset(x_path, y_path)
    data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if val_x_path and val_y_path:
        val_dataset = MeshDataset(val_x_path, val_y_path)
        val_data = DataLoader(val_dataset, batch_size=batch_size)

    model = BertMesh(
        pretrained_model,
        num_labels=dataset.num_labels,
        hidden_size=hidden_size,
        dropout=dropout,
        multilabel_attention=multilabel_attention,
    )
    model = torch.nn.DataParallel(model)
    model.to(device)

    wandb.watch(model, log_freq=100)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(data)
    )

    best_val_loss = 1
    metrics = []
    running_loss = 0
    for epoch in range(epochs):
        batches = tqdm(data, desc=f"Epoch {epoch+1:2d}/{epochs:2d}")
        for step, batch in enumerate(batches):
            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            if clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=clip_norm
                )

            scheduler.step()

            running_loss += loss.item()

            if step % log_interval == 0:
                batch_metrics = {
                    "loss": round(running_loss / log_interval, 5),
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                batches.set_postfix(batch_metrics)
                wandb.log(batch_metrics)
                metrics.append(batch_metrics)

            if dry_run:
                break

        epoch_model_path = model_path.replace(".pt", f"-epoch{epoch+1}.pt")
        torch.save(model, epoch_model_path)

        if val_x_path and val_y_path:
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for val_batch in val_data:
                    val_inputs, val_labels = val_batch[0].to(device), val_batch[1].to(
                        device
                    )

                    val_outputs = model(val_inputs)
                    val_loss_batch = criterion(val_outputs, val_labels.float())
                    val_loss += val_loss_batch.item()

                    if dry_run:
                        break

            val_loss /= len(val_dataset)

            if val_loss < best_val_loss:
                best_model_path = model_path.replace(".pt", "-best.pt")
                torch.save(model, best_model_path)

            wandb.log({"val_loss": val_loss})

            model.train()

    torch.save(model, model_path)

    if train_metrics_path:
        with open(train_metrics_path, "w") as f:
            f.write(json.dumps({"metrics": metrics}))


if __name__ == "__main__":
    typer.run(train_bertmesh)
