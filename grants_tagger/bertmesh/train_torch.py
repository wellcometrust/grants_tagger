from typing import Optional
from datetime import datetime
import yaml
import time
import json

from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoConfig
from tqdm import tqdm
import transformers
import torch
import typer
import wandb

from grants_tagger.bertmesh.data import MeshDataset
from grants_tagger.bertmesh.model import BertMesh, MultiLabelAttention
from grants_tagger.utils import get_ec2_instance_type, load_pickle


def train_bertmesh(
    x_path,
    y_path,
    model_path,
    label_binarizer_path,
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
    train_info: str = typer.Option(None, help="path to train times and instance"),
    experiment_name=datetime.now().strftime("%d/%m/%Y"),
    accelerate: bool = False,
    dry_run: bool = False,
):
    start = time.time()
    if not dry_run:
        config = {
            "hidden_size": hidden_size,
            "dropout": dropout,
            "multilabel_attention": multilabel_attention,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "pretrained_model": pretrained_model,
            "warmup_steps": warmup_steps,
            "clip_norm": clip_norm,
        }
        wandb.init(
            project="bertmesh", group=experiment_name, job_type="train", config=config
        )

    if accelerate:
        accelerator = Accelerator()
        device = accelerator.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MeshDataset(x_path, y_path)
    data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if val_x_path and val_y_path:
        val_dataset = MeshDataset(val_x_path, val_y_path)
        val_data = DataLoader(val_dataset, batch_size=batch_size)

    label_binarizer = load_pickle(label_binarizer_path)
    id2label = {i: label for i, label in enumerate(label_binarizer.classes_)}

    config = AutoConfig.from_pretrained(pretrained_model)
    config.update(
        {
            "pretrained_model": pretrained_model,
            "num_labels": dataset.num_labels,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "multilabel_attention": multilabel_attention,
            "id2label": id2label,
        }
    )
    model = BertMesh(config)
    if not accelerate:
        model = torch.nn.DataParallel(model)
    model.to(device)

    if not dry_run:
        wandb.watch(model, log_freq=log_interval)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(data)
    )

    if accelerate:
        model, optimizer, data = accelerator.prepare(model, optimizer, data)
        if val_x_path and val_y_path:
            val_data = accelerator.prepare(val_data)

    best_val_loss = 1
    metrics = []
    running_loss = 0
    global_step = 0
    for epoch in range(epochs):
        batches = tqdm(data, desc=f"Epoch {epoch+1:2d}/{epochs:2d}")
        for step, batch in enumerate(batches):
            if accelerate:
                inputs, labels = batch
            else:
                inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            if accelerate:
                accelerator.backward(loss)
            else:
                loss.backward()
            optimizer.step()

            if clip_norm:
                if accelerate:
                    accelerator.clip_grad_norm_(
                        parameters=model.parameters(), max_norm=clip_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=model.parameters(), max_norm=clip_norm
                    )

            scheduler.step()

            running_loss += loss.item()

            if step % log_interval == 0:
                batch_metrics = {
                    "loss": round(running_loss / log_interval, 5),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                batches.set_postfix(batch_metrics)
                if not dry_run:
                    wandb.log(batch_metrics, step=global_step)
                metrics.append(batch_metrics)

                running_loss = 0

            global_step += 1

            if dry_run:
                break

        epoch_model_path = f"{model_path}/epoch-{epoch+1}/"
        if accelerate:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
        else:
            unwrapped_model = model.module

        unwrapped_model.save_pretrained(epoch_model_path)

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
                best_model_path = f"{model_path}/best/"

                if accelerate:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                else:
                    unwrapped_model = model.module

                unwrapped_model.save_pretrained(best_model_path)

            if not dry_run:
                wandb.log({"val_loss": val_loss})

            model.train()

        if dry_run:
            break

    if accelerate:
        accelerator.wait_for_everyone()
        model = accelerator.unwrap_model(model)
    else:
        model = model.module

    model.save_pretrained(model_path)

    if train_metrics_path:
        with open(train_metrics_path, "w") as f:
            f.write(json.dumps({"metrics": metrics}))

    duration = time.time() - start
    instance = get_ec2_instance_type()
    print(f"Took {duration:.2f} to train")
    if train_info:
        with open(train_info, "w") as f:
            json.dump({"duration": duration, "ec2_instance": instance}, f)


if __name__ == "__main__":
    typer.run(train_bertmesh)
