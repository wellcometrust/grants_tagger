import random
import math
import os

from transformers import AutoModel, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset
from torch import nn
import scipy.sparse as sp
import numpy as np
import torch
import typer


class MultiLabelAttention(torch.nn.Module):
    def __init__(self, D_in, num_labels):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(D_in, num_labels))
        torch.nn.init.uniform(self.A, -0.1, 0.1)

    def forward(self, x):
        attention_weights = torch.nn.functional.softmax(torch.tanh(torch.matmul(x, self.A)), dim=1)
        return torch.matmul(torch.transpose(attention_weights, 2, 1), x)

class BertMesh(torch.nn.Module):
    def __init__(self, pretrained_model, num_labels):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.num_labels = num_labels

        self.bert = AutoModel.from_pretrained(pretrained_model) # 768
        self.multilabel_attention = MultiLabelAttention(768, num_labels) # num_labels, 768
        self.linear_1 = torch.nn.Linear(768, 512) # num_labels, 512
        self.linear_2 = torch.nn.Linear(512, 1) # num_labels, 1

    def forward(self, **inputs):
        cls = self.bert(inputs["input_ids"])[0]
        attention_outs = self.multilabel_attention(cls)
        outs = torch.nn.functional.relu(self.linear_1(attention_outs))
        outs = torch.sigmoid(self.linear_2(outs))
        return torch.flatten(outs)

class MeshDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.Y = sp.load_npz(y_path)
        self.num_labels = self.Y.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        input_ids = self.X[idx,:].tolist()
        labels = np.asarray(self.Y[idx,:].todense()).ravel().tolist()
        return {"input_ids": input_ids, "labels": labels} 

def compute_metrics(pred):
    logits, labels = pred
    predictions = logits > 0.5
    p, r, f1, _ = precision_recall_fscore_support(labels, predictions)
    return {"precision": p, "recall": r, "f1": f1}

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        logits = model(**inputs)
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(
            logits.view(-1, self.model.num_labels),
            labels.float().view(-1, self.model.num_labels))
        return (loss, outputs) if return_outputs else loss

def train_bertmesh(x_path, y_path, model_path, multilabel_attention:bool=False,
        batch_size: int=256, learning_rate: float=1e-4,
        epochs: int=5, pretrained_model="bert-base-uncased"):
    dataset = MeshDataset(x_path, y_path)

    model = BertMesh(pretrained_model, num_labels=dataset.num_labels)

    training_args = TrainingArguments(
        output_dir=model_path,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs
    )
    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == "__main__":
    typer.run(train_bertmesh)


