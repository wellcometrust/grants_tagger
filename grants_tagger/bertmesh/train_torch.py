import random
import math
import os

from transformers import AutoModel
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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

    def forward(self, inputs):
        hidden_states = self.bert(input_ids=inputs)[0]
        attention_outs = self.multilabel_attention(hidden_states)
        outs = torch.nn.functional.relu(self.linear_1(attention_outs))
        outs = torch.sigmoid(self.linear_2(outs))
        return torch.flatten(outs, start_dim=1)

class MeshDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.Y = sp.load_npz(y_path)
        self.num_labels = self.Y.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.X[idx,:].tolist())
        labels = torch.tensor(np.asarray(self.Y[idx,:].todense()).ravel().tolist())
        return input_ids, labels

def train_bertmesh(x_path, y_path, model_path, multilabel_attention:bool=False,
        batch_size: int=256, learning_rate: float=1e-4,
        epochs: int=5, pretrained_model="bert-base-uncased"):
    dataset = MeshDataset(x_path, y_path)

    model = BertMesh(pretrained_model, num_labels=dataset.num_labels)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16) # ignored at the moment

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for data in tqdm(train_loader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    typer.run(train_bertmesh)


