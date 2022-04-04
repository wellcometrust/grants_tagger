import os

from transformers import BertTokenizerFast, AutoConfig
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import torch

from grants_tagger.bertmesh.model import BertMesh


class MeshDataset(Dataset):
    def __init__(self, X):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.input_ids = self.tokenizer(X, truncation=True, padding=True)["input_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx])


class WellcomeBertMesh:
    def __init__(
        self,
        cutoff_prob=0.1,
        threshold=0.5,
        batch_size=16,
    ):
        self.cutoff_prob = cutoff_prob
        self.threshold = threshold
        self.batch_size = batch_size

    def set_params(self, **params):
        self.__init__(**params)

    def predict(self, X):
        return self.predict_proba(X) > self.threshold

    def predict_proba(self, X):
        dataset = MeshDataset(X)
        data = DataLoader(dataset, self.batch_size)

        self.model.eval()

        Y_pred_proba = []
        with torch.no_grad():
            for inputs in tqdm(data):
                outs = self.model(inputs)

                Y_pred_proba_batch = outs.cpu().numpy()
                Y_pred_proba_batch[Y_pred_proba_batch < self.cutoff_prob] = 0
                Y_pred_proba.append(sp.csr_matrix(Y_pred_proba_batch))

        Y_pred_proba = sp.vstack(Y_pred_proba)
        return Y_pred_proba

    def load(self, model_path):
        self.model = BertMesh.from_pretrained(model_path)
