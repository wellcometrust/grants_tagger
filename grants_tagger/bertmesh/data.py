from torch.utils.data import Dataset
import scipy.sparse as sp
import numpy as np
import torch


class MeshDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.Y = sp.load_npz(y_path)
        self.num_labels = self.Y.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.X[idx, :].tolist())
        labels = torch.tensor(np.asarray(self.Y[idx, :].todense()).ravel().tolist())
        return input_ids, labels
