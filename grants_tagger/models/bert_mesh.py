from transformers import BertTokenizerFast
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
        pretrained_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        num_labels=28761,
        hidden_size=1024,
        multilabel_attention=True,
    ):
        self.cutoff_prob = cutoff_prob
        self.threshold = threshold
        self.batch_size = batch_size
        self.model = BertMesh(
            pretrained_model=pretrained_model,
            num_labels=num_labels,
            hidden_size=hidden_size,
            multilabel_attention=multilabel_attention,
        )

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = torch.load(model_path, map_location=torch.device(device))
        if hasattr(model, "module"):
            # trained with DataParallel
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        self.model.load_state_dict(state_dict)
