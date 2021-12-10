from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import typer

from grants_tagger.bertmesh.data import MeshDataset
from grants_tagger.bertmesh.model import BertMesh, MultiLabelAttention


def train_bertmesh(
    x_path,
    y_path,
    model_path,
    multilabel_attention: bool = False,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    epochs: int = 5,
    pretrained_model="bert-base-uncased",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MeshDataset(x_path, y_path)

    model = BertMesh(
        pretrained_model,
        num_labels=dataset.num_labels,
        multilabel_attention=multilabel_attention,
    )
    model = torch.nn.DataParallel(model)
    model.to(device)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=16
    )  # ignored at the moment

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        batches = tqdm(
            train_loader, desc="Epoch {:1d}".format(epoch), leave=False, disable=False
        )
        for data in batches:
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            batches.set_postfix(
                {"training_loss": "{:.3f}".format(loss.item() / len(batch))}
            )

    torch.save(model, model_path)


if __name__ == "__main__":
    typer.run(train_bertmesh)
