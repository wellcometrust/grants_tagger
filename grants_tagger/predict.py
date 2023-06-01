"""
Predict function for disease part of mesh that optionally
exposes probabilities and that you can set the threshold
for making a prediction
"""
from pathlib import Path
import typer

from grants_tagger.models.utils import format_predictions
from typing import Optional
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer

import logging

logger = logging.getLogger(__name__)


def predict_tags(
    X,
    model_path,
    batch_size: int = 1,
    probabilities=False,
    threshold=0.5,
    parameters=None,
    config=None,
):
    """
    X: list or numpy array of texts
    model_path: path to trained model
    probabilities: bool, default False. When true probabilities are returned along with tags
    threshold: float, default 0.5. Probability threshold to be used to assign tags.
    parameters: any params required upon model creation
    config: Path to config file
    """

    tokenizer = AutoTokenizer.from_pretrained("Wellcome/WellcomeBertMesh")
    model = AutoModel.from_pretrained(
        "Wellcome/WellcomeBertMesh", trust_remote_code=True
    )

    if isinstance(X, str):
        X = [X]

    inputs = tokenizer(X, padding="max_length", max_length=512, truncation=True)

    # Do prediction in batches
    num_batches = len(X) // batch_size + 1
    labels = []

    for idx in tqdm(range(num_batches)):
        batch = {
            k: v[idx * batch_size : (idx + 1) * batch_size] for k, v in inputs.items()
        }

        if batch["input_ids"] == []:
            continue

        batch_labels = model(**batch, return_labels=True)
        labels.extend(batch_labels)

    return labels


predict_app = typer.Typer()


@predict_app.command()
def predict_cli(
    text: str,
    model_path: Path,
    batch_size: Optional[int] = typer.Option(1),
    probabilities: Optional[bool] = typer.Option(False),
    threshold: Optional[float] = typer.Option(0.5),
):
    labels = predict_tags([text], model_path, batch_size, probabilities, threshold)
    print(labels)


if __name__ == "__main__":
    predict_app()
