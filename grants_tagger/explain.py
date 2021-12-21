import pickle
import logging
import typer

from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)
try:
    import shap
except ModuleNotFoundError as e:
    logger.warning("shap missing. explain will not work")
    logger.debug(e)
from grants_tagger.models.mesh_cnn import MeshCNN


def explain(
    approach,
    data_path,
    model_path,
    label_binarizer_path,
    explanation_path,
    label,
    global_explanations=True,
):
    if approach == "mesh-cnn":
        mesh_cnn = MeshCNN(cutoff_prob=None)
        mesh_cnn.load(model_path)
        tokenizer = mesh_cnn.vectorizer.tokenizer

        with open(label_binarizer_path, "rb") as f:
            label_binarizer = pickle.loads(f.read())

        with open(data_path) as f:
            texts = f.readlines()

        if len(texts) > 50:
            print("Data contains >50 examples. Explanations might take a while...")

        masker = shap.maskers.Text(tokenizer, mask_token="")
        explainer = shap.Explainer(
            mesh_cnn.predict_proba, masker, output_names=label_binarizer.classes_
        )
        shap_values = explainer(texts)

        if global_explanations:
            plt = shap.plots.bar(shap_values[:, :, label], show=False)
            plt.savefig(explanation_path, format="svg")
        else:
            html = shap.plots.text(shap_values[0, :, label], display=False)
            with open(explanation_path, "w") as f:
                f.write(html)
    else:
        raise NotImplementedError


explain_app = typer.Typer()


@explain_app.command()
def explain_cli(
    texts_path: Path = typer.Argument(
        ..., help="path to txt file with one text in every line"
    ),
    label: str = typer.Argument(
        ..., help="label to explain with local or global explanations"
    ),
    approach: str = typer.Argument(..., help="model approach e.g. mesh-cnn"),
    model_path: Path = typer.Argument(..., help="path to model to explain"),
    label_binarizer_path: Path = typer.Argument(
        ..., help="path to label binarizer associated with mode"
    ),
    explanations_path: Path = typer.Argument(
        ..., help="path to save explanations html"
    ),
    global_explanations: bool = typer.Option(
        True, help="flag on whether global or local explanations should be produced"
    ),
):

    explain(
        approach,
        texts_path,
        model_path,
        label_binarizer_path,
        explanations_path,
        label,
        global_explanations,
    )


if __name__ == "__main__":
    explain_app()
