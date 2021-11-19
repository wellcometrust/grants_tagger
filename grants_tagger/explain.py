import pickle
import logging

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
        mesh_cnn = MeshCNN()
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
