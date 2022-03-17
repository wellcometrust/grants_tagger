import pickle
import json

import streamlit as st

import seaborn as sns
import pandas as pd

try:
    import shap

    SHAP_IMPORTED = True
except ImportError:
    print(
        "To get explanations for the predictions install shap "
        "with pip install git+https://github.com/nsorros/shap.git@dev"
    )
    SHAP_IMPORTED = False

from grants_tagger.predict import predict_tags, format_predictions
from grants_tagger.models.create_model import load_model

threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=1.0, value=0.2)

models = {
    "disease_mesh_cnn-2021.03.1": {
        "model_path": "models/disease_mesh_cnn-2021.03.1/",
        "label_binarizer_path": "models/disease_mesh_label_binarizer.pkl",
        "approach": "mesh-cnn",
        "enabled": False,
    },
    "tfidf-svm-2020.05.2": {
        "model_path": "models/tfidf-svm-2020.05.2.pkl",
        "label_binarizer_path": "models/label_binarizer.pkl",
        "approach": "tfidf-svm",
        "enabled": False,
    },
    "scibert-2020.05.5": {
        "model_path": "models/scibert-2020.05.5/",
        "label_binarizer_path": "models/label_binarizer.pkl",
        "approach": "scibert",
        "enabled": False,
    },
    "mesh-xlinear-0.2.3": {
        "model_path": "models/xlinear-0.2.3/model",
        "full_report_path": "results/mesh_xlinear_full_report.json",
        "label_binarizer_path": "models/xlinear-0.2.3/label_binarizer-0.2.3.pkl",
        "approach": "mesh-xlinear",
        "enabled": True,
    },
}

model_option = st.sidebar.selectbox(
    "Model",
    options=[model_name for model_name, params in models.items() if params["enabled"]],
)
full_report = {}
model_info = models[model_option]
DEFAULT_TEXT = "The cell is..."

text = st.text_area("Grant abstract", DEFAULT_TEXT, height=300)


@st.cache(suppress_st_warning=True)
def load_binarizer_app():
    with open(models[model_option]["label_binarizer_path"], "rb") as f:
        binarizer = pickle.loads(f.read())
    return binarizer


@st.cache(suppress_st_warning=True)
def load_model_app(model_option):
    # Caches model loading
    return load_model(
        approach=models[model_option]["approach"],
        model_path=models[model_option]["model_path"],
    )


label_binarizer = load_binarizer_app()
model = load_model_app(model_option)

probabilities = st.sidebar.checkbox("Display probabilities")

with st.spinner("Calculating tags..."):
    if (
        text != DEFAULT_TEXT and len(text) > 5
    ):  # Character limit to catch spurious texts
        Y_pred_proba = model.predict_proba([text])

        tags = format_predictions(
            Y_pred_proba,
            label_binarizer,
            threshold=threshold,
            probabilities=probabilities,
        )

        tags = tags[0]
        st.success("Done!")
    else:
        tags = {}

if probabilities:
    tag_probs = [
        {"Tag": tag, "Prob": prob} for tag, prob in tags.items() if prob > threshold
    ]
    st.table(pd.DataFrame(tag_probs))
    tags = [tag_prob["Tag"] for tag_prob in tag_probs]
else:
    for tag in tags:
        st.button(tag)

if SHAP_IMPORTED:
    from grants_tagger.models.mesh_cnn import MeshCNN

    if model_info["approach"] == "mesh-cnn":
        mesh_cnn = MeshCNN(threshold=threshold)
        mesh_cnn.load(model_info["model_path"])
        tokenizer = mesh_cnn.vectorizer.tokenizer

        with open(model_info["label_binarizer_path"], "rb") as f:
            label_binarizer = pickle.loads(f.read())

        with st.spinner("Calculating explanation..."):
            masker = shap.maskers.Text(tokenizer, mask_token="")
            explainer = shap.Explainer(
                mesh_cnn.predict_proba, masker, output_names=label_binarizer.classes_
            )
            shap_values = explainer([text])

        for tag in tags:
            st.write(tag)
            tag_index = list(label_binarizer.classes_).index(tag)

            html = shap.plots.text(shap_values[0, :, tag_index], display=False)
            st.components.v1.html(html, height=300, scrolling=True)

if model_info.get("full_report_path"):
    with open(model_info["full_report_path"], "r") as f:
        # Loads a report and "linearises" into a list (a report has tags as keys)
        full_report = [{**{"tag": tag}, **stats} for tag, stats in json.load(f).items()]


if full_report:
    n_top = 40
    df = pd.DataFrame(full_report)
    averages = ["micro avg", "macro avg", "samples avg", "weighted avg"]

    df.set_index("tag", inplace=True)
    df.rename(
        {"f1-score": "score", "support": "number of examples"}, axis=1, inplace=True
    )
    df = df[df["number of examples"] > 50]
    columns_of_interest = ["score", "number of examples"]

    micro_averages = df.loc["micro avg"]

    # Drop averages for sorting etc
    df.drop(averages, inplace=True)

    top_tags = df.sort_values(by="score")

    top_tags["precision"] = top_tags["precision"].apply(lambda x: f"{100 * x:2.0f}%")
    top_tags["recall"] = top_tags["recall"].apply(lambda x: f"{100 * x:2.0f}%")
    top_tags["score"] = top_tags["score"].apply(lambda x: f"{100 * x:2.0f}%")
    top_tags["number of examples"] = top_tags["number of examples"].apply(
        lambda x: f"{x:.0f}"
    )

    with st.expander("Top performing tags"):
        st.table(top_tags[-n_top:][::-1][columns_of_interest])

    with st.expander("Worst performing tags"):
        st.table(top_tags[:n_top][columns_of_interest])

    with st.expander("Most common tags"):
        # Have to convert number of examples to integer to sort properly
        st.table(
            top_tags.sort_values(
                by="number of examples", key=lambda col: col.astype(int)
            )[-n_top:][::-1][columns_of_interest]
        )

        # Eliminates everyone with "too many" and "too few" examples
        df_sans_outliers = df[
            (df["number of examples"] > 50) & (df["number of examples"] < 5000)
        ]

        p = sns.regplot(
            x=df_sans_outliers["number of examples"], y=df_sans_outliers["score"]
        )
        fig = p.get_figure()
        st.pyplot(fig)

    with st.expander("Search tag performance"):
        option = st.selectbox(
            "Select a MeSH tag to see how it compares to the overall performance",
            df.index,
            index=100,
        )
        precision = df.loc[option]["precision"]
        recall = df.loc[option]["recall"]
        score = df.loc[option]["score"]
        examples = df.loc[option]["number of examples"]

        delta_precision = precision - micro_averages["precision"]
        delta_recall = recall - micro_averages["recall"]
        delta_score = score - micro_averages["score"]
        # Below, for examples, I need median otherwise outliers will skew
        delta_examples = examples - df["number of examples"].median()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Precision",
            f"{100 * precision:.0f}%",
            delta=f"{100 * delta_precision:.0f}%",
        )
        col2.metric(
            "Recall", f"{100 * recall:.0f}%", delta=f"{100 * delta_recall:.0f}%"
        )
        col3.metric("Score", f"{100 * score:.0f}%", delta=f"{100 * delta_score:.0f}%")
        col4.metric("Examples", f"{examples:.0f}", delta=f"{delta_examples:.0f}")
