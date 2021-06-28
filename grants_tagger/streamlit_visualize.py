import streamlit as st
import pandas as pd

from grants_tagger import predict_tags

threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5)
text = st.text_area('Grant abstract', 'The cell is...', height=300)

models = {
    "disease_mesh_cnn-2021.03.1": {
        "model_path": "models/disease_mesh_cnn-2021.03.1/",
        "label_binarizer_path": "models/disease_mesh_label_binarizer.pkl",
        "approach": "mesh-cnn"
    },
    "disease_mesh_cnn-2020.09.0": {
        "model_path": "models/disease_mesh_cnn-2020.09.0/",
        "label_binarizer_path": "models/disease_mesh_label_binarizer.pkl",
        "approach": "mesh-cnn"
    },
    "tfidf-svm-2020.05.2": {
        "model_path": "models/tfidf-svm-2020.05.2.pkl",
        "label_binarizer_path": "models/label_binarizer.pkl",
        "approach": "tfidf-svm"
    },
    "scibert-2020.05.5": {
        "model_path": "models/scibert-2020.05.5/",
        "label_binarizer_path": "models/label_binarizer.pkl",
        "approach": "scibert"
    }
}

model_option = st.sidebar.selectbox("Model", options=list(models.keys()))
model = models[model_option]

probabilities = st.sidebar.checkbox("Display probabilities")

with st.spinner('Calculating tags...'):
    tags = predict_tags([text], model["model_path"], model["label_binarizer_path"],
        model["approach"], probabilities=probabilities, threshold=threshold)
    tags = tags[0]
st.success("Done!")

if probabilities:
    tags = [{"Tag": tag, "Prob": prob} for tag, prob in tags.items() if prob > threshold]
    st.table(pd.DataFrame(tags))
else:
    for tag in tags:
        st.button(tag)
