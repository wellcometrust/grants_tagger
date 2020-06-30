## Project

Collaborating with Science to automate the tagging of Science grants. In March 2019 Science manually tagged the Active portfolio with Science Specific and Disease (ICD-11) tags. These Science Specific tags will now be used to train an algorithm to enable the fitting of tags retrospectively to the entire Science portfolio (including historical grants) as well as to automate tagging of future grants. Tagging the Science portfolio will allow the Division to have a greater understanding of what they fund and how this has changed over time.

## Data

1. List of Science Specific tags applied to the Active Science portfolio (March 2019), with corresponding Grant ID
2. Data Warehouse - Application and Grant Details table. Using Grant ID to link to fields that provide additional detail on the grants e.g. title, synopses, lay summary, research question

To download the data run `make sync_data_from_s3`

## Approach

Multi label text classification with Scikit-Learn and Spacy

## Setup

### Automatic
The easiest to setup is
`make setup`

### Manual
```
python -c "import nltk;nltk.download('wordnet')"
python -m spacy download en_core_web_sm
python -m spacy download en_trf_bertbaseuncased_lg
```

## Reproducing results

### Preprocessing
```
python src/preprocess.py \
    --input data/raw/science_tags_full_version.xlsx \
    --output data/processed/science_grants_tagged.jsonl \
```

```
python src/create_label_binarizer.py \
    --data data/processed/science_grants_tagged.jsonl \
    --label_binarizer models/label_binarizer.pkl
```

### Pretraining

Some approaches require or benefit from pretraining a model
in a larger dataset that is unlabeled (with tags). For example,
doc2vec can be pretrained on all grants before used as a feature
extractor (vectorizer).

```
python -m science_tagger.pretrain \
    --data_path science_tagger/data/raw/grants.csv \
    --model_path science_tagger/models/pretrained_doc2vec \
    --model_name doc2vec
```

### Training
```
python src/train.py \
    --data data/processed/science_grants_tagged.jsonl \
    --model models/model.pkl \
    --label_binarizer models/label_binarizer.pkl
    --approach tfidf-svm
```

The approaches to choose from:

- tfidf-svm
- spacy-textclassifier
- classifierchain-tfidf-svm
- labelpowerset-tfidf-svm
- binaryrelevance-tfidf-svm
- binaryrelevance-tfidf-knn
- tfidf-bert
- tfidf-scibert
- bert
- scibert

TODO: Add a table with results

Training sklearn models should take less than 1 hour. Time greatly depends
on the embedding with TFIDF being very fast and any BERT embedding requiring
more than 30 minutes.

Training spacy takes more time on average e.g. ~10 minutes when not using a
pre-trained model.

### Pre-trained models
Training any BERT model end to end (fine tuning) would be prohibitive locally,
unless you have a GPU and even in that case it takes around ~1hour.

You can download pretrained models by running `make sync_models_from_s3`. Note
that these models are ~400MB each so downloading all of them could take from 4GB
to 10GB of space. Due to the size of this models you need to select which model
to download with the flag INCLUDE. The name of the approach or the version number
is a good way to download a specific model. e.g. `make sync_models_from_s3
INCLUDE=tfidf-svm`

### Metrics
```
python src/calculate_metrics.py \
   --data data/processed/science_grants_tagged.jsonl \
   --model models/model.pkl \
   --label_binarizer models/label_binarizer.pkl
```
This script prints useful metrics for the model and the data
and plots and saves figures in the figures folder

### Experiments

if you want to find the best params for a given approach
```
python src/optimise_params.py \
   --data data/processed/science_grants_tagged.jsonl \
   --label_binarizer models/label_binarizer.pkl \
   --approach tfidf-svm
```

optimising params takes a lot of time depending on the amount of grid
search space and model.

similarly if you want to reproduce the results and experiments for a given
dataset
```
python src/run_experiments.py \
   --data data/processed/science_grants_tagged.jsonl \
   --label_binarizer models/label_binarizer.pkl \
   --experiment test
```

## Prodigy

To install and use prodigy you need to run
`make install_prodigy`

Note that this will reinstall spacy as prodigy requires a
different version. At the moment this breaks other parts
so if you want to go back and run train or other functions
you should run `make virtualenv` again.

To use vanilla prodigy with a spacy model
```
prodigy textcat.teach \
    my_dataset \
    en_core_web_sm \
    data/processed/science_tags_teach.jsonl \
    --label '10: Fungi'
```

To use prodigy with a custom model.
Note that the only custom model that
will work with current prodigy version
is the sklearn tfidf one. The others are
using spacy 2.1+ which is not currently supported.
Will be supported in version 1.9 of Prodigy

```
prodigy textcat.teach-custom-model \
    my_dataset data/processed/science_tags_teach.jsonl \
    models/model.pkl models/label_binarizer.pkl \
    --label '10: Fungi' \
    -F src/recipe.py \
    --goal 100
```


