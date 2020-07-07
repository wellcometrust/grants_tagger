[![Build Status](https://travis-ci.com/wellcometrust/grants_tagger.svg?branch=master)](https://travis-ci.com/wellcometrust/grants_tagger)
[![codecov](https://codecov.io/gh/wellcometrust/grants_tagger/branch/master/graph/badge.svg)](https://codecov.io/gh/wellcometrust/grants_tagger)
![GitHub](https://img.shields.io/github/license/wellcometrust/grants_tagger)

# Grants tagger

Grants tagger is a machine learning powered tool that
assigns biomedically related tags to grants proposals.
Those tags can be custom to the organisation
or based upon a preexisting ontology like MeSH.

The tool is current being developed internally at the 
Wellcome Trust for internal use but both the models and the
code will be made available in a reusable manner.

This work started as a means to automate the tags of one
funding division within Wellcome but currently it has expanded
into the development and automation of a complete set of tags
that can cover past and future directions for the organisation.

Science tags refer to the custom tags for the Science funding
division. These tags are higly specific to the research Wellcome
funds so it is not advisable to use them.

MeSH tags are subset of tags from the MeSH ontology that aim to
tags grants according to:
- diseases
- themes of research
Those tags are generic enough to be used by other biomedical funders
but note that the selection of tags are highly specific to Wellcome
at the moment.

# Install and use

In the near future you will be able to find official releases
in Github, PyPi and through AWS. Until then the easiest way
to install and use Grants tagger is through a
manual installation. You need to git clone and pip install
the code in your environment of choice.

TODO: Entry point for predicting tags with a model given a
grant proposal

# Develop

## Data

If you work for Wellcome and have access to our AWS account,
you easily download the raw data by typing `make sync_data`.
This will give you access to both the custom science tags
dataset and the MeSH data.

The MeSH data can be downloaded from various places like EPMC.
Grants tagger currently uses a sample provided from the [BioASQ](http://www.bioasq.org)
competition that contains tags for approx 14M publications from PubMed.

## Development environment

To create and setup the development environment
```
make virtualenv
```
This will create a new virtualenv and install requirements for tests
and development. It will also install grants tagger in editable mode.

Grants tagger uses setup.py to track dependencies needed for any
official release. Other dependencies that are being experimentally
used are recorded in requirements.txt. Test dependencies live in 
requirements_text.txt.

The models that power Grants tagger mostly come from sklearn, spacy and
transformers but apart from sklearn all other libraries are hidden behind
[WellcomeML](https://github.com/wellcometrust/WellcomeML). WellcomeML
is a library developed at the Wellcome Trust that contains
easy to use APIs for some of the latest models in the BioNLP space
like SciBERT.

## Commands

- preprocess
- create_label_binarizer
- pretrain
- train

### preprocess

To preprocess either science tags from an Excel file of
annotations or MeSH from the file provided by BioASQ. The
output is a JSONL with one row per grant/publication with
a dict containing text and tags.

```
python -m grants_tagger.preprocess \
    --input data/raw/science_tags_full_version.xlsx \
    --output data/processed/science_grants_tagged.jsonl \
```

### create_label_binarizer

To transform the tags into a 2d matrix with a column
for each tag and 1 indicating the presence of a tag.
This matrix is sparse and especially for MeSH a sparse
representation is used to not blow up memory requirements.

```
python -m grants_tagger.create_label_binarizer \
    --data data/processed/science_grants_tagged.jsonl \
    --label_binarizer models/label_binarizer.pkl
```

### pretrain

Some approaches require or benefit from pretraining a model
in a larger dataset that is unlabeled (with tags). For example,
doc2vec can be pretrained on all grants before used as a feature
extractor (vectorizer).

```
python -m science_tagger.pretrain \
    --data_path data/raw/grants.csv \
    --model_path models/pretrained_doc2vec \
    --model_name doc2vec
```

### train
```
python -m grants_tagger.train \
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
- cnn
- bilstm
- doc2vec-sgd
- doc2vec-tfidf-sgd
- sent2vec-sgd
- sent2vec-tfidf-sgd
- tfidf-adaboost
- tfidf-gboost

Training sklearn models should take less than 1 hour. Time greatly depends
on the embedding with TFIDF being very fast and any BERT embedding requiring
more than 30 minutes.

Training spacy takes more time on average e.g. approx 10 minutes when not using a
pre-trained model.

Training any BERT model end to end (fine tuning) would be prohibitive locally,
unless you have a GPU and even in that case it takes around approx 1hour.


## Package

Packaging might change in the future but at the 
moment MANIFEST.in is being used to instruct the
build process which models to package. Do not forget
to include the label binarizer along with the model

To package a model run `make build`

## Reproducing results

As we are developing the models responsible for
tagging the grants, we want to ensure any intermediate
results are recorded and reproducible. To achieve that
we use configuration files that record all neccesary
parameters.

We record results in docs/results.md. You can reproduce
any result by running `./scripts/run_config.sh VERSION`
and substituting for the version you want to reproduce

# Test

`make test`
