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

Grants tagger comes with a nice CLI with the following commands

- preprocess     # preprocess data to use for training
- train          # trains a new model
- evaluate       # evaluate performance of pretrained model
- predict        # predict tags given a grant abstract using a pretrained model
- tag            # tag grants using a pretrained model
- tune           # tune params and threshold
- pretrain       # pretrains embeddings or language model using unlabeled data
- [download]     # download trained models and data from EPMC
- [explain]      # importance of feature be it words or tfidf numbers

### Preprocess

Preprocess creates a JSONL datafile with `text`, `tags` and `meta` as keys.
Text and tags are used for training whereas meta can be used either for annotation
through prodigy or to analyse predictions. Each dataset needs to implement its own
preprocessing so the current preprocess works with the wellcome-science dataset and
the bioasq-mesh one. If you want to use a different dataset see section on bringing
your own data under development.


*wellcome-science*
```
Usage: grants_tagger preprocess wellcome-science [OPTIONS] [INPUT_PATH]
                                                 [OUTPUT_PATH]

Arguments:
  [INPUT_PATH]   path to raw Excel file with tagged or untagged grant data
  [OUTPUT_PATH]  path to JSONL output file that will be generated

Options:
  --text-cols TEXT  comma delimited column names to concatenate to text
  --meta-cols TEXT  comma delimited column names to include in the meta
  --config PATH     path to config file that defines the arguments
  --help            Show this message and exit.
```

*bioasq-mesh*
```
Usage: grants_tagger preprocess bioasq-mesh [OPTIONS] [INPUT_PATH]
                                            [OUTPUT_PATH]

Arguments:
  [INPUT_PATH]   path to BioASQ JSON data
  [OUTPUT_PATH]  path to output JSONL data

Options:
  --mesh-metadata-path TEXT  path to xml file containing MeSH taxonomy
  --filter-tags TEXT         filter mesh subbranch like disease
  --test-split FLOAT         split percentage for test data. if None no split.
  --config PATH              path to config files that defines arguments
  --help                     Show this message and exit.
```

### Train

Train acts as the entry point command for training all models. You can control
which model you want to use with an `--approach` flag. This is convenient but 
also not ideal as different models require different params and considerations.
This will change in the future and the approach will be a subcommand so each
model will trigger a different train and have different params.

```
Usage: grants_tagger train [OPTIONS] [DATA_PATH] [LABEL_BINARIZER_PATH]
                           [MODEL_PATH]

Arguments:
  [DATA_PATH]             path to processed JSON data to be used for training
  [LABEL_BINARIZER_PATH]  path to label binarizer
  [MODEL_PATH]            path to output model.pkl or dir to save model

Options:
  --approach TEXT                 tfidf-svm, scibert, cnn, ...  [default:
                                  tfidf-svm]

  --parameters TEXT               model params in sklearn format e.g.
                                  {'svm__kernel: linear'}

  --test-data-path PATH           path to processed JSON test data
  --threshold FLOAT               threshold to assign a tag
  --data-format TEXT              format that will be used when loading the
                                  data. One of list,generator  [default: list]

  --test-size FLOAT               float or int indicating either percentage or
                                  absolute number of test examples  [default:
                                  0.25]

  --sparse-labels / --no-sparse-labels
                                  flat about whether labels should be sparse
                                  when binarized  [default: False]

  --cache-path PATH               path to cache data transformartions
  --config PATH
  --help                          Show this message and exit.
```

Label binarizer converts tags to a Y matrix with one column per tag. This is
neccesary as this is a multilabel problem so multiple tags may be assigned to
one grant.

Parameters is a JSON like dump (essentially a stringified dict) with all params
that the model expects in an sklearn fashion.

### Evaluate

Evaluate enables evaluation of the performance of various approaches including
human performance and other systems like MTI, SciSpacy and soon Dimensions. As
such evaluate has the followin subcommands

#### Model

Model is the generic entrypoint for model evaluation. Similar to train approach
controls which model will be evaluated. Approach which is a positional argument
in this command controls which model will be evaluated. Since the data in train
are sometimes split inside train, the same splitting is performed in evaluate.
Evaluate only supports some models, in particular those that have made it to
production. These are: `tfidf-svm`, `scibert`, `science-ensemble`, `mesh-tfidf-svm`
and `mesh-cnn`. 

```
Usage: grants_tagger evaluate model [OPTIONS] APPROACH MODEL_PATH DATA_PATH
                                    LABEL_BINARIZER_PATH

Arguments:
  APPROACH              model approach e.g.mesh-cnn  [required]
  MODEL_PATH            comma separated paths to pretrained models  [required]
  DATA_PATH             path to data that was used for training  [required]
  LABEL_BINARIZER_PATH  path to label binarize  [required]

Options:
  --threshold TEXT     threshold or comma separated thresholds used to assign
                       tags  [default: 0.5]

  --results-path TEXT  path to save results  [default: results.json]
  --config PATH        path to config file that defines arguments
  --help               Show this message and exit.
```

#### Human

Human evaluates human accuracy on the task in the case human annotations exists.
Currently only works for wellcome-science that has two annotations per grant but
it can be extended to support a more general format in the future.

```
Usage: grants_tagger evaluate human [OPTIONS] DATA_PATH LABEL_BINARIZER_PATH

Arguments:
  DATA_PATH             [required]
  LABEL_BINARIZER_PATH  [required]

Options:
  --help  Show this message and exit.
```

#### MTI

MTI is the automatic mesh indexer from NLM. To get MTI annotations you need to
submit grants for tagging through an email service and get the results which
you can use here for evaluation.

```
Usage: grants_tagger evaluate mti [OPTIONS] DATA_PATH LABEL_BINARIZER_PATH
                                  MTI_OUTPUT_PATH

Arguments:
  DATA_PATH             path to sample JSONL mesh data  [required]
  LABEL_BINARIZER_PATH  path to pickled mesh label binarizer  [required]
  MTI_OUTPUT_PATH       path to mti output txt  [required]

Options:
  --help  Show this message and exit.
```

#### SciSpacy

SciSpacy is a tool developed by AllenAI that identifies mainly entities in 
biomedical text using Spacy with almost state of the art accuracy. These
entities are linked to various ontologies some of which is MeSH. Even though
NER combined with Entity linking and Multilabel text classification are not 
the same problem, this command evaluate how SciSpacy would perform on our
problem.

```
Usage: grants_tagger evaluate scispacy [OPTIONS] DATA_PATH
                                       LABEL_BINARIZER_PATH MESH_METADATA_PATH

Arguments:
  DATA_PATH             JSONL of mesh data that contains text, tags and meta
                        per line  [required]

  LABEL_BINARIZER_PATH  label binarizer that transforms mesh names to
                        binarized format  [required]

  MESH_METADATA_PATH    csv that contains metadata about mesh such as UI, Name
                        etc  [required]


Options:
  --help  Show this message and exit.
```

### Predict

Predict assigns tags on a given abstract text that you can pass as argument.
It is not meant to be used for tagging multiple grants, tag command is reserved
for that. Similar to evaluate and train an approach is needed to know which
model will be used.

```
Usage: grants_tagger predict [OPTIONS] TEXT MODEL_PATH LABEL_BINARIZER_PATH
                             APPROACH

Arguments:
  TEXT                  [required]
  MODEL_PATH            [required]
  LABEL_BINARIZER_PATH  [required]
  APPROACH              [required]

Options:
  --probabilities / --no-probabilities
                                  [default: False]
  --threshold FLOAT               [default: 0.5]
  --help                          Show this message and exit.
```

### Tag

Tag is the main command of the tool as it allows you to tag with a pretrained
model. This command currently works on a CSV with "Grant ID, Reference, Grant No."
as fieldnames but will soon change to accomodate a more general format and
to be used in production at Wellcome for tagging. 

```
Usage: grants_tagger tag [OPTIONS] GRANTS_PATH TAGGED_GRANTS_PATH MODEL_PATH
                         LABEL_BINARIZER_PATH

Arguments:
  GRANTS_PATH           path to grants csv  [required]
  TAGGED_GRANTS_PATH    path to output csv  [required]
  MODEL_PATH            path to model  [required]
  LABEL_BINARIZER_PATH  label binarizer for Y  [required]

Options:
  --threshold FLOAT  threshold upon which to assign tag  [default: 0.5]
  --help             Show this message and exit.
```

### Tune

Tune optimises params of choice or the threshold that is used to
assign tags. Parameter optimisation makes use of sklearn GridSearch
and currently works on a predefined set of approaches and params. At a later
point it will accept the parameter search space and will perform the optimisation
either locally or in SageMaker. Threshold optimises the individual thresholds
that assign tags. Each tag can have an individual threshold that together maximises
f1 score.


*params*
```
Usage: grants_tagger tune params [OPTIONS] DATA_PATH LABEL_BINARIZER_PATH
                                 APPROACH

Arguments:
  DATA_PATH             [required]
  LABEL_BINARIZER_PATH  [required]
  APPROACH              [required]

Options:
  --params TEXT
  --help         Show this message and exit.
```

*threshold*
```
Usage: grants_tagger tune threshold [OPTIONS] APPROACH DATA_PATH MODEL_PATH
                                    LABEL_BINARIZER_PATH THRESHOLDS_PATH

Arguments:
  APPROACH              modelling approach e.g. mesh-cnn  [required]
  DATA_PATH             path to data in jsonl to train and test model
                        [required]

  MODEL_PATH            path to data in jsonl to train and test model
                        [required]

  LABEL_BINARIZER_PATH  path to label binarizer  [required]
  THRESHOLDS_PATH       path to save threshold values  [required]

Options:
  --sample-size INTEGER    sample size of text data to use for tuning
  --nb-thresholds INTEGER  number of thresholds to be tried divided evenly
                           between 0 and 1

  --init-threshold FLOAT   value to initialise threshold values
  --help                   Show this message and exit.
```

### Pretrain

Pretrain pretrains an embedding based model like Doc2Vec and in the
future a language model like BERT from unlabeled training examples.

It allows us to leverage transfer learning in domains with limited
data. It applies more to wellcome-science than bioasq-mesh since the
latter contains enough examples to not being in need of transfer 
learning.

```
Usage: grants_tagger pretrain [OPTIONS] DATA_PATH MODEL_PATH

Arguments:
  DATA_PATH   data to pretrain model on  [required]
  MODEL_PATH  path to save mode  [required]

Options:
  --model-name TEXT  name of model to pretrain  [default: doc2vec]
  --config PATH      config file with arguments for pretrain
  --help             Show this message and exit.
```

### Download

This command is under development. The goal is to be able to download
pretrained models and data from sources like EPMC.

### Explain 

This command is under development. The goals is to be able to get
feature importance scores on either words or features such as tfidf
values.

# Develop

## Data

If you work for Wellcome and have access to our AWS account,
you easily download the raw data by typing `make sync_data`.
This will give you access to both the custom science tags
dataset and the MeSH data.

The MeSH data can be downloaded from various places like EPMC.
Grants tagger currently uses a sample provided from the [BioASQ](http://www.bioasq.org)
competition that contains tags for approx 14M publications from PubMed.

## Venv

To create and setup the development environment
```
make virtualenv
```
This will create a new virtualenv and install requirements for tests
and development. It will also install grants tagger in editable mode.

## Reproduce

To reproduce production models for mesh and wellcome science you can
run `dvc repro`. Note that mesh models require a GPU to train and 
depending on the parameters it might take from 1 to several days.

You can reproduce individual experiments using one of the configs in
the dedicated `/configs` folder. You can run all steps of the pipeline
using `./scripts/run_config.sh VERSION`. You can also run individual steps
with the CLI commands e.g. `grants_tagger preprocess wellcome-science --config path_to_config`
and `grants_tagger train --config path_to_config`.

## Bring your own data

To use grants_tagger with your own data the main thing you need to
implement is a new preprocess function that creates a JSONL with the
fields `text`, `tags` and `meta`. Meta can be even left empty if you
do not plan to use it. You can easily plug the new preprocess into the
cli by importing your function to `grants_tagger/__main__.py` and
define the subcommand name for your preprocess. For example if the
function was preprocessing EPMC data for MESH it could be
```
@preprocess_app.command()
def epmc_mesh(...)
```
and you would be able to run `grants_tagger preprocess epmc_mesh ...`

## Bring your own models

To use grants_tagger with your own model you need to define a class
for your model that adheres to the sklearn api so implements a
`fit`, `predict`, `predict_proba` and `set_params` but also a `save` 
and `load`.

Then you need to import you class to `models.py` and add it in `create_model`
as a separate approach with a name. Assuming your new approach is a bilstm
with attention

```
from bilstm_attention import BiLSTMAttention

...

def create_model(approach, params)
...
elif approach == 'bilstm-attention':
	model = BiLSTMAttention()
...
```

## Experiment

To make our experiments reproducible we use a config system (not DVC).
As such you need to create a new config that describes all parameters
for the various steps and run each step with the config or use 
`./scripts/run_config.sh`

## Package

To package a model run `make build`. This will create a wheel in
dist that you can distribute and `pip install`

## Test

`make test`
