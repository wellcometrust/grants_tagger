[![Build Status](https://travis-ci.com/wellcometrust/grants_tagger.svg?branch=main)](https://travis-ci.com/wellcometrust/grants_tagger)
[![codecov](https://codecov.io/gh/wellcometrust/grants_tagger/branch/main/graph/badge.svg)](https://codecov.io/gh/wellcometrust/grants_tagger)
![GitHub](https://img.shields.io/github/license/wellcometrust/grants_tagger)

# Grants tagger 🔖

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

# 💻 Install and use

Easiest way to install is with pip.
The project is available in the PyPI repository at https://pypi.org/project/grants-tagger/.

```
pip install grants-tagger
```

Alternatively you can clone this repository and manually install with
```
pip install setup.py
```

You can download the mesh pretrained model using:
```
grants_tagger download model mesh
```
Or, you can download some pretrained models from the releases section of this repository. Unzip this in the `models` folder. This will contain the mesh model and the label-binarizer.

After that, you can predict a text's tag using the `predict` cli command.

Grants tagger comes with a nice CLI with the following commands

## ⌨️  Commands

| Commands        |                                                              |
| --------------- | ------------------------------------------------------------ |
| ⚙️  preprocess   | preprocess data to use for training                          |
| 🔥 train        | trains a new model                                           |
| 📈 evaluate     | evaluate performance of pretrained model                     |
| 🔖 predict      | predict tags given a grant abstract using a pretrained model |
| 🎛 tune         | tune params and threshold                                    |
| 📚 pretrain     | pretrains embeddings or language model using unlabeled data  |
| ⬇️  download    | download trained models and data from EPMC                   |
| 🐋  docker      | how to run grants_tagger in a docker container               |
| 🌐 visualize    | creates a streamlit app to interactively tag grants          |

in square brackets the commands that are not implemented yet

### ⚙️  Preprocess

Preprocess creates a JSONL datafile with `text`, `tags` and `meta` as keys.
Text and tags are used for training whereas meta can be useful during annotation
or to analyse predictions and performance. Each dataset needs its own
preprocessing so the current preprocess works with the wellcome-science dataset and
the bioasq-mesh one. If you want to use a different dataset see section on bringing
your own data under development.

### inclusion list

The preprocessing step, described below, can also exlcude terms we do not want to use
in training. Such terms can be
  1) terms that have an attribute that contains "DO NOT USE", which are there to help
  catalogue or organise the tree structure. These terms should not be used for tagging
  however
  2) terms that we manually want to exclude. These can be stored in a .csv which can
  feed into `create_inclusion_list.py`


#### wellcome-science
```
Usage: grants_tagger preprocess wellcome-science [OPTIONS] [INPUT_PATH]
                                                 [OUTPUT_PATH]
                                                 [LABEL_BINARIZER_PATH]

Arguments:
  [INPUT_PATH]   path to raw Excel file with tagged or untagged grant data
  [TRAIN_OUTPUT_PATH]  path to JSONL output file that will be generated for the train set
  [LABEL_BINARIZER_PATH] path to pickle file that will contain the label binarizer

Options:
  --test-output-path PATH   path to JSONL output file that will be generated for the test set
  --text-cols TEXT          comma delimited column names to concatenate to text
  --meta-cols TEXT          comma delimited column names to include in the meta
  --test-split FLOAT        split percentage for test data. if None no split.
  --config PATH             path to config file that defines the arguments
  --help                    Show this message and exit.
```

#### bioasq-mesh
```
Usage: grants_tagger preprocess bioasq-mesh [OPTIONS] [INPUT_PATH]
                                            [TRAIN_OUTPUT_PATH]
                                            [LABEL_BINARIZER_PATH]

Arguments:
  [INPUT_PATH]            path to BioASQ JSON data
  [TRAIN_OUTPUT_PATH]     path to JSONL output file that will be generated for
                          the train set

  [LABEL_BINARIZER_PATH]  path to pickle file that will contain the label
                          binarizer


Options:
  --test-output-path TEXT  path to JSONL output file that will be generated
                           for the test set

  --mesh-tags-path TEXT    path to mesh tags to filter
  --test-split FLOAT       split percentage for test data. if None no split.
                           [default: 0.01]

  --filter-years TEXT      years to keep in form min_year,max_year with both
                           inclusive

  --config PATH            path to config files that defines arguments
  --help                   Show this message and exit.
```

### 🔥 Train

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

  --threshold FLOAT               threshold to assign a tag
  --data-format TEXT              format that will be used when loading the
                                  data. One of list,generator  [default: list]

  --sparse-labels / --no-sparse-labels
                                  flat about whether labels should be sparse
                                  when binarized  [default: False]

  --cache-path PATH               path to cache data transformartions
  --config PATH
  --cloud / --no-cloud            flag to train using Sagemaker  [default:
                                  False]

  --instance-type TEXT            instance type to use when training with
                                  Sagemaker  [default: local]

  --help                          Show this message and exit.
```

Label binarizer converts tags to a Y matrix with one column per tag. This is
neccesary as this is a multilabel problem so multiple tags may be assigned to
one grant.

Parameters is a JSON like dump (essentially a stringified dict) with all params
that the model expects in an sklearn fashion.

#### Slack

There is a small functionality to notify a Slack channel after you train. For this,
you need to `chmod +x /train-slack-notify.sh`, set up the environement variables
`SLACK_HOOK` and `SLACK_USER` and, when running your training command, run it in
quotation marks like:

```bash
./train-slack-notify.sh 'grants_tagger train etc...' # Quotations needed!
```

### 📈 Evaluate

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
and `mesh-cnn`. Note that train also outputs evaluation scores so for models
not made into production this is the way to evaluate. The plan is to extend
evaluate to all models when train starts training explicit model approaches.

```
Usage: grants_tagger evaluate model [OPTIONS] APPROACH MODEL_PATH DATA_PATH
                                    LABEL_BINARIZER_PATH

Arguments:
  APPROACH              model approach e.g.mesh-cnn  [required]
  MODEL_PATH            comma separated paths to pretrained models  [required]
  DATA_PATH             path to data that was used for training  [required]
  LABEL_BINARIZER_PATH  path to label binarize  [required]

Options:
  --threshold TEXT                threshold or comma separated thresholds used
                                  to assign tags  [default: 0.5]

  --results-path TEXT             path to save results
  --mesh-tags-path TEXT           path to mesh subset to evaluate
  --split-data / --no-split-data  flag on whether to split data in same way as
                                  was done in train  [default: True]

  --grants / --no-grants          flag on whether the data is grants data
                                  instead of publications to evaluate MeSH
                                  [default: False]

  --config PATH                   path to config file that defines arguments
  --help                          Show this message and exit.
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

MTI is the automatic mesh indexer from NLM. https://ii.nlm.nih.gov/MTI/
To get MTI annotations you need to submit grants for tagging through an
email service and get the results which you can use here for evaluation.
The service is called Batch MetaMap and to use it you need to have an
account https://uts.nlm.nih.gov/uts/

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

### 🔖 Predict

Predict assigns tags on a given abstract text that you can pass as argument.
It is not meant to be used for tagging multiple grants, tag command is reserved
for that. Similar to evaluate and train an approach is needed to know which
model will be used. Similar to evaluate it also works on models that have
made it to production, see list in evaluate.

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

### 🎛 Tune

Tune optimises params of choice or the threshold that is used to
assign tags. Parameter optimisation makes use of sklearn GridSearch
and currently works on a predefined set of approaches and params. At a later
point it will accept the parameter search space and will perform the optimisation
either locally or in SageMaker. Threshold optimises the individual thresholds
that assign tags. Each tag can have an individual threshold that together maximises
f1 score.


#### params
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

#### threshold
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
  --val-size FLOAT                validation size of text data to use for
                                  tuning  [default: 0.8]

  --nb-thresholds INTEGER         number of thresholds to be tried divided
                                  evenly between 0 and 1

  --init-threshold FLOAT          value to initialise threshold values
  --split-data / --no-split-data  flag on whether to split data as was done
                                  for train  [default: True]

  --help                          Show this message and exit.
```

### 📚 Pretrain

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

### ⬇️  Download

This commands enables you to download mesh data from EPMC
and pre trained models.

```
Usage: grants_tagger download [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  epmc-mesh
  model
```

Available models:

* mesh

### 🐋 Run in a Docker container

Grants_tagger can be installed using pip: `pip install grants-tagger`.
Have a look on [pypi.org for more information](https://pypi.org/project/grants-tagger/).

The latest classifier uses [PECOS eXtreme Multi-label Classification](https://github.com/amzn/pecos/blob/mainline/pecos/xmc/xlinear/README.md)
developed by Amazon. It relies on a library called [pecos](https://pypi.org/project/libpecos/)
which only works on Linux.

As such, a user might want to create a docker image which can run
grants-tagger. In order to do so, you need to create a Dockerfile,
which sets up a linux environment. E.g.
```
FROM python:3.8-slim-bullseye
```

Download the wheel from Pypi and add the following as a minimum
to the Dockerfile:
```
RUN apt-get update && \
    apt-get install -y git build-essential

RUN DEBIAN_FRONTEND="noninteractive" && \
    pip install --upgrade pip && \
    pip install grants_tagger
```

You can now build a docker container using `docker build --no-cache -t grants_tagger .`
and run (and jump in) a container using `docker run -dit grants_tagger`.

### 🌐 Visualize

This command uses streamlit to create a web app in which you
can interactively tag grants while choosing the threshold
and the model to use. It currently works only with Wellcome
trained models.

If you want to visualise

```bash
make build-streamlit-docker
```

And then
```bash
docker run -p 8501:8501 streamlitapp
```

And when you're really happy with it you can push the tag

```bash
make -f Makefile.streamlit docker-push
```

# 🧑🏻‍💻  Develop

## 📖 Data

If you work for Wellcome and have access to our AWS account,
you easily download the raw data by typing `make sync_data`.
This will give you access to both the custom science tags
dataset and the MeSH data. Note that the mesh data is 50+GB.
If you want to download only the science or mesh data, you
can do so with `make sync_science_data` and `make sync_mesh_data`
respectively.

The MeSH data can be downloaded from various places like EPMC.
Grants tagger currently uses a sample provided from the [BioASQ](http://www.bioasq.org)
competition that contains tags for approx 14M publications from PubMed.

## 🐍 Venv

To create and setup the base environment
```
make virtualenv
```
This will create a new virtualenv and install requirements for tests. It will also install grants tagger in editable mode.

For full development environment, install with:
```
make virtualenv-dev
```

If you want to add additional dependencies, add the library to
`unpinned_requirements.txt` and run `make update-requirements`. This
will ensure that all requirements in the development enviroment are pinned
to exact versions which ensures the code will continue running as
expected in the future when newer versions will have been published.

## 📋 Env variables

You need to set the following variables for sagemaker or sync to work. If you want to
participate to BIOASQ competition you need to also set some variables.

Variable              | Required for       | Description
--------------------- | ------------------ | ----------
PROJECTS_BUCKET       | sagemaker, sync    | s3 bucket for data and models e.g. datalabs-data
PROJECT_NAME          | sagemaker, sync    | s3 prefix for specific project e.g.grants_tagger
AWS_ACCOUNT_ID        | sagemaker          | aws organisational account id, ask aws adminstrator
ECR_IMAGE             | sagemaker          | ecr image with dependencies to run grants tagger
SAGEMAKER_ROLE        | sagemaker          | aws sagemaker role, ask aws administrator
AWS_ACCESS_KEY_ID     | sagemaker, sync    | aws access key, for aws cli to work
AWS_SECRET_ACCESS_KEY | sagemaker, sync    | aws secret key, for aws cli to work
BIOASQ_USERNAME       | bioasq             | username with which registered in BioASQ
BIOASQ_PASSWORD       | bioasq             | password            --//--

There is a `.envrc.template` with the env variables needed. If you
use [direnv](https://direnv.net) then you can use it to populate
your `.envrc` which will export the variables automatically, otherwise
ensure you export every time or include in your bash profile.

Note that aws keys are not included as there are various ways to
setup aws, for example using `aws configure` or `AWS_PROFILE` and
`.aws/credentials`

## ✔️  Reproduce

To reproduce production models we use DVC. DVC defines a directed
acyclic graph (DAG) of steps that need to run to reproduce a model
or result. You can see all steps with `dvc dag`. You can reproduce
all steps with `dvc repro`. You can reproduce any step of the DAG
with `dvc repro STEP_NAME` for example `dvc repro train_tfidf_svm`.
Note that mesh models require a GPU to train and depending on the
parameters it might take from 1 to several days.

You can reproduce individual experiments using one of the configs in
the dedicated `/configs` folder. You can run all steps of the pipeline
using `./scripts/run_DATASET_config.sh path_to_config` where DATASET
can be one of science or mesh. You can also run individual steps
with the CLI commands e.g. `grants_tagger preprocess wellcome-science --config path_to_config`
and `grants_tagger train --config path_to_config`.

## 💾 Bring your own data

To use grants_tagger with your own data the main thing you need to
implement is a new preprocess function that creates a JSONL with the
fields `text`, `tags` and `meta`. Meta can be even left empty if you
do not plan to use it. You can easily plug the new preprocess into the
cli by importing your function to `grants_tagger/cli.py` and
define the subcommand name for your preprocess. For example if the
function was preprocessing EPMC data for MESH it could be
```
@preprocess_app.command()
def epmc_mesh(...)
```
and you would be able to run `grants_tagger preprocess epmc_mesh ...`

## 📦 Bring your own models

To use grants_tagger with your own model you need to define a class
for your model that adheres to the sklearn api so implements a
`fit`, `predict`, `predict_proba` and `set_params` but also a `save`
and `load`. Each custom model is defined in their own python script inside
the `grants_tagger/models` folder.

Then you need to import you class to `grants_tagger/models/create_model.py` and add it in
`create_model` as a separate approach with a name. Assuming your new approach
is a bilstm with attention

```
from grants_tagger.models.bilstm_attention import BiLSTMAttention

...

def create_model(approach, params)
...
elif approach == 'bilstm-attention':
	model = BiLSTMAttention()
...
```

## 🔬 Experiment

To make our experiments reproducible we use a config system (not DVC).
As such you need to create a new config that describes all parameters
for the various steps and run each step with the config or use
`./scripts/run_science_config.sh` or `./scripts/run_mesh_config.sh`
depending on whether you want to reproduce a wellcome science or mesh
model.

We record the results of experiments in `docs/results.md` for wellcome
science and `docs/mesh_results.md` for bioasq mesh.

## 📦 Package

To package the code run `make build`. This will create a wheel in
dist that you can distribute and `pip install`. `make deploy` pushes
that wheel in a public Wellcome bucket that you can use if you have
access to write into it.

Packaged models are produced through dvc by running `dvc repro` and
stored in s3 by running `make sync_artifacts`. This might soon change to
a separate command that pushes models to a public bucket so others
can download.

Thus the recommended way for producing models that are released is
through DVC repro and as such you need to define the params that
produce it in the params.yaml as well as the pipeline that produces
it in `dvc.yaml`. The `params.yaml` is the equivalent of a config file and
`dvc.yaml` is the equivalent of `run_config.sh`

## 🚦 Test

Run tests with `make test`. If you want to write some additional tests,
they should go in the subfolde `tests/`

## 🏗️ Makefile

A Makefile is being used to automate various operations. You can
get all make commands by running `make`. All the commands have been
mentioned in previous sections.

```
Usage: make [task]

task                 help
------               ----
sync_data            Sync data to and from s3
sync_artifacts       Sync processed data and models to and from s3
virtualenv           Creates virtualenv
update-requirements  Updates requirement
test                 Run tests
build                Create wheel distribution
deploy               Deploy wheel to public s3 bucket
clean                Clean hidden and compiled files
help                 Show help message
```

## ✍️ Scripts

Additional scripts, mostly related to Wellcome Trust-specific code can be
found in `/scripts`. Please refer to the [readme](scripts/README.md) therein for more info
on how to run those.

## 🏋️ Slim

```bash
docker build -f Dockerfile.xlinear -t xlinear .
```

Then run mounting the model volumes

```bash
docker run -v $(PWD)/models/:/code/models -d -t xlinear
```

Get your image number, and ssh into the container to debug

```bash
predict_tags(['This is a malaria grant'], model_path='models/xlinear-0.2.3', label_binarizer_path='models/label_binarizer.pkl', probabilities=True, threshold=0.01)
```

### Train a slim model

To train a xlinear model, you can use the following functionality:

```python
from grants_tagger.slim.mesh_xlinear import train_and_evaluate

parameters={'ngram_range': (1, 1), 'beam_size': 30, 'only_topk': 200, 'min_weight_value': 0.1, 'max_features': 400_000}

train_and_evaluate(train_data_path='data/processed/train_mesh2021_toy.jsonl',
                   test_data_path='data/processed/test_mesh_2021_toy.jsonl',
                   label_binarizer_path='models/label_binarizer_toy.pkl',
                   model_path='models/xlinear-toy',
                   results_path='results/mesh_xlinear_toy.json',
                   full_report_path='results/mesh_xlinear_toy_full_report.json')
```

If you want to change the parameters, you can either pass it as a dictionary to `train_and_evaluate`, as above,
or create a config file (check `configs/mesh/2022.3.0.ini`).
