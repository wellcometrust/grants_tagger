#!/usr/bin/env bash

set -e

# get config from input
CONFIG=$1

# preprocess
venv/bin/python -m grants_tagger.preprocess --config ${CONFIG}

# create label binarizer
venv/bin/python -m grants_tagger.create_label_binarizer --config ${CONFIG}

# pretrain
venv/bin/python -m grants_tagger.pretrain --config ${CONFIG}

# train
venv/bin/python -m grants_tagger.train --config ${CONFIG}

# evaluate
# evaluate command is reserved for production models but train also evaluates
