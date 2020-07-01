#!/usr/bin/env bash

set -e

# get config from input
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
VERSION=$1
CONFIG="${SCRIPT_DIR}/../configs/${VERSION}.ini"

# preprocess
venv/bin/python -m grants_tagger.preprocess --config ${CONFIG}

# create label binarizer
venv/bin/python -m grants_tagger.create_label_binarizer --config ${CONFIG}

# pretrain
venv/bin/python -m grants_tagger.pretrain --config ${CONFIG}

# train
venv/bin/python -m grants_tagger.train --config ${CONFIG}

# evaluate
