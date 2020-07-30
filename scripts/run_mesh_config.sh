#!/usr/bin/env bash

set -e

# get config from input
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
VERSION=$1
CONFIG="${SCRIPT_DIR}/../configs/disease_mesh/${VERSION}.ini"

# preprocess
venv/bin/python -m grants_tagger.preprocess_mesh --config ${CONFIG}

# create label binarizer
venv/bin/python -m grants_tagger.create_mesh_label_binarizer --config ${CONFIG}

# train
venv/bin/python -m grants_tagger.train --config ${CONFIG}
