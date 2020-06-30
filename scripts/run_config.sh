#!/usr/bin/env bash

set -e

# get config from input
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
VERSION=$1
CONFIG="${SCRIPT_DIR}/science_tagger/configs/${VERSION}.ini"

# preprocess
build/virtualenv/bin/python -m science_tagger.preprocess --config ${CONFIG}

# create label binarizer
build/virtualenv/bin/python -m science_tagger.create_label_binarizer --config ${CONFIG}

# pretrain
build/virtualenv/bin/python -m science_tagger.pretrain --config ${CONFIG}

# train
build/virtualenv/bin/python -m science_tagger.train --config ${CONFIG}

# evaluate
