#!/usr/bin/env bash

set -e

# get config from input
CONFIG=$1

# preprocess
venv/bin/python -m grants_tagger preprocess bioasq-mesh --config ${CONFIG}

# train
venv/bin/python -m grants_tagger train --config ${CONFIG}

# evaluate
# evaluate command is reserved for production models but train also evaluates
