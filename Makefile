PROJECT_NAME := grants_tagger

DATA_BUCKET := datalabs-data
PUBLIC_BUCKET := datalabs-public
PROJECT_BUCKET := $(DATA_BUCKET)/$(PROJECT_NAME)

VIRTUALENV := venv

# Default include to match no file
INCLUDE := include_nothing

# Determine OS (from https://gist.github.com/sighingnow/deee806603ec9274fd47)
# Used when downloading prodigy wheel

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	OSFLAG := linux
endif
ifeq ($(UNAME_S),Darwin)
	OSFLAG := macosx_10_13
endif

.PHONY:sync_data
sync_data:
	aws s3 sync science_tagger/data/ s3://$(PROJECT_BUCKET)/data/
	aws s3 sync s3://$(PROJECT_BUCKET)/data/raw science_tagger/data/raw --exclude "*allMeSH*"

.PHONY: sync_artifacts
sync_artifacts:
	echo "Sync processed data"
	aws s3 sync s3://$(PROJECT_BUCKET)/data/processed science_tagger/data/processed
	echo "Sync models"
	aws s3 sync models/ s3://$(PROJECT_BUCKET)/models/
	aws s3 sync s3://$(PROJECT_BUCKET)/models/ models/

.PHONY:virtualenv
virtualenv:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	python3 -m venv $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r requirements.txt
	$(VIRTUALENV)/bin/pip3 install -e .

.PHONY: test
test:
	$(VIRTUALENV)/bin/pytest --disable-warnings -v

.PHONY: build
build:
	$(VIRTUALENV)/bin/python setup.py bdist_wheel

.PHONY: deploy
deploy:
	aws s3 cp --recursive --exclude "*" --include "*.whl" --acl public-read dist/ s3://$(PUBLIC_BUCKET)/$(PROJECT_NAME)
