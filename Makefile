BUCKET := datalabs-data
PUBLIC_BUCKET := datalabs-public
PROJECT := science_tagging
PROJECT_BUCKET := $(BUCKET)/science_tagging
PRODIGY_BUCKET := datalabs-packages/Prodigy
VIRTUALENV := build/virtualenv

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

PRODIGY_WHEEL := prodigy-1.8.5-cp35.cp36.cp37-cp35m.cp36m.cp37m-$(OSFLAG)_x86_64.whl

.PHONY:sync_data_to_s3
sync_data_to_s3:
	aws s3 sync science_tagger/data/ s3://$(PROJECT_BUCKET)/data/

.PHONY:sync_data_from_s3
sync_data_from_s3:
	aws s3 sync s3://$(PROJECT_BUCKET)/data/raw science_tagger/data/raw --exclude "*allMeSH*"

.PHONY:sync_processed_data_from_s3
sync_processed_data_from_s3:
	aws s3 sync s3://$(PROJECT_BUCKET)/data/processed science_tagger/data/processed --exclude "*disease_mesh*"

.PHONY:sync_models_to_s3
sync_models_to_s3:
	aws s3 sync science_tagger/models/ s3://$(PROJECT_BUCKET)/models/

.PHONY:sync_models_from_s3
sync_models_from_s3:
	aws s3 sync s3://$(PROJECT_BUCKET)/models/ science_tagger/models/ --exclude "*" --include "*label_binarizer*" --include "*$(INCLUDE)*"

.PHONY:virtualenv
virtualenv:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python python3 $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r requirements.txt
	$(VIRTUALENV)/bin/pip3 install -e .

.PHONY: download_spacy_models
download_spacy_models: 
	$(VIRTUALENV)/bin/python -m spacy download en_core_web_sm
	$(VIRTUALENV)/bin/python -m spacy download en_trf_bertbaseuncased_lg

.PHONY: install_prodigy
install_prodigy: 
	aws s3 cp s3://$(PRODIGY_BUCKET)/$(PRODIGY_WHEEL) ./build/$(PRODIGY_WHEEL)
	$(VIRTUALENV)/bin/pip3 install ./build/$(PRODIGY_WHEEL)

.PHONY: setup
setup: virtualenv download_spacy_models

.PHONY: test
test:
	$(VIRTUALENV)/bin/pytest --disable-warnings -v

.PHONY: build
build:
	$(VIRTUALENV)/bin/python setup.py bdist_wheel

.PHONY: deploy
deploy:
	aws s3 cp --recursive --exclude "*" --include "*.whl" --acl public-read dist/ s3://$(PUBLIC_BUCKET)/$(PROJECT)
