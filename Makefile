.DEFAULT_GOAL := help


PRIVATE_PROJECT_BUCKET := $(PROJECTS_BUCKET)/$(PROJECT_NAME)
PUBLIC_PROJECT_BUCKET := datalabs-public/$(PROJECT_NAME)

PACKAGE_VERSION := $(shell venv/bin/python -c "import grants_tagger;print(grants_tagger.__version__.__version__)")
MESH_MODEL_PACKAGE := xlinear-$(PACKAGE_VERSION).tar.gz
MESH_MODEL := xlinear/model/
MESH_LABEL_BINARIZER := xlinear/label_binarizer.pkl
SCIENCE_TFIDF_SVM_MODEL:= tfidf-svm.pkl
SCIENCE_SCIBERT_MODEL := scibert
SCIENCE_LABEL_BINARIZER := label_binarizer.pkl

PYTHON := python3.7
VIRTUALENV := venv
PIP := $(VIRTUALENV)/bin/pip

UNAME := $(shell uname)

.PHONY: sync_data
sync_data: sync_science_data sync_mesh_data ## Sync data to and from s3

.PHONY:sync_science_data
sync_science_data: ## Sync science data to and from s3
	aws s3 sync data/raw/ s3://$(PRIVATE_PROJECT_BUCKET)/data/raw/ --exclude "*allMeSH*" --exclude "*desc*" --exclude "*disease_tags*"
	aws s3 sync s3://$(PRIVATE_PROJECT_BUCKET)/data/raw data/raw --exclude "*allMeSH*" --exclude "*desc*" --exclude "*disease_tags*"

.PHONY: sync_mesh_data
sync_mesh_data: ## Sync mesh data to and from s3
	aws s3 sync data/raw/ s3://$(PRIVATE_PROJECT_BUCKET)/data/raw/ --exclude "*" --include "*allMeSH*" --include "*desc*" --include "*disease_tags*"
	aws s3 sync s3://$(PRIVATE_PROJECT_BUCKET)/data/raw data/raw/ --exclude "*" --include "*allMeSH*" --include "*desc*" --include "*disease_tags*"

.PHONY: sync_artifacts
sync_artifacts: sync_science_artifacts sync_mesh_artifacts ## Sync processed data and models to and from s3

.PHONY: sync_science_artifacts
sync_science_artifacts: ## Sync science processed data and models
	echo "Sync processed data"
	aws s3 sync data/processed s3://$(PRIVATE_PROJECT_BUCKET)/data/processed --exclude "*" --include "*science*" 
	aws s3 sync s3://$(PRIVATE_PROJECT_BUCKET)/data/processed data/processed --exclude "*" --include "*science*"
	echo "Sync models"
	aws s3 sync models/ s3://$(PRIVATE_PROJECT_BUCKET)/models/ --exclude "*" --include "*science*"
	aws s3 sync s3://$(PRIVATE_PROJECT_BUCKET)/models/ models/ --exclude "*" --include "*science*"

.PHONY: sync_mesh_artifacts
sync_mesh_artifacts: ## Sync mesh processed data and models
	echo "Sync processed data"
	aws s3 sync data/processed s3://$(PRIVATE_PROJECT_BUCKET)/data/processed --exclude "*" --include "*mesh*"
	aws s3 sync s3://$(PRIVATE_PROJECT_BUCKET)/data/processed data/processed --exclude "*" --include "*mesh*"
	echo "Sync models"
	aws s3 sync models/ s3://$(PRIVATE_PROJECT_BUCKET)/models/ --exclude "*" --include "*mesh*"
	aws s3 sync s3://$(PRIVATE_PROJECT_BUCKET)/models/ models/ --exclude "*" --include "*mesh*" --exclude "*tfidf*"

virtualenv: ## Creates virtualenv
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python $(PYTHON) $(VIRTUALENV)
	$(PIP) install --upgrade pip
# As to why no identation see https://stackoverflow.com/questions/4483313/make-error-for-ifeq-syntax-error-near-unexpected-token
ifeq ($(UNAME), Linux)
	$(PIP) install libpecos==0.1.0
endif		
	$(PIP) install pytest tox
	$(PIP) install -r requirements.txt
	$(PIP) install --no-deps -e .
	$(PIP) install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz

update-requirements: VIRTUALENV := /tmp/update-requirements-venv/
update-requirements: ## Updates requirement
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python $(PYTHON) $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip install --upgrade pip
	$(VIRTUALENV)/bin/pip install -r unpinned_requirements.txt
	echo "#Created by Makefile. Do not edit." > requirements.txt
	$(VIRTUALENV)/bin/pip freeze | grep -v pkg_resources==0.0.0 >> requirements.txt
#	echo "-e git://github.com/wellcometrust/WellcomeML.git@tokenizer-decode#egg=wellcomeml" >> requirements.txt
	echo "git+https://github.com/nsorros/shap.git@dev" >> requirements.txt

.PHONY: test
test: ## Run tests
	$(VIRTUALENV)/bin/pytest --disable-warnings -v --cov=grants_tagger
	$(VIRTUALENV)/bin/tox


.PHONY: build
build: ## Create wheel distribution
	$(VIRTUALENV)/bin/python setup.py bdist_wheel
	tar -c -z -v -f models/$(MESH_MODEL_PACKAGE) models/$(MESH_MODEL) models/$(MESH_LABEL_BINARIZER)

.PHONY: deploy
deploy: ## Deploy wheel to public s3 bucket
	aws s3 cp --recursive --exclude "*" --include "*.whl" --acl public-read dist/ s3://$(PUBLIC_PROJECT_BUCKET)
	git tag v$(shell python setup.py --version)
	git push --tags
	$(VIRTUALENV)/bin/python -m twine upload --repository pypi dist/*
	aws s3 cp --acl public-read models/$(MESH_MODEL_PACKAGE) s3://datalabs-public/grants_tagger/models/$(MESH_MODEL_PACKAGE)
	aws s3 cp --recursive models/$(MESH_MODEL) s3://datalabs-data/grants_tagger/$(MESH_MODEL)
	aws s3 cp models/$(MESH_LABEL_BINARIZER) s3://datalabs-data/grants_tagger/models/$(MESH_LABEL_BINARIZER)
	aws s3 cp models/$(SCIENCE_TFIDF_SVM_MODEL) s3://datalabs-data/grants_tagger/$(SCIENCE_TFIDF_SVM_MODEL)
	aws s3 cp --recursive models/$(SCIENCE_SCIBERT_MODEL) s3://datalabs-data/grants_tagger/$(SCIENCE_SCIBERT_MODEL)
	aws s3 cp models/$(SCIENCE_LABEL_BINARIZER) s3://datalabs-data/grants_tagger/$(SCIENCE_LABEL_BINARIZER)
	# XLinear model >2GB that GitHub accepts
 	# gh release upload v$(shell python setup.py --version) models/$(MESH_MODEL).tar.gz

.PHONY: clean
clean: ## Clean hidden and compiled files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*flymake*" -delete
	find . -type f -name "#*#" -delete

.PHONY: aws-docker-login
aws-docker-login:
	aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $(AWS_ACCOUNT_ID).dkr.ecr.eu-west-1.amazonaws.com

.PHONY: build-docker
build-docker: ## Builds Docker container with grants_tagger
	docker build -t $(ECR_IMAGE):latest -f Dockerfile .

.PHONY: push-docker
push-docker: aws-docker-login ## Pushes Docker container to ECR
	docker push $(ECR_IMAGE):latest

.PHONY: build-streamlit-docker
build-streamlit-docker: ## Builds Docker with streamlit and models
	aws s3 cp --recursive s3://datalabs-data/grants_tagger/models/disease_mesh_cnn-2021.03.1/ models/disease_mesh_cnn-2021.03.1/
	aws s3 cp s3://datalabs-data/grants_tagger/models/disease_mesh_label_binarizer.pkl models/
	aws s3 cp s3://datalabs-data/grants_tagger/models/tfidf-svm-2020.05.2.pkl models/
	aws s3 cp --recursive s3://datalabs-data/grants_tagger/models/scibert-2020.05.5/ models/
	aws s3 cp s3://datalabs-data/grants_tagger/models/label_binarizer.pkl models/
	docker build -t streamlitapp -f Dockerfile.streamlit .

.PHONY: install-private-requirements
install-private-requirements: ## Install the private datascience utils
	pip install httpx pyodbc psycopg2
	pip install -e git+ssh://git@github.com/wellcometrust/datascience.git#egg=wellcome-datascience-common
 

help: ## Show help message
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%s\n\n" "Usage: make [task]"; \
	printf "%-20s %s\n" "task" "help" ; \
	printf "%-20s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-20s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done
