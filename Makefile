.DEFAULT_GOAL := help

PRIVATE_PROJECT_BUCKET := $(PROJECTS_BUCKET)/$(PROJECT_NAME)
PUBLIC_PROJECT_BUCKET := datalabs-public/$(PROJECT_NAME)

PYTHON := python3.8
VIRTUALENV := venv
PIP := $(VIRTUALENV)/bin/pip


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
	$(PIP) install -r requirements.txt
	$(PIP) install --no-deps -e .
	$(PIP) install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz

update-requirements: VIRTUALENV := /tmp/update-requirements-venv/
update-requirements: ## Updates requirement
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	$(PYTHON) -m venv $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip install --upgrade pip
	$(VIRTUALENV)/bin/pip install -r unpinned_requirements.txt
	$(VIRTUALENV)/bin/pip install -r unpinned_test_requirements.txt
	echo "#Created by Makefile. Do not edit." > requirements.txt
	$(VIRTUALENV)/bin/pip freeze | grep -v pkg-resources==0.0.0 | grep -v wellcomeml >> requirements.txt
	echo "-e git://github.com/wellcometrust/WellcomeML.git@4e96150ff98ccbb3a12e137771fab362c02fa7f1#egg=wellcomeml" >> requirements.txt

.PHONY: test
test: ## Run tests
	$(VIRTUALENV)/bin/pytest --disable-warnings -v --cov=grants_tagger


.PHONY: build
build: ## Create wheel distribution
	$(VIRTUALENV)/bin/python setup.py bdist_wheel

.PHONY: deploy
deploy: ## Deploy wheel to public s3 bucket
	aws s3 cp --recursive --exclude "*" --include "*.whl" --acl public-read dist/ s3://$(PUBLIC_PROJECT_BUCKET)
	git tag v$(shell python setup.py --version)
	git push --tags
	$(VIRTUALENV)/bin/python -m twine upload --repository pypi dist/*

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
