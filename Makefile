#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = model-training
PYTHON_VERSION = 3.12.9
PYTHON_INTERPRETER = python
VENV_NAME = .venv
VENV_BIN = $(VENV_NAME)/bin
PYTHON_VENV = $(VENV_BIN)/python
PIP_VENV = $(VENV_BIN)/pip

# Use venv python on Unix/macOS, handle Windows differently if needed
ifeq ($(OS),Windows_NT)
	VENV_BIN = $(VENV_NAME)/Scripts
	PYTHON_VENV = $(VENV_BIN)/python.exe
	PIP_VENV = $(VENV_BIN)/pip.exe
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up Python virtual environment
.PHONY: create_environment
create_environment:
	@if [ ! -d "$(VENV_NAME)" ]; then \
		$(PYTHON_INTERPRETER) -m venv $(VENV_NAME); \
		echo ">>> Virtual environment created. Activate with:"; \
		echo ">>> Unix/macOS: source $(VENV_BIN)/activate"; \
		echo ">>> Windows: $(VENV_NAME)\\Scripts\\activate"; \
	else \
		echo ">>> Virtual environment already exists at $(VENV_NAME)"; \
	fi

## Install Python dependencies
.PHONY: requirements
requirements: create_environment
	$(PIP_VENV) install --upgrade pip
	$(PIP_VENV) install -r requirements.txt

## Delete all compiled Python files and virtual environment
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf $(VENV_NAME)

## Run tests using pytest
.PHONY: test
test:
	$(PYTHON_VENV) -m pytest

## Lint using pylint, flake8, and bandit
.PHONY: lint
lint:
	$(PYTHON_VENV) -m pylint model_training/ tests/

.PHONY: lint_flake8
flake8:
	$(PYTHON_VENV) -m flake8 model_training/ tests/

.PHONY: lint_bandit
bandit:
	$(PYTHON_VENV) -m bandit -r model_training/

## Format source code (placeholder - could add black or autopep8 if needed)
.PHONY: format
format:
	@echo ">>> Formatting with pylint, flake8, bandit - run 'make lint' to check code quality"
	$(PYTHON_VENV) -m pylint model_training/ tests/ || true


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Download raw dataset
.PHONY: download
download:
	$(PYTHON_VENV) model_training/dataset.py download

## Preprocess raw data into cleaned dataset and vectorizer
.PHONY: preprocess
preprocess:
	$(PYTHON_VENV) model_training/dataset.py preprocess

## Split processed data into train/test sets
.PHONY: split
split:
	$(PYTHON_VENV) model_training/dataset.py split

## Train machine learning model
.PHONY: train
train:
	$(PYTHON_VENV) model_training/modeling/train.py

## Evaluate trained model and generate metrics
.PHONY: evaluate
evaluate:
	$(PYTHON_VENV) model_training/modeling/evaluate.py

## Run complete ML pipeline from start to finish
.PHONY: pipeline
pipeline: download preprocess split train evaluate
	@echo ">>> Complete ML pipeline finished successfully!"

## Legacy: Make dataset (same as preprocess)
.PHONY: data
data: preprocess

## Set up project and run all tasks
.PHONY: all
all: create_environment requirements lint test download preprocess split train evaluate
	@echo ">>> All tasks completed successfully!"


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
