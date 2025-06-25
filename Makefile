.PHONY: run clean install

# Default Python interpreter
PYTHON = python

# Project directories
ENGINE_DIR = engine
MODEL_PIPELINE_DIR = model-pipeline
OUTPUT_DIR = output

# Default configuration file
CONFIG_FILE = $(ENGINE_DIR)/config.yaml

# Default target
all: install run

install:
	pip install -r requirements.txt

run:
	$(PYTHON) main.py

cli:
	$(PYTHON) cli.py

api:
	$(PYTHON) api.py

run-with-config:
	$(PYTHON) cli.py --config $(CONFIG_FILE)

build-docker-api:
	docker build -t ai-engine-api -f Dockerfile.api .

build-docker-cli:
	docker build -t ai-engine-cli -f Dockerfile.cli .

run-docker-api:
	docker run --name ai-engine-api -p 8083:8083 ai-engine-api

run-docker-cli:
	docker run --name ai-engine-cli -p 8083:8083 ai-engine-cli

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type f -name "recommendations.csv" -delete
	find $(OUTPUT_DIR) -type f -name "*_predictions.csv" -delete

help:
	@echo "Available targets:"
	@echo "  all          : Default target, runs the engine"
	@echo "  install      : Install dependencies"
	@echo "  run          : Run the engine module"
	@echo "  run-with-config : Run with a specific config file"
	@echo "  clean        : Clean generated files"
	@echo "  help         : Show this help message"
