.PHONY: run clean install

# Default Python interpreter
PYTHON = python3

# Project directories
ENGINE_DIR = engine
MODEL_PIPELINE_DIR = model-pipeline
OUTPUT_DIR = output

# Default configuration file
CONFIG_FILE = config.yaml

AI_ENGINE_CONTAINER_NAME := ai-engine-simulator
AI_ENGINE_IMAGE_NAME := ai-engine-api
AI_ENGINE_DIR := ai-engine

# Default target
all: install run

install:
	pip install -r requirements.txt

run:
	$(PYTHON) cli.py

cli:
	$(PYTHON) cli.py

api:
	$(PYTHON) api.py

fake-monitor:
	$(PYTHON) -m uvicorn fake_monitor:app --host 0.0.0.0 --port 8082 --reload

fake-actuator:
	$(PYTHON) -m uvicorn fake_actuator:app --host 0.0.0.0 --port 8084 --reload

run-with-config:
	$(PYTHON) cli.py --config $(CONFIG_FILE)

build-docker-api:
	docker build -t $(AI_ENGINE_IMAGE_NAME) -f Dockerfile.api .

build-docker-cli:
	docker build -t $(AI_ENGINE_IMAGE_NAME) -f Dockerfile.cli .

run-docker-api:
	sudo docker run -d --rm \
			--name $(AI_ENGINE_CONTAINER_NAME) \
			--network host \
			-p 8083:8083 \
			$(AI_ENGINE_IMAGE_NAME);

run-docker-cli:
	docker run --name $(AI_ENGINE_CONTAINER_NAME) -p --rm 8083:8083 $(AI_ENGINE_IMAGE_NAME)

stop-rm-container:
	docker stop $(AI_ENGINE_CONTAINER_NAME)
	docker rm $(AI_ENGINE_CONTAINER_NAME)

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
