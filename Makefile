.PHONY: run clean install

# Default Python interpreter
PYTHON = python

# Project directories
ENGINE_DIR = engine
MODEL_PIPELINE_DIR = model-pipeline
OUTPUT_DIR = output

# Default configuration file
CONFIG_FILE = config-test.yaml

AI_ENGINE_CONTAINER_NAME := ai-engine-simulator
AI_ENGINE_IMAGE_NAME := ai-engine-api
AI_ENGINE_DIR := ai-engine
PROJECT_DIR := $(shell pwd)

# OS detection
UNAME_S := $(shell uname -s)

# Function-like variable to open a new terminal and run a make target depending on OS
ifeq ($(UNAME_S),Darwin)
RUN_IN_TERMINAL = @osascript -e 'tell application "Terminal" to do script "cd $(PROJECT_DIR); make $(1)"'
else
RUN_IN_TERMINAL = gnome-terminal -- bash -lc 'cd $(PROJECT_DIR); make $(1); exec bash'
endif

# Default target
all: install cli

install:
	pip install -r requirements.txt

cli:
	$(PYTHON) cli.py

api:
	$(PYTHON) api.py

fake-monitor:
	$(PYTHON) -m uvicorn fake_monitor:app --host 0.0.0.0 --port 8082 --reload

fake-actuator:
	$(PYTHON) -m uvicorn fake_actuator:app --host 0.0.0.0 --port 8084 --reload

# --- Cross-platform helpers to open new terminal windows and run services ---
.PHONY: start-api-terminal start-fake-monitor-terminal start-fake-actuator-terminal start-all-terminals dev-start curl-start

start-api-terminal:
	$(call RUN_IN_TERMINAL,api)

start-fake-monitor-terminal:
	$(call RUN_IN_TERMINAL,fake-monitor)

start-fake-actuator-terminal:
	$(call RUN_IN_TERMINAL,fake-actuator)

start-all-terminals: start-api-terminal start-fake-monitor-terminal start-fake-actuator-terminal
	@echo "Launched API, Fake Monitor, and Fake Actuator in separate terminal windows."

curl-start:
	@curl -X POST http://0.0.0.0:8083/start || true

# Launch all services in terminals and then trigger the start curl
dev-start: start-all-terminals
	@sleep 5
	@$(MAKE) curl-start

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
			--env-file .env \
			$(AI_ENGINE_IMAGE_NAME);

run-docker-cli:
	docker run --rm --name $(AI_ENGINE_CONTAINER_NAME) -p 8083:8083 --env-file .env $(AI_ENGINE_IMAGE_NAME)

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
	@echo "  help         : Show this help message"
	@echo "  ---"
	@echo "> API Mode:"
	@echo "  ---"
	@echo "  api          : Run the engine module on api mode"
	@echo "  run-docker-api : Run the engine module on api mode on docker"
	@echo "  build-docker-api : Build the engine module on api mode on docker"
	@echo "  stop-rm-container : Stop and remove the docker container"
	@echo "  fake-monitor : Run the fake monitor module"
	@echo "  fake-actuator : Run the fake actuator module" 
	@echo "  start-api-terminal : Open a new terminal window (macOS Terminal or gnome-terminal) and run the API"
	@echo "  start-fake-monitor-terminal : Open a new terminal window (macOS Terminal or gnome-terminal) and run the Fake Monitor"
	@echo "  start-fake-actuator-terminal : Open a new terminal window (macOS Terminal or gnome-terminal) and run the Fake Actuator"
	@echo "  start-all-terminals : Launch API, Fake Monitor, and Fake Actuator in separate terminal windows"
	@echo "  dev-start : Start all terminals and trigger the start curl (POST /start)"
	@echo "  ---"
	
	@echo "> CLI Mode:"
	@echo "  ---"
	@echo "  cli          : Run the engine module on cli mode"
	@echo "  all          : Default target, install and runs the engine on cli mode"
	@echo "  install      : Install dependencies"
	@echo "  run-with-config : Run with a specific config file on cli mode"
	@echo "  clean        : Clean generated files"
	@echo "  build-docker-cli : Build the engine module on cli mode on docker"
	@echo "  run-docker-cli : Run the engine module on cli mode on docker"
	@echo "  stop-rm-container : Stop and remove the docker container"
	@echo "  ---"
	