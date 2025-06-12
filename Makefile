.PHONY: run clean install test

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

# Install dependencies
install:
	pip install -r requirements.txt

# Run the engine module
run:
	cd $(ENGINE_DIR) && $(PYTHON) main.py

api:
	cd $(ENGINE_DIR) && $(PYTHON) api.py

# Run with custom config
run-with-config:
	cd $(ENGINE_DIR) && $(PYTHON) main.py --config $(CONFIG_FILE)

# Clean generated files
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

# Run tests
test:
	$(PYTHON) -m pytest tests/

# Show help
help:
	@echo "Available targets:"
	@echo "  all          : Default target, runs the engine"
	@echo "  install      : Install dependencies"
	@echo "  run          : Run the engine module"
	@echo "  run-with-config : Run with a specific config file"
	@echo "  clean        : Clean generated files"
	@echo "  test         : Run tests"
	@echo "  help         : Show this help message"
