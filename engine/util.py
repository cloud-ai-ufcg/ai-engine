import os
import logging
import logging.handlers
import yaml
from typing import Optional


def setup_logger(
    name: str = "ai_engine",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Setup and configure a logger.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional file path for log file
        log_format: Optional custom log format

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "[%(asctime)s] %(levelname)s [%(name)s] - %(message)s"

    formatter = logging.Formatter(log_format)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Create default logger for the package
logger = setup_logger()


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger with the specified name.

    If name is None, returns the default logger.
    Otherwise, returns a child logger with the specified name.

    Args:
        name: Logger name (optional)

    Returns:
        Logger instance
    """
    if name is None:
        return logger

    return logging.getLogger(f"ai_engine.{name}")


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    """
    Configure global logging settings.

    Args:
        level: Logging level
        log_file: Optional file path for log file
        log_format: Optional custom log format
    """
    global logger
    level = config.get('logging', {}).get('level', logging.INFO)
    log_file = config.get('logging', {}).get('file', None)
    logger = setup_logger(level=level, log_file=log_file, log_format=log_format)


def load_config(config_path=None):
    """
    Loads configuration from a YAML file

    Args:
        config_path: Path to the configuration file. If None, uses the default config.yaml

    Returns:
        dict: Configuration dictionary
    """
    global config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config
