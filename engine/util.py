import os
import sys
import logging
import logging.handlers
import yaml
from typing import Optional, Dict, Any, Union
import re
import json

# Default directory paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "model-pipeline/models"))
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "data/output"))
ENGINE_LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "logs"))

# Ensure log directory exists
os.makedirs(ENGINE_LOG_DIR, exist_ok=True)

# ANSI color codes
COLORS = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "BLUE": "\033[34m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "RED": "\033[31m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
    "BG_BLUE": "\033[44m",
    "BG_GREEN": "\033[42m",
    "BG_YELLOW": "\033[43m",
    "BG_RED": "\033[41m",
    "BG_MAGENTA": "\033[45m",
}


class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: f"{COLORS['BLUE']}[%(asctime)s] {COLORS['BOLD']}DEBUG{COLORS['RESET']}{COLORS['BLUE']} [%(name)s] - %(message)s{COLORS['RESET']}",
        logging.INFO: f"{COLORS['GREEN']}[%(asctime)s] {COLORS['BOLD']}INFO{COLORS['RESET']}{COLORS['GREEN']} [%(name)s] - %(message)s{COLORS['RESET']}",
        logging.WARNING: f"{COLORS['YELLOW']}[%(asctime)s] {COLORS['BOLD']}WARNING{COLORS['RESET']}{COLORS['YELLOW']} [%(name)s] - %(message)s{COLORS['RESET']}",
        logging.ERROR: f"{COLORS['RED']}[%(asctime)s] {COLORS['BOLD']}ERROR{COLORS['RESET']}{COLORS['RED']} [%(name)s] - %(message)s{COLORS['RESET']}",
        logging.CRITICAL: f"{COLORS['BG_RED']}{COLORS['WHITE']}[%(asctime)s] {COLORS['BOLD']}CRITICAL{COLORS['RESET']}{COLORS['BG_RED']}{COLORS['WHITE']} [%(name)s] - %(message)s{COLORS['RESET']}",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


# Flag to track if logging has been configured
_logging_configured = False


def setup_logger(
    name: str = "ai_engine",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Setup and configure a logger.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional file path for log file
        log_format: Optional custom log format
        use_colors: Whether to use colored output in console (default: True)

    Returns:
        Configured logger instance
    """
    global _logging_configured

    if log_format is None:
        log_format = "[%(asctime)s] %(levelname)s [%(name)s] - %(message)s"

    # Standard formatter for file logging
    standard_formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Configure root logger only once to avoid duplicates
    root_logger = logging.getLogger()

    # Clear all existing handlers from the root logger
    if not _logging_configured:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set the root logger level
        root_logger.setLevel(level)

        # Add console handler to root logger
        console_handler = logging.StreamHandler(sys.stdout)
        if use_colors:
            console_handler.setFormatter(ColoredFormatter())
        else:
            console_handler.setFormatter(standard_formatter)
        root_logger.addHandler(console_handler)

        # Add file handler to root logger if specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10485760, backupCount=5
            )
            file_handler.setFormatter(standard_formatter)
            root_logger.addHandler(file_handler)

        # Log a separator
        logger = logging.getLogger("ai_engine")
        if use_colors:
            logger.info(f"{COLORS['BOLD']}{'='*80}{COLORS['RESET']}")
            logger.info(f"{COLORS['BOLD']}{'🚀 AI ENGINE STARTING'}{COLORS['RESET']}")
            logger.info(f"{COLORS['BOLD']}{'='*80}{COLORS['RESET']}")

        _logging_configured = True

    # Get the requested logger (will inherit settings from root)
    requested_logger = logging.getLogger(name)
    requested_logger.setLevel(level)

    # Don't add any handlers to non-root loggers to avoid duplication
    return requested_logger


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
    level = config.get("logging", {}).get("level", logging.INFO)
    log_file = config.get("logging", {}).get("file", None)
    logger = setup_logger(level=level, log_file=log_file, log_format=log_format)


def format_message(message, icon=None, color=None, bold=False):
    """
    Format a message with an optional icon and color.

    Args:
        message: The message to format
        icon: Optional emoji icon to prepend
        color: Optional color from COLORS dict
        bold: Whether to make the text bold

    Returns:
        Formatted message string
    """
    formatted = ""

    # Add icon if provided
    if icon:
        formatted += f"{icon} "

    # Add color and bold formatting if provided
    if color and color in COLORS:
        formatted += COLORS[color]
    if bold:
        formatted += COLORS["BOLD"]

    # Add the message
    formatted += message

    # Reset formatting
    if color or bold:
        formatted += COLORS["RESET"]

    return formatted


def load_config(config_path=None) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file, injects API keys from environment variables (GOOGLE_API_KEY, GROQ_API_KEY), and sets up global paths and logging

    Args:
        config_path: Path to the configuration file. If None, uses the default config.yaml

    Returns:
        dict: Configuration dictionary with all settings
    """
    global config
    if config_path is None:
        # Default to the config.yaml located one directory above the current module
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml")
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Inject API keys from environment variables, overriding YAML values
    # ------------------------------------------------------------------
    api_key_cfg: Dict[str, Any] = config.get("api-key", {}) or {}
    google_env_key = os.getenv("GOOGLE_API_KEY")
    if google_env_key:
        api_key_cfg["google"] = google_env_key
    groq_env_key = os.getenv("GROQ_API_KEY")
    if groq_env_key:
        api_key_cfg["groq"] = groq_env_key

    if api_key_cfg:
        config["api-key"] = api_key_cfg

    # Set up paths from config or use defaults
    paths_config = config.get("paths", {})
    global MODELS_DIR, OUTPUT_DIR, ENGINE_LOG_DIR

    # Update global paths if specified in config
    if "models_dir" in paths_config:
        raw_path = paths_config["models_dir"]
        MODELS_DIR = (
            raw_path if os.path.isabs(raw_path) else os.path.abspath(os.path.join(BASE_DIR, raw_path))
        )

    if "output_dir" in paths_config:
        raw_path = paths_config["output_dir"]
        OUTPUT_DIR = (
            raw_path if os.path.isabs(raw_path) else os.path.abspath(os.path.join(BASE_DIR, raw_path))
        )

    if "log_dir" in paths_config:
        raw_path = paths_config["log_dir"]
        ENGINE_LOG_DIR = (
            raw_path if os.path.isabs(raw_path) else os.path.abspath(os.path.join(BASE_DIR, raw_path))
        )

    # Ensure directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ENGINE_LOG_DIR, exist_ok=True)

    # Configure logging
    log_level_str = config.get("logging", {}).get("level", "INFO")
    log_level = getattr(logging, log_level_str)

    # Get log filename from config, but store it in ENGINE_LOG_DIR
    log_filename = config.get("logging", {}).get("file", "ai_engine.log")

    # If log_filename is an absolute path, extract just the filename
    log_filename = os.path.basename(log_filename)

    # Create full log file path in ENGINE_LOG_DIR
    log_file = os.path.join(ENGINE_LOG_DIR, log_filename)

    # Use our setup_logger function which handles duplicate prevention
    global logger
    logger = setup_logger(level=log_level, log_file=log_file)

    # Log the paths being used
    logger.debug(f"Using config file: {config_path}")
    logger.debug(f"Using base directory: {BASE_DIR}")
    logger.debug(f"Using models directory: {MODELS_DIR}")
    logger.debug(f"Using output directory: {OUTPUT_DIR}")
    logger.debug(f"Using log directory: {ENGINE_LOG_DIR}")
    logger.debug(f"Log file: {log_file}")

    return config



def estimate_tokens(text: Union[str, Dict, Any]) -> int:
    """
    Token estimation using strict 4 characters = 1 token rule.
    Handles strings, dictionaries (JSON), and other serializable formats.
    
    Args:
        text: Input text or serializable object to count tokens for
        
    Returns:
        Estimated token count using ceiling(character_count / 4)
    """
    if not isinstance(text, str):
        text = json.dumps(text)
    
    # Count all characters (including spaces and punctuation)
    char_count = len(text)
    
    # Apply 4 chars = 1 token rule with ceiling division
    return (char_count + 3) // 4  # Equivalent to math.ceil(char_count / 4)

def log_token_usage(
    prompt: str, 
    response: str, 
    model_type: str = "generic"
) -> Dict[str, int]:
    """
    Unified token counting using 4 characters = 1 token rule.
    No longer uses exact_counts since we're using generalized counting.
    
    Args:
        prompt: Input prompt text
        response: Model response text
        model_type: LLM provider identifier
        
    Returns:
        Dictionary with token counts and metadata
    """
    token_counts = {
        "input_tokens": estimate_tokens(prompt),
        "output_tokens": estimate_tokens(response),
        "model_type": model_type,
        "counting_method": "4_chars_per_token",
        "is_estimate": True  # Since we're using one consistent method
    }
    token_counts["total_tokens"] = token_counts["input_tokens"] + token_counts["output_tokens"]
    
    return token_counts
