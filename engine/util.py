import os
import sys
import logging
import logging.handlers
import yaml
from typing import Optional

# ANSI color codes
COLORS = {
    'RESET': '\033[0m',
    'BOLD': '\033[1m',
    'BLUE': '\033[34m',
    'GREEN': '\033[32m',
    'YELLOW': '\033[33m',
    'RED': '\033[31m',
    'MAGENTA': '\033[35m',
    'CYAN': '\033[36m',
    'WHITE': '\033[37m',
    'BG_BLUE': '\033[44m',
    'BG_GREEN': '\033[42m',
    'BG_YELLOW': '\033[43m',
    'BG_RED': '\033[41m',
    'BG_MAGENTA': '\033[45m'
}

# Custom formatter with colors
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: f"{COLORS['BLUE']}[%(asctime)s] {COLORS['BOLD']}DEBUG{COLORS['RESET']}{COLORS['BLUE']} [%(name)s] - %(message)s{COLORS['RESET']}",
        logging.INFO: f"{COLORS['GREEN']}[%(asctime)s] {COLORS['BOLD']}INFO{COLORS['RESET']}{COLORS['GREEN']} [%(name)s] - %(message)s{COLORS['RESET']}",
        logging.WARNING: f"{COLORS['YELLOW']}[%(asctime)s] {COLORS['BOLD']}WARNING{COLORS['RESET']}{COLORS['YELLOW']} [%(name)s] - %(message)s{COLORS['RESET']}",
        logging.ERROR: f"{COLORS['RED']}[%(asctime)s] {COLORS['BOLD']}ERROR{COLORS['RESET']}{COLORS['RED']} [%(name)s] - %(message)s{COLORS['RESET']}",
        logging.CRITICAL: f"{COLORS['BG_RED']}{COLORS['WHITE']}[%(asctime)s] {COLORS['BOLD']}CRITICAL{COLORS['RESET']}{COLORS['BG_RED']}{COLORS['WHITE']} [%(name)s] - %(message)s{COLORS['RESET']}"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


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
    if log_format is None:
        log_format = "[%(asctime)s] %(levelname)s [%(name)s] - %(message)s"

    # Standard formatter for file logging
    standard_formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()

    # Console handler with optional color formatting
    console_handler = logging.StreamHandler(sys.stdout)
    if use_colors:
        console_handler.setFormatter(ColoredFormatter())
    else:
        console_handler.setFormatter(standard_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified) - always use standard formatter for files
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        file_handler.setFormatter(standard_formatter)
        logger.addHandler(file_handler)
        
    # Log a separator when creating a new logger
    if name == "ai_engine" and use_colors:
        logger.info(f"{COLORS['BOLD']}{'='*80}{COLORS['RESET']}")
        logger.info(f"{COLORS['BOLD']}{'🚀 AI ENGINE STARTING'}{COLORS['RESET']}")
        logger.info(f"{COLORS['BOLD']}{'='*80}{COLORS['RESET']}")

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
        formatted += COLORS['BOLD']
        
    # Add the message
    formatted += message
    
    # Reset formatting
    if color or bold:
        formatted += COLORS['RESET']
        
    return formatted


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

    # Configure logging
    log_level = config.get("logging", {}).get("level", "INFO")
    log_file = config.get("logging", {}).get("file", None)

    # Clear existing handlers
    root_logger = logging.getLogger("")
    for handler in root_logger.handlers[:]: 
        root_logger.removeHandler(handler)

    # Basic configuration
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="[%(asctime)s] %(levelname)s [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
    )

    # Add console handler with colored output
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, log_level))
    console.setFormatter(ColoredFormatter())
    logging.getLogger("").addHandler(console)

    # Log a separator to make it easier to see where a new run starts
    logger = get_logger("util")
    logger.info(f"{COLORS['BOLD']}{'='*80}{COLORS['RESET']}")
    logger.info(f"{COLORS['BOLD']}{'🚀 AI ENGINE STARTING'}{COLORS['RESET']}")
    logger.info(f"{COLORS['BOLD']}{'='*80}{COLORS['RESET']}")

    return config
