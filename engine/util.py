import os
import sys
import logging
import logging.handlers
from copy import deepcopy
from dotenv import load_dotenv
import yaml
from typing import Optional, Dict, Any, Union, List
import json
from engine.data_types import WorkloadRecommendation


# Default directory paths
# Calculate BASE_DIR as the project root (two levels up from this file)
# Using realpath to resolve any symbolic links
_file_dir = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.realpath(os.path.join(_file_dir, "../../"))
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
_config_loaded = False  # Flag to prevent repeated config loading

# Global variables for configuration and logging
config: Optional[Dict[str, Any]] = None
logger: Optional[logging.Logger] = None


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


def _deep_merge_dicts(
    base: Optional[Dict[str, Any]], overrides: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries without mutating the originals.

    Args:
        base: Base dictionary that provides default values
        overrides: Dictionary whose values take precedence

    Returns:
        dict: Result of merging overrides onto base
    """
    if not base and not overrides:
        return {}
    if not base:
        return deepcopy(overrides) if overrides else {}
    if not overrides:
        return deepcopy(base)

    merged = deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: str) -> Optional[Dict[str, Any]]:
    """
    Safely read a YAML file returning a dictionary or None if not found/invalid.
    """
    if not path:
        return None
    try:
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
            if isinstance(data, dict):
                return data
            logger.warning(f"Config file {path} must contain a YAML object.")
    except FileNotFoundError:
        return None
    except Exception as err:
        logger.warning(f"Failed to read config file {path}: {err}")
    return None


def _resolve_simulator_config_candidates() -> list:
    """
    Build an ordered list of candidate paths for the central simulator config.
    """
    candidates = []
    env_path = os.getenv("SIMULATOR_CONFIG")
    if env_path:
        candidates.append(os.path.abspath(env_path))

    repo_path = os.path.abspath(
        os.path.join(BASE_DIR, os.pardir, "simulator", "data", "config.yaml")
    )
    candidates.append(repo_path)
    return candidates


def _extract_ai_engine_config(
    raw_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Extract the AI Engine specific configuration from the global simulator config.
    """
    if not raw_config:
        return None

    ai_section_keys = {
        "data",
        "ai",
        "paths",
        "logging",
        "server",
        "monitor",
        "actuator",
    }

    # Treat files that only contain AI Engine keys as full AI configs
    raw_keys = set(raw_config.keys())
    if raw_keys and raw_keys.issubset(ai_section_keys):
        return raw_config

    # Otherwise look for dedicated ai-engine sections
    for key in ("ai-engine", "ai_engine"):
        ai_engine_section = raw_config.get(key)
        if isinstance(ai_engine_section, dict):
            return ai_engine_section

    return None


def _apply_flat_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map well-known flat keys (e.g., scheduler_interval) into the nested AI config structure.
    """
    if not config_dict:
        return config_dict

    mapping = {
        "scheduler_interval": ("ai", "scheduler_interval"),
        "timestamp_lookback_seconds": ("ai", "timestamp_lookback_seconds"),
        "workloads_shard_size": ("ai", "workloads_shard_size"),
        "fetch_metrics_immediately": ("ai", "fetch_metrics_immediately"),
        "monitored_duration_sec": ("ai", "monitored_duration_sec"),
        "mode": ("ai", "mode"),
    }

    for key, path in mapping.items():
        if key not in config_dict:
            continue
        target = config_dict
        *parents, leaf = path
        for segment in parents:
            target = target.setdefault(segment, {})
        target[leaf] = config_dict[key]

    return config_dict


def load_config(config_path=None) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file, injects API keys from environment variables (GOOGLE_API_KEY, GROQ_API_KEY), and sets up global paths and logging

    Args:
        config_path: Path to the configuration file. If None, uses the default config.yaml

    Returns:
        dict: Configuration dictionary with all settings
    """
    global config, _config_loaded, logger

    # Return cached config if already loaded
    if _config_loaded and config is not None:
        return config

    # Track the actual config file used for logging
    actual_config_file = None

    # Explicit config path has highest priority (primarily used for tests/tools)
    if config_path is not None:
        resolved_path = os.path.abspath(config_path)
        config = _read_yaml(resolved_path)
        if config is None:
            raise FileNotFoundError(
                f"Unable to load configuration from {resolved_path}"
            )
        actual_config_file = resolved_path
        logger.info(f"Loaded AI Engine configuration from {resolved_path}")
    else:
        local_default_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml")
        )
        local_config = _read_yaml(local_default_path) or {}

        central_config = None
        central_source = None
        central_monitor_interval = None
        for candidate in _resolve_simulator_config_candidates():
            central_data = _read_yaml(candidate)
            ai_engine_config = _extract_ai_engine_config(central_data)
            if ai_engine_config is not None:
                central_config = _apply_flat_overrides(deepcopy(ai_engine_config))
                central_source = candidate
                monitor_block = central_data.get("monitor", {}) if central_data else {}
                central_monitor_interval = monitor_block.get(
                    "collection_interval"
                ) or monitor_block.get("interval")
                break

        if central_config:
            config = _deep_merge_dicts(local_config, central_config)
            source_desc = f"central config: {central_source}"
            actual_config_file = central_source
        else:
            config = deepcopy(local_config)
            source_desc = f"local fallback: {local_default_path}"
            actual_config_file = local_default_path

        if not config:
            raise FileNotFoundError(
                "No AI Engine configuration found. Checked central simulator config and local ai-engine/config.yaml."
            )
        logger.info(f"Loaded AI Engine configuration from {source_desc}")

        if central_monitor_interval is not None:
            monitor_cfg = config.setdefault("monitor", {})
            monitor_cfg["interval"] = central_monitor_interval

    # ------------------------------------------------------------------
    # Inject API keys from environment variables, overriding YAML values
    # ------------------------------------------------------------------
    dotenv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, ".env")
    )
    load_dotenv(dotenv_path)

    # Set up paths from config or use defaults
    paths_config = config.get("paths", {})
    global OUTPUT_DIR, ENGINE_LOG_DIR

    if "output_dir" in paths_config:
        raw_path = paths_config["output_dir"]
        OUTPUT_DIR = (
            raw_path
            if os.path.isabs(raw_path)
            else os.path.abspath(os.path.join(BASE_DIR, raw_path))
        )

    if "log_dir" in paths_config:
        raw_path = paths_config["log_dir"]
        ENGINE_LOG_DIR = (
            raw_path
            if os.path.isabs(raw_path)
            else os.path.abspath(os.path.join(BASE_DIR, raw_path))
        )

    # Ensure directories exist
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
    logger = setup_logger(level=log_level, log_file=log_file)

    # Log the paths being used (only on first load)
    logger.debug(f"Using config file: {actual_config_file}")
    logger.debug(f"Using base directory: {BASE_DIR}")
    logger.debug(f"Using output directory: {OUTPUT_DIR}")
    logger.debug(f"Using log directory: {ENGINE_LOG_DIR}")
    # Mark config as loaded to prevent repeated loading
    _config_loaded = True
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
    prompt: str, response: str, model_type: str = "generic"
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
        "is_estimate": True,  # Since we're using one consistent method
    }
    token_counts["total_tokens"] = (
        token_counts["input_tokens"] + token_counts["output_tokens"]
    )

    return token_counts


def build_workload_recommendations(
    result_df,
    explanations: Dict[str, Any],
    workloads: List[Dict[str, Any]],
    batch_id: int,
) -> List[WorkloadRecommendation]:
    """
    Transform analysis outputs into WorkloadRecommendation objects.

    Args:
        result_df: DataFrame containing analysis results.
        explanations: Dictionary containing explanations for the decisions.
        workloads: List of original workload dictionaries.
        batch_id: The current batch ID for these recommendations.

    Returns:
        List[WorkloadRecommendation]: A list of recommendation objects.
    """
    # Map workload_id -> origin cluster label from original workloads
    origin_by_id = {}
    for w in workloads:
        wid = w.get("workload_id")
        if wid is not None:
            origin_by_id[wid] = w.get("cluster_label", "private")

    explanations_list = (explanations or {}).get("workload_explanations", [])
    recs = []

    for idx, row in result_df.iterrows():
        wid = row.get("workload_id")
        kind = row.get("kind")
        label = int(row.get("label", 0))

        origin_label = origin_by_id.get(wid, "private")
        origin_cluster = 0 if origin_label == "private" else 1
        destination_cluster = label

        # Skip invalid entries if necessary, or handle them.
        # The original code in api.py skipped if destination_cluster == -1,
        # but cli.py didn't seem to explicitly skip in the loop shown (though it might have filtered before).
        # Let's keep the behavior consistent: if it's -1, it's an error/unknown, maybe we should skip or keep.
        # In api.py: if destination_cluster == -1: continue
        if destination_cluster == -1:
            continue

        reason = (
            explanations_list[idx]
            if idx < len(explanations_list)
            else (
                f"Recommended to {'public' if destination_cluster == 1 else 'private'} cluster based on resource analysis"
            )
        )

        recs.append(
            WorkloadRecommendation(
                batch_id=batch_id,
                workload_id=wid,
                kind=kind,
                origin_cluster=origin_cluster,
                destination_cluster=destination_cluster,
                reason=reason,
            )
        )

    return recs
