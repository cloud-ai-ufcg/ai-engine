import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from .util import load_config
from .data_types import WorkloadLabelOutput

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv()


MODEL_CONFIGS: Dict[str, Any] = {}  # Deprecated placeholder

PROMPTS = {
    "label_workloads": {
        "version": "1.0",
        "output_schema": WorkloadLabelOutput,
        "template": """You are a Kubernetes orchestrator. For each workload, decide if it should run in the 'private' cluster (0) or 'public' cluster (1).
Rules:
- If private is overloaded or workload needs high resources, prefer public (1).
- If workload is in public and private has capacity, allow migrating back to private (0).
- Use percent_pending and cluster_load to guide decisions.

Respond with JSON:
{{
  "decisions": [0, 1, ...],
  "explanations": [
    "Short explanation for workload 1",
    "Short explanation for workload 2",
    ...
  ]
}}

Workloads: {workloads_json}
""",
    },
    "cpu_checker": {
        "version": "1.0",
        "template": """You are a CPU usage specialist for Kubernetes clusters.

Your task is to analyze each workload and decide whether it should migrate to the public cluster (1) or stay in the private cluster (0), based **only on CPU usage**.

Guidelines:
- If the workload is in a cluster where CPU usage is high (above 80%), suggest migration (1).
- Otherwise, recommend staying (0).
- Ignore memory and pending pods.

Respond with a JSON list of 0s and 1s only.
Example: [0, 1, 1, 0]

Workloads: {workloads_json}
""",
    },
    "mem_checker": {
        "version": "1.0",
        "template": """You are a memory usage specialist for Kubernetes workloads.

Your task is to analyze each workload and decide whether it should migrate to the public cluster (1) or stay in the private cluster (0), based **only on memory usage**.

Guidelines:
- If the workload demands high memory and the current cluster is overloaded, suggest migration (1).
- Otherwise, recommend staying (0).
- Ignore CPU and pending pods.

Respond with a JSON list of 0s and 1s only.
Example: [1, 0, 1, 0]

Workloads: {workloads_json}
""",
    },
    "pending_checker": {
        "version": "1.0",
        "template": """You are a pending pod specialist in Kubernetes.

Your task is to analyze each workload and decide whether it should migrate to the public cluster 
(1) or stay in the private cluster (0), based **only on the percentage of pending pods**.

Guidelines:
- If the workload has 50% or more of the pods pending, suggest migration (1).
- Otherwise, recommend staying (0).
- Ignore CPU and memory.

Respond with a JSON list of 0s and 1s only.
Example: [0, 1, 1, 0]

Workloads: {workloads_json}
""",
    },
    "decision": {
        "version": "1.0",
        "output_schema": WorkloadLabelOutput,
        "template": """You are a pending pod specialist in kubernetes.

Your task is decided by the lists of votes received in the format: JSON list containing 
only 0s and 1s where (1) indicates to migrate to the public cluster or 0 indicates to 
remain in the private cluster; by the 'pending', 'cpu' and 'memory' nodes. Considering 
that the pending votes must outnumber the other nodes, you must decide whether the 
workload should migrate to the public or remain in the private.


Respond with a JSON list of 0s and 1s only.
Example: [0, 1, 1, 0]
Votes: {workload_json}
""",
    },
}


def get_prompt(prompt_key, **kwargs):
    """
    Get a prompt by key and format it with provided kwargs.

    Args:
        prompt_key: The key for the prompt template
        **kwargs: Format arguments for the prompt template

    Returns:
        Formatted prompt string
    """
    prompt_data = PROMPTS.get(prompt_key, {})
    template = prompt_data.get("template", "")
    return template.format(**kwargs)


def get_agent_mode():
    """
    Get the current agent mode from configuration.

    Returns:
        str: 'single_agent' or 'multi_agent'
    """
    cfg = load_config()
    ai_config = cfg.get("ai", {})

    # Check for new mode field
    mode = ai_config.get("mode")
    if mode:
        return mode

    # Legacy support for multi_agent boolean
    if "multi_agent" in ai_config and isinstance(ai_config["multi_agent"], bool):
        return "multi_agent" if ai_config["multi_agent"] else "single_agent"

    # Default to single_agent
    return "single_agent"


def get_agent_config(mode=None):
    """
    Get the configuration for a specific agent mode.

    Args:
        mode (str, optional): Agent mode. If None, uses current mode from config.

    Returns:
        dict: Configuration for the specified agent mode
    """
    cfg = load_config()
    ai_config = cfg.get("ai", {})

    if mode is None:
        mode = get_agent_mode()

    # Get mode-specific config, falling back to default_config
    mode_config = ai_config.get(mode, {})
    default_config = ai_config.get("default_config", {})

    # Merge with defaults
    config = {**default_config, **mode_config}

    return config


def build_system_prompt_from_config(mode=None):
    """
    Build the system prompt by loading it from a file based on the agent mode.

    Args:
        mode (str, optional): Agent mode ('single_agent' or 'multi_agent').
                             If None, reads from config.

    Returns:
        str: The complete system prompt from the file
    """
    import os

    cfg = load_config()
    ai_config = cfg.get("ai", {})

    if mode is None:
        mode = ai_config.get("mode", "single_agent")

    mode_config = ai_config.get(mode, {})

    selected_prompt = mode_config.get("selected_prompt")

    prompt_file = f"prompts/{selected_prompt}.txt"

    system_prompt = None
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
            logger.info(f"Loaded {mode} prompt from file: {prompt_file}")
        except Exception as e:
            logger.warning(f"Failed to load prompt from {prompt_file}: {e}")

    return system_prompt


def build_cluster_selection_from_config():
    """
    Considers the configuration and builds the list of clusters to be used. Ensure that the clusters
    are strings and strips any leading/trailing whitespace.

    Returns:
        List of cluster labels as strings
    """
    cfg = load_config()
    cluster_selection = cfg.get("ai", {}).get("cluster_selection", {})
    listed_clusters = cluster_selection.get("listed_clusters", [])

    # Ensure clusters are strings and strip whitespace
    normalized_clusters = []
    for cluster in listed_clusters:
        new_cluster = str(cluster).strip()

        if new_cluster:
            normalized_clusters.append(new_cluster)

    return normalized_clusters
