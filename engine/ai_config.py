import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from .util import load_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv()


class WorkloadLabelOutput(BaseModel):
    """Structured output for workload labeling"""

    decisions: List[int] = Field(
        description="Array of decisions where 0=private, 1=public for each workload"
    )
    explanations: List[str] = Field(
        description="Array of explanations for each decision"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkloadLabelOutput":
        """Create instance from dictionary"""
        return cls(**data)

    def validate_output(self) -> bool:
        """Validate that decisions and explanations have matching lengths"""
        return len(self.decisions) == len(self.explanations)


class WorkloadRecommendation(BaseModel):
    """Structured output for workload recommendations"""

    batch_id: int | None = None
    workload_id: str
    kind: str
    origin_cluster: int
    destination_cluster: int
    reason: str  # Explanation for the decision


MODEL_CONFIGS: Dict[str, Any] = {}  # Deprecated placeholder

PROMPTS = {
    "label_workloads": {
        "version": "1.0",
        "output_schema": WorkloadLabelOutput,
        "template": """You are a Kubernetes orchestrator. For each workload, decide if it should run in the 'private' cluster (0) or 'public' cluster (1).

{historical_context}

Rules:
- If private is overloaded or workload needs high resources, prefer public (1).
- If workload is in public and private has capacity, allow migrating back to private (0).
- Use percent_pending and cluster_load to guide decisions.
- Consider the historical migration patterns to maintain consistency and avoid unnecessary migrations.
- If a workload was recently migrated, prefer stability unless there's a compelling reason to migrate again.

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


def get_model_config(model_key="gemini"):
    """
    Get model configuration by key.

    Args:
        model_key: The key for the model configuration

    Returns:
        Dictionary with model configuration
    """
    cfg = load_config()
    return cfg.get("ai", {}).get("models", {}).get(model_key, {})


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


def build_system_prompt_from_config():
    cfg = load_config()
    model_config = cfg.get("ai", {}).get("default_config", {})
    system_prompt = model_config.get(
        "system_prompt",
        "You are an expert Kubernetes workload migration advisor. Analyze the provided workloads and make migration decisions.",
    )
    rules = model_config.get("rules", [])
    for rule in rules:
        system_prompt += f"\n- {rule}"
    return system_prompt
