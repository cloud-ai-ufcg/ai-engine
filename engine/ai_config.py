"""
Configuration management for AI models and prompts.
This module centralizes all AI-related configurations to make versioning and updates easier.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .util import load_config


# Output structure definitions
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
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkloadLabelOutput':
        """Create instance from dictionary"""
        return cls(**data)
    
    def validate_output(self) -> bool:
        """Validate that decisions and explanations have matching lengths"""
        return len(self.decisions) == len(self.explanations)


MODEL_CONFIGS: Dict[str, Any] = {}  # Deprecated placeholder

# Prompt templates
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
"""
    }
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
