"""
Configuration management for AI models and prompts.
This module centralizes all AI-related configurations to make versioning and updates easier.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


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

# Model configurations
MODEL_CONFIGS = {
    "gemini": {
        "model_name": "gemini-2.0-flash-001",
        "generation_config": {
            "temperature": 0.2,
            "max_output_tokens": 1024
        }
    }
}

# Prompt templates
PROMPTS = {
    "label_workloads": {
        "version": "1.0",
        "output_schema": WorkloadLabelOutput,
        "template": """You are a Kubernetes cluster orchestrator. For each workload, decide whether it should stay in the 'private' cluster (0) or migrate to the 'public' cluster (1).
Decision rules:
- Workloads with high demand (high CPU or memory usage) while the private cluster is overloaded should go to the public cluster.
- If percent_pending is high, consider moving to the public cluster.
- Consider the cluster_load to avoid overloading both the destination cluster and the actual cluster.

For each workload, provide:
1. The decision (0 for private, 1 for public)
2. A brief explanation of why you made this decision based on the workload's characteristics

YOUR RESPONSE MUST STRICTLY FOLLOW THIS JSON SCHEMA:
{{
  "decisions": [0, 1, 0, ...],  // Array of 0s and 1s for each workload, required
  "explanations": [
    "Explanation for workload 1",
    "Explanation for workload 2",
    ...
  ]  // Array of string explanations, required, must have same length as decisions
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
    return MODEL_CONFIGS.get(model_key, {})

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
