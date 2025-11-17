import json
import re
from typing import Dict, List, Union

import pandas as pd
from dotenv import load_dotenv
from langsmith import traceable

from engine.ai_config import get_prompt
from engine.client import OpenRouterClient
from engine.util import get_logger, load_config, log_token_usage


logger = get_logger("langgraph_agents")

load_dotenv()


class OpenRouterInvokeModel:
    """
    Lightweight wrapper exposing a LangChain-like `.invoke(input)` interface,
    backed by our OpenRouterClient (OpenAI compatible).
    """

    def __init__(
        self, client: OpenRouterClient, model_name: str, system_prompt: str, **gen_cfg
    ):
        self._client = client
        self._model_name = model_name
        self._system_prompt = system_prompt
        # Normalize generation config keys to OpenAI chat params
        self._gen_cfg = {
            "temperature": gen_cfg.get("temperature", 0.1),
            # In config we may store as max_output_tokens; OpenAI param is max_tokens
            "max_tokens": gen_cfg.get("max_output_tokens", 2048),
        }

    def invoke(self, user_prompt: str) -> str:
        return self._client.chat(
            model=self._model_name,
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            **self._gen_cfg,
        )


_model_instance = None


def get_llm():
    """Initialize and return a model wrapper with `.invoke` and its config."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    
    try:
        client = OpenRouterClient()
        cfg = load_config()
        ai_cfg = cfg.get("ai", {})

        selected_model = ai_cfg.get("selected_model", "gemini")
        models_cfg: Dict[str, Dict] = ai_cfg.get("models", {})

        model_cfg = models_cfg.get(selected_model)

        if not model_cfg:
            for key, candidate in models_cfg.items():
                if candidate.get("model_name") == selected_model:
                    model_cfg = candidate
                    selected_model = key
                    break

        if not model_cfg:
            fallback_key = "gemini"
            logger.warning(
                "Model '%s' not found in configuration. Falling back to '%s'.",
                selected_model,
                fallback_key,
            )
            model_cfg = models_cfg.get(fallback_key)
            selected_model = fallback_key

        if not model_cfg:
            raise ValueError(
                f"Model configuration for '{selected_model}' not found in configuration."
            )

        model_name = model_cfg.get("model_name", selected_model)

        generation_config = dict(model_cfg.get("generation_config", {}))

        if "max_tokens" not in generation_config and "max_output_tokens" not in generation_config:
            default_cfg = (
                ai_cfg.get("default_config", {})
                .get("generation_config", {})
            )
            if "max_output_tokens" in default_cfg:
                generation_config["max_output_tokens"] = default_cfg["max_output_tokens"]

        system_prompt = generation_config.get("system_prompt")
        if not system_prompt:
            default_cfg = (
                ai_cfg.get("default_config", {})
                .get("generation_config", {})
            )
            system_prompt = default_cfg.get(
                "system_prompt",
                "You are an expert Kubernetes workload migration advisor. Analyze the provided workloads and make migration decisions.",
            )

        _model_instance = OpenRouterInvokeModel(
            client,
            model_name,
            system_prompt,
            **generation_config,
        )
        return _model_instance
    except Exception as e:
        logger.error(f"Failed to initialize OpenRouter client/model: {e}")
        raise ValueError(f"Failed to initialize OpenRouter client/model: {e}")


def get_model():
    """Lazy getter for the model instance."""
    return get_llm()


# For backward compatibility, provide a property-like access
# but don't initialize until first use
class _LazyModel:
    def invoke(self, prompt: str):
        return get_model().invoke(prompt)


model = _LazyModel()


# -----------------------
# Helper Functions
# -----------------------
def normalize_votes(votes: List[int], expected_length: int) -> List[int]:
    """Normalize a list of votes to ensure it matches the expected length.
    Args:
        votes (List[int]): List of votes (0 or 1).
        expected_length (int): The expected number of votes.

    Returns:
        List[int]: Normalized list of votes with the expected length.
    """
    if len(votes) < expected_length:
        votes.extend([0] * (expected_length - len(votes)))
    elif len(votes) > expected_length:
        votes = votes[:expected_length]
    return votes


def _parse_llm_response(response: str, expected_length: int) -> List[int]:
    """Parse LLM response to extract votes as a list of integers (0 or 1).

    Args:
        response (str): The raw response from the LLM.
        expected_length (int): The expected number of votes.

    Returns:
        List[int]: A list of votes (0 or 1) normalized to the expected length
    """
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict) and "decisions" in parsed:
            votes = parsed["decisions"]
        elif isinstance(parsed, list):
            votes = parsed
        else:
            votes = [int(c) for c in str(parsed) if c in "01"]
    except Exception:
        votes = [int(c) for c in response if c in "01"]

    return normalize_votes(votes, expected_length)


def _invoke_model(prompt: str) -> str:
    """Centraliza chamada ao modelo OpenRouter para ter consistência"""
    try:
        # Use the LangChain-like interface with `.invoke`
        response_text = model.invoke(prompt)

        # Token accounting (estimate) for observability
        token_counts = log_token_usage(prompt, response_text, model_type="openrouter")
        logger.info(
            f"Token usage (estimate): input={token_counts['input_tokens']}, output={token_counts['output_tokens']}, total={token_counts['total_tokens']}"
        )

        logger.debug(f"OpenRouter response: {response_text}")
        return response_text
    except Exception as e:
        logger.error(f"Error calling OpenRouter: {e}")
        raise


# -----------------------
# Langraph agents
# -----------------------
@traceable(name="cpu_checker")
def cpu_checker(
    workloads: Union[list, "pd.DataFrame"],
) -> List[int]:
    """Check CPU usage of workloads and return migration votes.
    Args:
        workloads (Union[list, pd.DataFrame]): List or DataFrame of workload items.
    Returns:
        List[int]: List of votes (0 or 1) for each workload.
    """
    if isinstance(workloads, list):
        workloads = pd.DataFrame(workloads)
    workloads_list = workloads.to_dict(orient="records")
    prompt = get_prompt(
        "cpu_checker", workloads_json=json.dumps(workloads_list, indent=2)
    )
    text = _invoke_model(prompt)
    return _parse_llm_response(text, len(workloads_list))


@traceable(name="mem_checker")
def mem_checker(
    workloads: Union[list, "pd.DataFrame"],
) -> List[int]:
    """Check Memory usage of workloads and return migration votes.

    Args:
        workloads (Union[list, pd.DataFrame]): List or DataFrame of workload items.

    Returns:
        List[int]: List of votes (0 or 1) for each workload.
    """

    if isinstance(workloads, list):
        workloads = pd.DataFrame(workloads)
    workloads_list = workloads.to_dict(orient="records")
    prompt = get_prompt(
        "mem_checker", workloads_json=json.dumps(workloads_list, indent=2)
    )
    text = _invoke_model(prompt)
    return _parse_llm_response(text, len(workloads_list))


@traceable(name="pending_checker")
def pending_checker(
    workloads: Union[list, "pd.DataFrame"],
) -> List[int]:
    """Check Pending status of workloads and return migration votes.

    Args:
        workloads (Union[list, pd.DataFrame]): List or DataFrame of workload items.

    Returns:
        List[int]: List of votes (0 or 1) for each workload.
    """
    if isinstance(workloads, list):
        workloads = pd.DataFrame(workloads)
    workloads_list = workloads.to_dict(orient="records")
    prompt = get_prompt(
        "pending_checker", workloads_json=json.dumps(workloads_list, indent=2)
    )
    text = _invoke_model(prompt)
    return _parse_llm_response(text, len(workloads_list))


@traceable(name="decision_agent")
def decision_agent(
    cpu_votes: List[int], mem_votes: List[int], pending_votes: List[int]
) -> List[int]:
    """Make final migration decisions based on votes from CPU, Memory, and Pending agents.

    Args:
        cpu_votes (List[int]): Votes from CPU checker.
        mem_votes (List[int]): Votes from Memory checker.
        pending_votes (List[int]): Votes from Pending checker.

    Returns:
        List[int]: Final migration decisions (0 or 1) for each workload.
    """
    votes = {"cpu": cpu_votes, "mem": mem_votes, "pending": pending_votes}
    prompt = get_prompt("decision", workload_json=json.dumps(votes, indent=2))
    text = _invoke_model(prompt)
    return _parse_llm_response(text, len(pending_votes))


@traceable(name="explainer_agent")
def explainer_agent(
    workloads: pd.DataFrame, votes: Dict[str, List[int]], final_decisions: List[int]
) -> Dict:
    """Generate explanations for migration decisions based on votes and workload characteristics.

    Args:
        workloads (pd.DataFrame): DataFrame of workload items.
        votes (Dict[str, List[int]]): Dictionary of votes from different agents.
        final_decisions (List[int]): Final migration decisions for each workload.

    Returns:
        Dict: Explanation of decisions including per-workload explanations.
    """
    workloads_list = workloads.to_dict(orient="records")
    prompt = f"""
You are a Kubernetes systems expert.

Briefly explain (in one sentence per workload) the decision to migrate (1) or maintain (0),
based on the experts' votes (CPU, Memory, Pending) and the characteristics of the workload.

Answer in JSON format:
{{
  "explanation": "General decision strategy",
  "workload_explanations": [
    {{
      "workload_id": "...",
      "decision": 1,
      "explanation": "..."
    }},
    ...
  ]
}}
Workloads:
{json.dumps(workloads_list, indent=2)}

Votes:
CPU: {votes["cpu"]}
Memory: {votes["mem"]}
Pending: {votes["pending"]}

Final decisions: {final_decisions}
"""
    try:
        text = _invoke_model(prompt)
        if not text:
            raise ValueError("Empty response from LLM")

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            return parsed
        else:
            raise ValueError("No JSON found in response")

    except Exception as e:

        logger.error("❌ Error processing explanation: %s", e)
        return {
            "explanation": f"Error generating explanation: {str(e)}",
            "workload_explanations": [
                {
                    "workload_id": w.get("workload_id", f"#{i}"),
                    "decision": (
                        final_decisions[i] if i < len(final_decisions) else "unknown"
                    ),
                    "explanation": "No explanation.",
                }
                for i, w in enumerate(workloads_list)
            ],
        }
