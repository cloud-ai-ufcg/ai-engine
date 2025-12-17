"""Nodes and helpers for LangGraph agents using the OpenRouter-backed LLM."""

import json

from typing import Any, Dict

from dotenv import load_dotenv
from langsmith import traceable

from engine.ai_config import build_system_prompt_from_config
from engine.data_types import WorkloadLabelOutput
from engine.client import OpenRouterClient
from engine.util import get_logger, load_config
import os

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
        self._client: OpenRouterClient = client
        self._model_name: str = model_name
        self._system_prompt: str = system_prompt
        # Normalize generation config keys to OpenAI chat params
        self._gen_cfg: Dict[str, Any] = {
            "temperature": gen_cfg.get("temperature", 0.1),
            # In config we may store as max_output_tokens; OpenAI param is max_tokens
            "max_tokens": gen_cfg.get("max_output_tokens", 2048),
        }

    def invoke(self, user_prompt: str) -> str:
        """Invoke the model with the given user prompt."""
        return self._client.chat_structured(
            model=self._model_name,
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            schema=WorkloadLabelOutput.model_json_schema(),
            **self._gen_cfg,
        )


def get_llm():
    """Initialize and return a model wrapper with `.invoke` and its config."""
    try:
        client = OpenRouterClient()
        config = load_config()
        selected_model = config["ai"].get("selected_model", "gemini")
        model_cfg = config["ai"]["default_config"]

        if model_cfg is None:
            raise ValueError(f"Model '{selected_model}' not found in configuration.")

        model_name = selected_model
        # Work on a mutable copy to avoid mutating global config inadvertently
        generation_config = dict(model_cfg.get("generation_config", {}))

        # Extract system prompt then remove it so it is not forwarded twice via **kwargs
        system_prompt = build_system_prompt_from_config()

        generation_config.pop("system_prompt", None)
        generation_config.pop("rules", None)

        llm_model = OpenRouterInvokeModel(
            client, model_name, system_prompt, **generation_config
        )
        return llm_model
    except Exception as e:  # pragma: no cover - defensive logging
        logger.error("Failed to initialize OpenRouter client/model: %s", e)
        raise ValueError("Failed to initialize OpenRouter client/model") from e


model = get_llm()


# -----------------------
# Helper Functions
# -----------------------
def _load_agent_prompt(prompt_file_key: str) -> str:
    """
    Load a specific agent prompt from configuration.
    
    Args:
        prompt_file_key: Config key like 'performance_prompt_file', 'cost_prompt_file', etc.
    
    Returns:
        str: The prompt content, or empty string if not found
    """
    try:
        config = load_config()
        multi_agent_config = config.get("ai", {}).get("multi_agent", {})
        prompts_config = multi_agent_config.get("prompts", {})
        
        prompt_name = prompts_config.get(prompt_file_key, "")
        
        if not prompt_name:
            logger.warning(f"No prompt file configured for {prompt_file_key}")
            return ""
        
        prompt_file = f"prompts/{prompt_name}.txt"
        
        if not os.path.exists(prompt_file):
            logger.warning(f"Prompt file not found: {prompt_file}")
            return ""
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_content = f.read().strip()
        
        logger.debug(f"Loaded prompt from {prompt_file}")
        return prompt_content
        
    except Exception as e:
        logger.error(f"Failed to load agent prompt {prompt_file_key}: {e}")
        return ""


# -----------------------
# Langraph agents
# -----------------------
@traceable(name="recommendations_node")
def recommendations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invokes the LLM to generate migration recommendations and explanations.
    Now includes historical context if available.
    """
    workloads = state.get("workloads", [])
    clusters = state.get("cluster_info", [])
    historical_context = state.get("historical_context", "")  # Get from state

    pending_percentage_result_all_timestamps = state.get("pending_percentage", {})

    valid_timestamps = [
        key for key in pending_percentage_result_all_timestamps.keys() if key.isdigit()
    ]
    if valid_timestamps:
        latest_timestamp = max(valid_timestamps, key=int)
        pending_percentage_result_latest = pending_percentage_result_all_timestamps.get(
            latest_timestamp, {}
        )
    else:
        pending_percentage_result_latest = {}

    for w in workloads:
        workload_id = w.get("workload_id")
        w["percent_pending"] = pending_percentage_result_latest.get(workload_id, "0%")

    prompt_data = {
        "workloads_json": json.dumps(workloads, indent=2),
        "clusters_json": json.dumps(clusters, indent=2),
        "pending_json": json.dumps(pending_percentage_result_all_timestamps, indent=2),
        "historical_context": historical_context,  # Pass to prompt
    }

    user_prompt = json.dumps(prompt_data, separators=(',', ':'))

    resp = model.invoke(user_prompt)

    try:
        # `chat_structured` may already return a parsed dict; fall back to JSON parse otherwise
        parsed = resp if isinstance(resp, dict) else json.loads(resp)
        wl_output = WorkloadLabelOutput.from_dict(parsed)
        decisions_list = wl_output.decisions
        explanations_list = wl_output.explanations
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        # Log a small snippet of the raw response (if available) to aid debugging
        try:
            raw_preview = resp[:500] if isinstance(resp, str) else str(resp)[:500]
            logger.error(
                "LLM failed to generate a valid WorkloadLabelOutput: %s. Raw response preview: %s",
                exc,
                raw_preview,
            )
        except Exception:
            logger.error(
                "LLM failed to generate a valid WorkloadLabelOutput and raw response could not be logged: %s",
                exc,
            )

        # Instead of marking all workloads as invalid (-1), preserve the original
        # cluster assignment when available. This avoids treating the entire batch
        # as ignored just because the LLM output was slightly malformed.
        decisions_list = []
        explanations_list = []

        for w in workloads:
            original_cluster_label = w.get("cluster_label", "private")
            if original_cluster_label == "public":
                decision_val = 1
            elif original_cluster_label == "private":
                decision_val = 0
            else:
                # Unknown/absent label - fall back to -1 for this workload only
                decision_val = -1

            decisions_list.append(decision_val)
            explanations_list.append(
                f"Maintaining original cluster ({original_cluster_label}) due to invalid LLM response"
            )

    final_decisions: list[int] = []
    workload_explanations: list[str] = []

    for label, expl in zip(decisions_list, explanations_list):
        label_int = -1

        try:
            val = int(label)

            if val in (0, 1, -1):
                label_int = val
            else:
                logger.warning(
                    "Label value %s outside of expected range [0, 1, -1]. Setting to -1.",
                    val,
                )
        except (ValueError, TypeError):
            logger.warning(
                "Non-numeric label received: %s. Setting to -1.",
                label,
            )

        final_decisions.append(label_int)
        workload_explanations.append(expl)

    state["decisions"] = final_decisions
    state["explanations"] = {
        "workload_explanations": workload_explanations,
    }

    return state

@traceable(name="performance_agent")
def performance_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performance agent analyzes workloads based on performance metrics.
    Uses the performance_agent prompt from configuration.
    Returns ONLY the keys it modifies (decisions, explanations).
    """
    workloads = state.get("workloads", [])
    clusters = state.get("cluster_info", [])
    historical_context = state.get("historical_context", "")
    pending_percentage_result_all_timestamps = state.get("pending_percentage", {})

    _validation_timestamps(pending_percentage_result_all_timestamps, workloads)

    prompt_data = {
        "workloads_json": json.dumps(workloads, indent=2),
        "clusters_json": json.dumps(clusters, indent=2),
        "pending_json": json.dumps(pending_percentage_result_all_timestamps, indent=2),
        "historical_context": historical_context,
    }
    user_prompt = json.dumps(prompt_data, separators=(',', ':'))

    # Load performance-specific system prompt
    performance_system_prompt = _load_agent_prompt("performance_prompt_file")
    if performance_system_prompt:
        client = OpenRouterClient()
        config = load_config()
        selected_model = config["ai"].get("selected_model", "gemini")
        generation_config = config["ai"]["multi_agent"].get("generation_config", {})
        temp_model = OpenRouterInvokeModel(client, selected_model, performance_system_prompt, **generation_config)
        resp = temp_model.invoke(user_prompt)
    else:
        resp = model.invoke(user_prompt)

    decisions_list, explanations_list, overall_explanation = _parse_response(resp, workloads)

    final_decisions, workload_explanations = _error_handling(decisions_list, explanations_list)

    # Return ONLY the keys this node modifies, not 'workloads' or full state
    return {
        "decisions": final_decisions,
        "explanations": {
            "overall_explanation": overall_explanation,
            "workload_explanations": workload_explanations,
        }
    }

@traceable(name="cost_agent")
def cost_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cost agent analyzes workloads based on pricing and cost metrics.
    Uses the cost_agent prompt from configuration.
    Returns ONLY the keys it modifies (cost_decisions, cost_explanations).
    """
    workloads = state.get("workloads", [])
    clusters = state.get("cluster_info", [])
    historical_context = state.get("historical_context", "")
    pending_percentage_result_all_timestamps = state.get("pending_percentage", {})

    _validation_timestamps(pending_percentage_result_all_timestamps, workloads)

    prompt_data = {
        "workloads_json": json.dumps(workloads, indent=2),
        "clusters_json": json.dumps(clusters, indent=2),
        "pending_json": json.dumps(pending_percentage_result_all_timestamps, indent=2),
        "historical_context": historical_context,
    }

    user_prompt = json.dumps(prompt_data, separators=(',', ':'))

    # Load cost-specific system prompt
    cost_system_prompt = _load_agent_prompt("cost_prompt_file")
    if cost_system_prompt:
        client = OpenRouterClient()
        config = load_config()
        selected_model = config["ai"].get("selected_model", "gemini")
        generation_config = config["ai"]["multi_agent"].get("generation_config", {})
        temp_model = OpenRouterInvokeModel(client, selected_model, cost_system_prompt, **generation_config)
        resp = temp_model.invoke(user_prompt)
    else:
        resp = model.invoke(user_prompt)

    decisions_list, explanations_list, overall_explanation = _parse_response(resp, workloads)

    final_decisions, workload_explanations = _error_handling(decisions_list, explanations_list)

    # Return ONLY the keys this node modifies (with 'cost_' prefix to avoid conflicts
    # with performance agent)
    return {
        "cost_decisions": final_decisions,
        "cost_explanations": {
            "overall_explanation": overall_explanation,
            "workload_explanations": workload_explanations,
        }
    }

@traceable(name="consolidator")
def consolidator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Consolidate decisions from performance_agent and cost_agent using LLM.
    Uses the consolidator_agent prompt from configuration.
    
    The consolidator receives decisions and explanations from two agents:
    - performance_agent (considers CPU, memory, pending pods)
    - cost_agent (considers pricing)
    
    The LLM applies its own consolidation logic based on the prompt instructions.
    Returns ONLY the keys it modifies (final_decisions, final_explanations).
    """
    # Get decisions and explanations from performance agent
    performance_decisions = state.get("decisions", [])
    performance_explanations = state.get("explanations", {}).get("workload_explanations", [])

    # Get decisions and explanations from cost agent
    cost_decisions = state.get("cost_decisions", [])
    cost_explanations = state.get("cost_explanations", {}).get("workload_explanations", [])

    # Build prompt with both agents' outputs for LLM to consolidate
    prompt_data = {
        "performance_decisions": performance_decisions,
        "performance_explanations": performance_explanations,
        "cost_decisions": cost_decisions,
        "cost_explanations": cost_explanations,
    }

    user_prompt = json.dumps(prompt_data, separators=(',', ':'))
    
    # Load consolidator-specific system prompt
    consolidator_system_prompt = _load_agent_prompt("consolidator_prompt_file")
    if consolidator_system_prompt:
        client = OpenRouterClient()
        config = load_config()
        selected_model = config["ai"].get("selected_model", "gemini")
        generation_config = config["ai"]["multi_agent"].get("generation_config", {})
        temp_model = OpenRouterInvokeModel(client, selected_model, consolidator_system_prompt, **generation_config)
        resp = temp_model.invoke(user_prompt)
    else:
        resp = model.invoke(user_prompt)

    # Pass empty list for fallback since we don't have workloads in consolidator
    decisions_list, explanations_list, overall_explanation = _parse_response(resp, [])

    final_decisions, workload_explanations = _error_handling(decisions_list, explanations_list)

    logger.info(
        f"Consolidation complete: {len(final_decisions)} workload decisions finalized"
    )

    # Return ONLY the keys this node modifies (with 'final_' prefix for consolidated results)
    return {
        "final_decisions": final_decisions,
        "final_explanations": {
            "overall_explanation": overall_explanation,
            "workload_explanations": workload_explanations,
        }
    }

def _validation_timestamps(pending_percentage_result_all_timestamps: Any = None,
                           workloads: Any = None):
    """Validate and extract latest pending percentage data for workloads."""
    valid_timestamps = [
        key for key in pending_percentage_result_all_timestamps.keys() if key.isdigit()
    ]
    if valid_timestamps:
        latest_timestamp = max(valid_timestamps, key=int)
        pending_percentage_result_latest = pending_percentage_result_all_timestamps.get(
            latest_timestamp, {}
        )
    else:
        pending_percentage_result_latest = {}

    for w in workloads:
        workload_id = w.get("workload_id")
        w["percent_pending"] = pending_percentage_result_latest.get(workload_id, "0%")
    return pending_percentage_result_all_timestamps, pending_percentage_result_latest

def _recover_truncated_json(resp: str, num_workloads: int) -> dict:
    """
    Attempt to recover from truncated JSON response by finding the last valid structure.
    Returns a partial dict with decisions list (truncated) and empty explanations.
    """
    try:
        # Try to find and close incomplete structures
        # Look for the last valid decision in the decisions array
        decisions_match = resp.find('"decisions"')
        if decisions_match == -1:
            return None

        # Find the array opening bracket
        array_start = resp.find('[', decisions_match)
        if array_start == -1:
            return None

        # Try to extract whatever decisions we can parse
        # Find the last complete number followed by comma or closing bracket
        last_valid_idx = array_start
        depth = 0
        for i in range(array_start, len(resp)):
            if resp[i] == '[':
                depth += 1
            elif resp[i] == ']':
                depth -= 1
                if depth == 0:
                    last_valid_idx = i
                    break

        if last_valid_idx == array_start:
            return None

        # Extract decisions substring and try to parse
        decisions_str = resp[array_start:last_valid_idx+1]
        try:
            decisions = json.loads(decisions_str)
            if isinstance(decisions, list) and len(decisions) > 0:
                logger.warning(
                    f"Recovered truncated JSON with {len(decisions)} decisions out of ~{num_workloads}"
                )
                return {
                    "decisions": decisions,
                    "explanations": [f"Decision {i}" for i in range(len(decisions))],
                    "overall_explanation": "Recovered from truncated LLM response"
                }
        except json.JSONDecodeError:
            pass
    except Exception:
        pass
    
    return None


def _parse_response(resp: Any, workloads: Any):
    """Parse LLM response into decisions and explanations,
        with error handling and truncation recovery."""
    overall_explanation = ""
    parsed = None
    wl_output = None
    try:
        # `chat_structured` may already return a parsed dict; fall back to JSON parse otherwise
        parsed = resp if isinstance(resp, dict) else json.loads(resp)
        wl_output = WorkloadLabelOutput.from_dict(parsed)
        decisions_list = wl_output.decisions
        explanations_list = wl_output.explanations
        # Try to extract an overall explanation if present in the parsed dict
        if isinstance(parsed, dict):
            overall_explanation = parsed.get("overall_explanation", "")
        # As a fallback, try attribute access on the pydantic model (if available)
        if not overall_explanation and wl_output is not None:
            overall_explanation = getattr(wl_output, "overall_explanation", "")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        # Log a small snippet of the raw response (if available) to aid debugging
        try:
            raw_preview = resp[:500] if isinstance(resp, str) else str(resp)[:500]
            logger.error(
                "LLM failed to generate a valid WorkloadLabelOutput: %s. " \
                "Raw response preview: %s",
                exc,
                raw_preview,
            )
        except Exception:
            logger.error(
                "LLM failed to generate a valid WorkloadLabelOutput and " \
                "raw response could not be logged: %s",
                exc,
            )

        # Try to recover truncated JSON
        recovered = None
        if isinstance(resp, str) and len(resp) > 100:
            recovered = _recover_truncated_json(resp, len(workloads))
        
        if recovered:
            decisions_list = recovered.get("decisions", [])
            explanations_list = recovered.get("explanations", [])
            overall_explanation = recovered.get("overall_explanation", "")
        else:
            # Fallback: preserve the original cluster assignment
            decisions_list = []
            explanations_list = []

            for w in workloads:
                original_cluster_label = w.get("cluster_label", "private")
                if original_cluster_label == "public":
                    decision_val = 1
                elif original_cluster_label == "private":
                    decision_val = 0
                else:
                    decision_val = -1

                decisions_list.append(decision_val)
                explanations_list.append(
                    f"Maintaining original cluster ({original_cluster_label})\n"
                    f" due to invalid LLM response"
                )
    # Ensure we always return an overall_explanation string (may be empty)
    if not overall_explanation:
        try:
            if isinstance(parsed, dict):
                overall_explanation = parsed.get("overall_explanation", "")
        except Exception:
            overall_explanation = ""

    return decisions_list, explanations_list, overall_explanation


def _error_handling(decisions_list: Any, explanations_list: Any):
    """Handle errors in LLM output and ensure valid decisions and explanations."""
    final_decisions: list[int] = []
    workload_explanations: list[str] = []

    for label, expl in zip(decisions_list, explanations_list):
        label_int = -1

        try:
            val = int(label)

            if val in (0, 1, -1):
                label_int = val
            else:
                logger.warning(
                    "Label value %s outside of expected range [0, 1, -1]. Setting to -1.",
                    val,
                )
        except (ValueError, TypeError):
            logger.warning(
                "Non-numeric label received: %s. Setting to -1.",
                label,
            )

        final_decisions.append(label_int)
        workload_explanations.append(expl)

    return final_decisions, workload_explanations