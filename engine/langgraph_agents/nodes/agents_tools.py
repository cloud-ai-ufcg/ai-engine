import json

from typing import Any, Dict
from dotenv import load_dotenv
from engine.ai_config import (
    build_system_prompt_from_config,
    get_prompt,
    WorkloadLabelOutput,
)
from engine.util import load_config, get_logger
from langsmith import traceable

from engine.client import OpenRouterClient

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

        model = OpenRouterInvokeModel(
            client, model_name, system_prompt, **generation_config
        )
        return model
    except Exception as e:
        logger.error(f"Failed to initialize OpenRouter client/model: {e}")
        raise ValueError(f"Failed to initialize OpenRouter client/model: {e}")


model = get_llm()


# -----------------------
# Langraph agents
# -----------------------
@traceable(name="recommendations_node")
def recommendationsNode(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invokes the LLM to generate migration recommendations and explanations.
    """
    workloads = state.get("workloads", [])
    clusters = state.get("cluster_info", [])
    pending_percentage_result_latest, pending_percentage_result_all_timestamps = _validation_timestamps(state)

    prompt_data = _get_prompt_data(pending_percentage_result_all_timestamps,
                                   pending_percentage_result_latest, clusters,
                                   workloads)
    
    prompt = get_prompt("label_workloads", **prompt_data)
    resp = model.invoke(prompt)

    decisions_list, explanations_list, overall_explanation = _parse_response(resp,
                                                                             "RecommendationsNode")

    final_decisions, workload_explanations = error_handling(decisions_list,
                                                            explanations_list)

    state["decisions"] = final_decisions
    state["explanations"] = {
        "overall_explanation": overall_explanation,
        "workload_explanations": workload_explanations,
    }

    return state

@traceable(name="performance_agent")
def performance_agent(state:Dict[str, Any]) -> Dict[str, Any]:
    workloads = state.get("workloads", [])
    clusters = state.get("cluster_info", [])

    pending_percentage_result_latest, pending_percentage_result_all_timestamps = _validation_timestamps(state)

    prompt_data = _get_prompt_data(pending_percentage_result_all_timestamps,
                                   pending_percentage_result_latest, clusters,
                                   workloads)

    prompt = get_prompt("label_workloads", **prompt_data)
    resp = model.invoke(prompt)

    decisions_list, explanations_list, overall_explanation = _parse_response(resp,
                                                                             "PerformanceAgent")

    final_decisions, workload_explanations = error_handling(decisions_list,
                                                            explanations_list)

    state["decisions"] = final_decisions
    state["explanations"] = {
        "overall_explanation": overall_explanation,
        "workload_explanations": workload_explanations,
    }

    return state

@traceable(name="cost_agent")
def cost_agent(state:Dict[str, Any]) -> Dict[str, Any]:

    workloads = state.get("workloads", [])
    clusters = state.get("cluster_info", [])

    pending_percentage_result_latest, pending_percentage_result_all_timestamps = _validation_timestamps(state)

    prompt_data = _get_prompt_data(pending_percentage_result_all_timestamps,
                                   pending_percentage_result_latest, clusters,
                                   workloads)

    prompt = get_prompt("label_workloads", **prompt_data)
    resp = model.invoke(prompt)

    decisions_list, explanations_list, overall_explanation = _parse_response(resp, "CostAgent")

    final_decisions, workload_explanations = error_handling(decisions_list, explanations_list)

    state["decisions"] = final_decisions
    state["explanations"] = {
        "overall_explanation": overall_explanation,
        "workload_explanations": workload_explanations,
    }

    return state

def _validation_timestamps(state: Dict[str, Any]):
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
    return pending_percentage_result_all_timestamps, pending_percentage_result_latest


def _get_prompt_data(pending_percentage_result_all_timestamps: Any,
                     pending_percentage_result_latest: Any,
                     clusters: Any,
                     workloads:Any):
    for w in workloads:
        workload_id = w.get("workload_id")
        w["percent_pending"] = pending_percentage_result_latest.get(workload_id, "0%")

    prompt_data = {
        "workloads_json": json.dumps(workloads, indent=2),
        "clusters_json": json.dumps(clusters, indent=2),
        "pending_json": json.dumps(pending_percentage_result_all_timestamps, indent=2),
    }
    return prompt_data

def _parse_response(resp: Any, agent):
    try:
        # `chat_structured` may already return a parsed dict; fall back to JSON parse otherwise
        parsed = resp if isinstance(resp, dict) else json.loads(resp)
        wl_output = WorkloadLabelOutput.from_dict(parsed)
        decisions_list = wl_output.decisions
        explanations_list = wl_output.explanations
        overall_explanation = parsed.get(
            "overall_explanation", "Migration decisions generated by LLM"
        )
    except (Exception,):
        # Fallback path when response is not valid or schema fails
        logger.error(
            "LLM failed to generate a valid WorkloadLabelOutput."
        )
        decisions_list = [-1] * len(workloads)
        explanations_list = [
            f"{agent} failed to produce WorkloadLabelOutput"
            for _ in workloads
        ]
        overall_explanation = (
            f"{agent} failed to produce WorkloadLabelOutput"
        )
    return decisions_list, explanations_list, overall_explanation


def error_handling(decisions_list: Any, explanations_list: Any):
    final_decisions: list[int] = []
    workload_explanations: list[str] = []

    for label, expl in zip(decisions_list, explanations_list):
        label_int = -1

        try:
            val = int(label)
            if val in [0, 1]:
                label_int = val
            elif val == -1:
                label_int = -1
            else:
                logger.warning(f"Label value {val} outside of expected range [0, 1, -1]. Setting to -1.")
                label_int = -1
        except (ValueError, TypeError):
            logger.warning(f"Non-numeric label received: {label}. Setting to -1.")
            label_int = -1
            
        final_decisions.append(label_int)
        workload_explanations.append(expl)

    return final_decisions, workload_explanations