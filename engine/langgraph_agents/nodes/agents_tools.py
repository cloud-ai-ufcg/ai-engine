"""Nodes and helpers for LangGraph agents using the OpenRouter-backed LLM."""

import json

from typing import Any, Dict

from dotenv import load_dotenv
from langsmith import traceable

from engine.ai_config import build_system_prompt_from_config
from engine.data_types import WorkloadLabelOutput
from engine.client import OpenRouterClient
from engine.util import get_logger, load_config

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
# Langraph agents
# -----------------------
@traceable(name="recommendations_node")
def recommendations_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invokes the LLM to generate migration recommendations and explanations.
    """
    workloads = state.get("workloads", [])
    clusters = state.get("cluster_info", [])

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

        overall_explanation = "RecommendationsNode preserved original cluster assignments due to invalid LLM response"

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
        "overall_explanation": overall_explanation,
        "workload_explanations": workload_explanations,
    }

    return state
