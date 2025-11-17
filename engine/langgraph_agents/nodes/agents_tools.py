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


_model_instance = None


def get_llm():
    """Initialize and return a model wrapper with `.invoke` and its config."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    
    try:
        client = OpenRouterClient()
        config = load_config()
        ai_cfg = config.get("ai", {})
        
        selected_model = ai_cfg.get("selected_model", "gemini")
        models_cfg = ai_cfg.get("models", {})
        default_cfg = ai_cfg.get("default_config", {})

        # Try to find model config by key first
        model_cfg = models_cfg.get(selected_model)

        # If not found by key, try to find by model_name
        if not model_cfg:
            for key, candidate in models_cfg.items():
                if candidate.get("model_name") == selected_model:
                    model_cfg = candidate
                    selected_model = key
                    break

        # If still not found, fallback to gemini
        if not model_cfg:
            fallback_key = "gemini"
            logger.warning(
                "Model '%s' not found in configuration. Falling back to '%s'.",
                selected_model,
                fallback_key,
            )
            model_cfg = models_cfg.get(fallback_key)
            if not model_cfg:
                raise ValueError(
                    f"Model configuration for '{selected_model}' not found in configuration."
                )

        # Get the actual model_name from the model config, or use selected_model as fallback
        model_name = model_cfg.get("model_name", selected_model)
        
        # Get generation config from model_cfg, with fallback to default_cfg
        generation_config = dict(model_cfg.get("generation_config", {}))
        
        # Fill in missing values from default config
        if "max_tokens" not in generation_config and "max_output_tokens" not in generation_config:
            default_gen_cfg = default_cfg.get("generation_config", {})
            if "max_output_tokens" in default_gen_cfg:
                generation_config["max_output_tokens"] = default_gen_cfg["max_output_tokens"]
        
        # Extract system prompt then remove it so it is not forwarded twice via **kwargs
        system_prompt = build_system_prompt_from_config()

        generation_config.pop("system_prompt", None)
        generation_config.pop("rules", None)

        _model_instance = OpenRouterInvokeModel(
            client, model_name, system_prompt, **generation_config
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
# Langraph agents
# -----------------------
@traceable(name="recommendations_node")
def recommendationsNode(state: Dict[str, Any]) -> Dict[str, Any]:
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

    prompt = get_prompt("label_workloads", **prompt_data)
    resp = model.invoke(prompt)

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
        logger.warning(
            "LLM failed to generate a valid WorkloadLabelOutput. Using fallback."
        )
        decisions_list = [0] * len(workloads)
        explanations_list = [
            "Workload recommended to stay in private cluster (fallback)"
            for _ in workloads
        ]
        overall_explanation = (
            "LLM failed to generate a tool call. Using a fallback decision."
        )

    final_decisions: list[int] = []
    workload_explanations: list[str] = []

    for label, expl in zip(decisions_list, explanations_list):
        label_int = (
            int(label) if isinstance(label, (int, str)) and str(label).isdigit() else 0
        )
        final_decisions.append(label_int)
        workload_explanations.append(expl)

    state["decisions"] = final_decisions
    state["explanations"] = {
        "overall_explanation": overall_explanation,
        "workload_explanations": workload_explanations,
    }

    return state
