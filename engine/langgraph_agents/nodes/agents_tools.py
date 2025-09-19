import os
import json
import logging

from typing import Any, Dict
from dotenv import load_dotenv
from engine.ai_config import get_prompt
from engine.util import load_config, get_logger
from engine.ai_config import get_prompt
from engine.util import load_config, log_token_usage
from langsmith import traceable
from ..tools.tool_percent_pending import pending_percentage
from ..tools.tool_output import final_recommendations
from langchain_google_genai import ChatGoogleGenerativeAI
from engine.client import OpenRouterClient
logger = get_logger("agents")
logger = logging.getLogger(__name__)
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


def get_llm():
    """Initialize and return a model wrapper with `.invoke` and its config."""
    try:
        client = OpenRouterClient()
        config = load_config()
        selected_model = config["ai"].get("selected_model", "gemini")
        model_cfg = config["ai"]["models"].get(selected_model, {})
        if model_cfg is None:
            raise ValueError(f"Model '{selected_model}' not found in configuration.")

        # Map internal model names to OpenRouter model names
        model_mapping = {
            "gemini": "google/gemini-2.0-flash-001",
            "gpt-4": "openai/gpt-4",
            "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
            "llama": "meta-llama/llama-3.1-8b-instruct",
        }

        model_name = model_mapping.get(selected_model, "google/gemini-2.0-flash-001")
        generation_config = model_cfg.get("generation_config", {})

        system_prompt = "You are an expert Kubernetes workload migration advisor. Analyze the provided workloads and make migration decisions."

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

@traceable(name="explanations_node")
def explanations(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invokes the LLM to generate migration recommendations and explanations.
    """
    workloads = state.get("workloads", [])
    clusters = state.get("cluster_info", [])
    
    pending_percentage_result_all_timestamps = state.get("pending_percentage", {})

    valid_timestamps = [key for key in pending_percentage_result_all_timestamps.keys() if key.isdigit()]
    if valid_timestamps:
        latest_timestamp = max(valid_timestamps, key=int)
        pending_percentage_result_latest = pending_percentage_result_all_timestamps.get(latest_timestamp, {})
    else:
        pending_percentage_result_latest = {}

    for w in workloads:
        workload_id = w.get("workload_id")
        w["percent_pending"] = pending_percentage_result_latest.get(workload_id, "0%")
    
    prompt_data = {
        "workloads_json": json.dumps(workloads, indent=2),
        "clusters_json": json.dumps(clusters, indent=2),
        "pending_json": json.dumps(pending_percentage_result_all_timestamps, indent=2)
    }

    prompt = get_prompt("label_workloads", **prompt_data)
    resp = model.invoke(prompt)

    try:
        result = json.loads(resp.content)
    except (json.JSONDecodeError, AttributeError):
        logger.warning("LLM failed to generate a valid JSON. Using fallback.")
        result = {
            "decisions": [0] * len(workloads),
            "overall_explanation": "LLM failed to generate a tool call. Using a fallback decision."
        }

    overall_explanation = result.get("overall_explanation", "No overall explanation provided.")
    decisions_list = result.get("decisions", [])
    workload_explanations = []
    final_decisions = []
    
    for decision_data in decisions_list:
        if isinstance(decision_data, dict):
            final_decisions.append(decision_data.get("decision", 0))
            workload_explanations.append(decision_data.get("reason", "No explanation provided."))
        else:
            label = int(decision_data) if isinstance(decision_data, (int, str)) and str(decision_data).isdigit() else 0
            final_decisions.append(label)
            if label == 1:
                explanation = "Workload recommended for public cluster."
            else:
                explanation = "Workload recommended for private cluster."
            workload_explanations.append(explanation)

    state["final_decisions"] = final_decisions
    state["explanations"] = {
        "overall_explanation": overall_explanation,
        "workload_explanations": workload_explanations
    }
    
    return state