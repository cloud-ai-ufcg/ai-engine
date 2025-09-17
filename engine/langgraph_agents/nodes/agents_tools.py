import os
import json
import logging

from typing import Any, Dict
from dotenv import load_dotenv
from engine.ai_config import get_prompt
from engine.util import load_config, get_logger
from engine.ai_config import get_prompt
from engine.util import load_config
from langsmith import traceable
from ..tools.tool_percent_pending import pending_percentage
from ..tools.tool_output import final_recommendations
from langchain_google_genai import ChatGoogleGenerativeAI

logger = get_logger("agents")
logger = logging.getLogger(__name__)
load_dotenv()

# -----------------------
# Inicializador de LLM via LangChain
# -----------------------
def get_llm():
    config = load_config()
    selected_model = config["ai"].get("selected_model", "gemini")
    model_cfg = config["ai"]["models"].get(selected_model)

    if model_cfg is None:
        raise ValueError(f"Model '{selected_model}' not found in configuration.")

    provider = model_cfg.get("provider", "").lower()
    model_name = model_cfg["model_name"]
    api_key = (
        model_cfg.get("api_key")
        or os.environ.get("GOOGLE_API_KEY")
    )
    generation_config = model_cfg.get("generation_config", {})

    tools = [pending_percentage, final_recommendations] # Add more tools as needed

    if provider == "google":
        model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=generation_config.get("temperature", 0.1),
            max_output_tokens=generation_config.get("max_output_tokens", 2048),
        ).bind_tools(tools=tools)
        return model, generation_config

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=generation_config.get("temperature", 0.1),
            max_tokens=generation_config.get("max_output_tokens", 2048),
        ).bind_tools(tools=tools)
        return model, generation_config

    else:
        raise ValueError(f"Provider LLM not support: {provider}")

model, generation_config = get_llm()

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

    if "tool_output" in state and isinstance(state["tool_output"], dict):
        result = state["tool_output"]
    else:
        logger.warning("LLM failed to generate a tool call. Using fallback.")
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