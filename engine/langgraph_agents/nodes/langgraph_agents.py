import os
import json
import pandas as pd
from typing import List, Dict, Union
from dotenv import load_dotenv
from engine.ai_config import get_prompt
from engine.util import load_config, get_logger
from engine.ai_config import get_prompt
from engine.util import load_config
from langsmith import traceable
import re
import logging

logger = get_logger("agents")
from langchain_google_genai import ChatGoogleGenerativeAI

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

    if provider == "google":
        model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=generation_config.get("temperature", 0.1),
            max_output_tokens=generation_config.get("max_output_tokens", 2048),
        )
        return model, generation_config

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=generation_config.get("temperature", 0.1),
            max_tokens=generation_config.get("max_output_tokens", 2048),
        )
        return model, generation_config

    else:
        raise ValueError(f"Provider LLM not support: {provider}")


model, generation_config = get_llm()


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
    """Centraliza chamada ao modelo para ter consistência e token usage"""
    resp = model.invoke(prompt)

    # resp é um ChatResult -> pegar texto
    text = resp.content if hasattr(resp, "content") else str(resp)

    # token usage (se disponível)
    usage = getattr(resp, "response_metadata", {}).get("token_usage", {})
    if usage:
        print("📊 Token usage:", usage)

    return text


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
    prompt = get_prompt("cpu_checker", workloads_json=json.dumps(workloads_list, indent=2))
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
    prompt = get_prompt("mem_checker", workloads_json=json.dumps(workloads_list, indent=2))
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
    prompt = get_prompt("pending_checker", workloads_json=json.dumps(workloads_list, indent=2))
    text = _invoke_model(prompt)
    return _parse_llm_response(text, len(workloads_list))


@traceable(name="decision_agent")
def decision_agent(cpu_votes: List[int], mem_votes: List[int], pending_votes: List[int]) -> List[int]:
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
def explainer_agent(workloads: pd.DataFrame, votes: Dict[str, List[int]], final_decisions: List[int]) -> Dict:
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
                    "decision": final_decisions[i] if i < len(final_decisions) else "unknown",
                    "explanation": "No explanation."
                }
                for i, w in enumerate(workloads_list)
            ]
        }
