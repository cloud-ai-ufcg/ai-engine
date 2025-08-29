import os
import json
import pandas as pd
from typing import List, Dict, Union
from dotenv import load_dotenv
from engine.ai_config import get_prompt, get_model_config
from engine.util import load_config
import google.generativeai as genai
from openai import OpenAI
import re

load_dotenv()

def get_llm():
    config = load_config()
    selected_model = config["ai"].get("selected_model", "gemini")
    model_cfg = config["ai"]["models"].get(selected_model)

    if model_cfg is None:
        raise ValueError(f"Modelo '{selected_model}' não encontrado")

    provider = model_cfg.get("provider", "").lower()
    model_name = model_cfg["model_name"]
    api_key = model_cfg.get("api_key") or os.environ.get("GOOGLE_API_KEY")
    generation_config = model_cfg.get("generation_config", {})

    if provider == "google":
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        return model, generation_config

    elif provider == "openai":
        from openai import OpenAI
        model = OpenAI(api_key=api_key)
        return model, generation_config
    
    else:
        raise ValueError(f"Provedor LLM não suportado: {provider}")

    
model, generation_config = get_llm()


# -----------------------
# Utilitários
# -----------------------
def normalize_votes(votes: List[int], expected_length: int) -> List[int]:
    """Garante que a lista de votos tenha exatamente o tamanho esperado."""
    if len(votes) < expected_length:
        votes.extend([0] * (expected_length - len(votes)))
    elif len(votes) > expected_length:
        votes = votes[:expected_length]
    return votes


def _parse_llm_response(response: str, expected_length: int) -> List[int]:
    """Tenta extrair lista de 0/1 de uma resposta do LLM, com múltiplos fallbacks."""
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


# -----------------------
# Agentes LangGraph
# -----------------------
def cpu_checker(
    workloads: Union[list, "pd.DataFrame"],
) -> List[int]:
    # Backup: caso venha lista, converter
    if isinstance(workloads, list):
        workloads = pd.DataFrame(workloads)

    workloads_list = workloads.to_dict(orient="records")
    prompt = get_prompt("", workloads_json=json.dumps(workloads_list, indent=2))
    response = model.generate_content(prompt, generation_config=generation_config)
    text = response.text
    return _parse_llm_response(text, len(workloads_list))


def mem_checker(
    workloads: Union[list, "pd.DataFrame"],
) -> List[int]:
    # Backup: caso venha lista, converter
    if isinstance(workloads, list):
        workloads = pd.DataFrame(workloads)

    workloads_list = workloads.to_dict(orient="records")
    prompt = get_prompt("mem_checker", workloads_json=json.dumps(workloads_list, indent=2))
    response = model.generate_content(prompt, generation_config=generation_config)
    text = response.text
    return _parse_llm_response(text, len(workloads_list))


def pending_checker(
    workloads: Union[list, "pd.DataFrame"],
) -> List[int]:
    # Backup: caso venha lista, converter
    if isinstance(workloads, list):
        workloads = pd.DataFrame(workloads)

    workloads_list = workloads.to_dict(orient="records")
    prompt = get_prompt("pending_checker", workloads_json=json.dumps(workloads_list, indent=2))
    response = model.generate_content(prompt, generation_config=generation_config)
    text = response.text
    return _parse_llm_response(text, len(workloads_list))


def decision_agent(cpu_votes: List[int], mem_votes: List[int], pending_votes: List[int]) -> List[int]:
    votes = {"cpu": cpu_votes, "mem": mem_votes, "pending": pending_votes}
    prompt = get_prompt("decision", workload_json=json.dumps(votes, indent=2))
    response = model.generate_content(prompt, generation_config=generation_config)
    text = response.text
    return _parse_llm_response(text, len(pending_votes))


def explainer_agent(workloads: pd.DataFrame, votes: Dict[str, List[int]], final_decisions: List[int]) -> Dict:
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
        response = model.generate_content(prompt, generation_config=generation_config)
        text = response.text
        if not text:
            raise ValueError("Empty response from LLM")

        # Pega apenas o JSON da resposta
        match = re.search(r"\{.*\}", text, re.DOTALL) 
        if match:
            parsed = json.loads(match.group(0))
            return parsed
        else:
            raise ValueError("No JSON found in response")

    except Exception as e:
        print("❌ Error processing explanation:", e)
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
