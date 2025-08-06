import os
import json
from typing import List, Dict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from utils.ai_config import get_groq_llm, get_prompt 
import re

load_dotenv()

llm = get_groq_llm()

def normalize_votes(votes: List[int], expected_length: int) -> List[int]:
    if len(votes) < expected_length:
        votes.extend([0] * (expected_length - len(votes)))
    elif len(votes) > expected_length:
        votes = votes[:expected_length]
    return votes

def cpu_checker(workloads: List[Dict]) -> List[int]:
    prompt = get_prompt("cpu_checker", workloads_json=json.dumps(workloads, indent=2))
    response = llm.invoke(prompt).content
    try:
        parsed = json.loads(response)
        decisions = parsed.get("decisions", [])
    except Exception:
        decisions = [int(c) for c in response if c in "01"]
    
    return normalize_votes(decisions, len(workloads))

def mem_checker(workloads: List[Dict]) -> List[int]:
    prompt = get_prompt("mem_checker", workloads_json=json.dumps(workloads, indent=2))
    response = llm.invoke(prompt).content
    
    try:
        parsed = json.loads(response)
        decisions = parsed.get("decisions", [])
    except Exception:
        decisions = [int(c) for c in response if c in "01"]
    return normalize_votes(decisions, len(workloads))

def pending_checker(workloads: List[Dict]) -> List[int]:
    prompt = get_prompt("pending_checker", workloads_json=json.dumps(workloads, indent=2))
    response = llm.invoke(prompt).content
    
    try:
        parsed = json.loads(response)
        decisions = parsed.get("decisions", [])
    except Exception:
        decisions = [int(c) for c in response if c in "01"]
    return normalize_votes(decisions, len(workloads))

def decision_agent(cpu_votes: List[int], mem_votes: List[int], pending_votes: List[int]) -> List[int]:
    decisions = []
    votes = {
        "cpu": cpu_votes,
        "mem": mem_votes,
        "pending": pending_votes
    }
    prompt = get_prompt("decision", workload_json=json.dumps(votes))
    response = llm.invoke(prompt).content

    try:
        parsed = json.loads(response)
        decisions = parsed.get("decisions", [])
    except Exception:
        decisions = [int(c) for c in response if c in "01"]

    return normalize_votes(decisions, len(pending_votes))


def explainer_agent(workloads: List[Dict], votes: Dict[str, List[int]], final_decisions: List[int]) -> Dict:
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
{json.dumps(workloads, indent=2)}

Votes:
CPU: {votes["cpu"]}
Memory: {votes["mem"]}
Pending: {votes["pending"]}

Final decisions: {final_decisions}
"""

    try:
        response = llm.invoke(prompt).content.strip()
        
        if not response:
            raise ValueError("Empty response from llm")

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed = json.loads(json_str)
            return parsed
        else:
            raise ValueError("No JSON found in response")

    except Exception as e:
        print("❌ Error processing explanation: ", e)
        return {
            "explanation": f"Error generating explanation: {str(e)}",
            "workload_explanations": [
                {
                    "workload_id": w.get("workload_id", f"#{i}"),
                    "decision": final_decisions[i] if i < len(final_decisions) else "unknown",
                    "explanation": "No explanation."
                }
                for i, w in enumerate(workloads)
            ]
        }
