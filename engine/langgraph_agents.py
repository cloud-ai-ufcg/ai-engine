from typing import List, Dict
import os
from dotenv import load_dotenv
import google.generativeai as genai
from ai_config import get_model_config

get_model_config("groq")

def decision_agent(cpu_votes: List[int], mem_votes: List[int], pending_votes: List[int]) -> List[int]:
    """
    Combines the votes from the specialist agents and decides the final migration outcome.
    Default rule: if two or more agents vote 1, then migrate (1); otherwise, do not migrate (0).
    """
    decisions = []

    for i in range(len(cpu_votes)):
        votes = [cpu_votes[i], mem_votes[i], pending_votes[i]]
        if sum(votes) >= 2:
            decisions.append(1)  # Migrar
        else:
            decisions.append(0)  # Não migrar

    return decisions
