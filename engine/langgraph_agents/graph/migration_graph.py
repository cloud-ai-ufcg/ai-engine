import pandas as pd
from typing import Dict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import uuid

from ..nodes import (
    cpu_checker,
    mem_checker,
    pending_checker,
    decision_agent,
    explainer_agent
)

# -----------------------
# Funções de nós
# -----------------------
def cpu_node(state: dict) -> dict:
    # Garantir que workloads seja DataFrame
    if isinstance(state["workloads"], list):
        state["workloads"] = pd.DataFrame(state["workloads"])
    state["cpu_votes"] = cpu_checker(state["workloads"])
    return state

def mem_node(state: dict) -> dict:
    if isinstance(state["workloads"], list):
        state["workloads"] = pd.DataFrame(state["workloads"])
    state["mem_votes"] = mem_checker(state["workloads"])
    return state

def pending_node(state: dict) -> dict:
    if isinstance(state["workloads"], list):
        state["workloads"] = pd.DataFrame(state["workloads"])
    state["pending_votes"] = pending_checker(state["workloads"])
    return state

def decision_node(state: dict) -> dict:
    state["final_decisions"] = decision_agent(
        state["cpu_votes"],
        state["mem_votes"],
        state["pending_votes"]
    )
    return state

def explainer_node(state: dict) -> dict:
    state["explanations"] = explainer_agent(
        state["workloads"],
        {
            "cpu": state["cpu_votes"],
            "mem": state["mem_votes"],
            "pending": state["pending_votes"]
        },
        state["final_decisions"]
    )
    return state

# -----------------------
# Criação do grafo
# -----------------------
def create_migration_graph():
    graph = StateGraph(dict)

    graph.add_node("cpu", cpu_node)
    graph.add_node("mem", mem_node)
    graph.add_node("pending", pending_node)
    graph.add_node("decision", decision_node)
    graph.add_node("explainer", explainer_node)

    graph.set_entry_point("cpu")
    graph.add_edge("cpu", "mem")
    graph.add_edge("mem", "pending")
    graph.add_edge("pending", "decision")
    graph.add_edge("decision", "explainer")
    graph.add_edge("explainer", END)

    return graph.compile(checkpointer=MemorySaver())

# -----------------------
# Execução
# -----------------------
def run_migration_pipeline(workloads_df):
    # Padronizar logo no começo
    if isinstance(workloads_df, list):
        workloads_df = pd.DataFrame(workloads_df)
    
    initial_state = {
        "workloads": workloads_df,
        "cpu_votes": [],
        "mem_votes": [],
        "pending_votes": [],
        "final_decisions": [],
        "explanations": {}
    }
    conversation_id = str(uuid.uuid4()) 
    config = {
        "configurable": {
            "thread_id": conversation_id
        }
    }
    app = create_migration_graph()
    final_state = app.invoke(initial_state, config=config)

    result_df = final_state["workloads"].copy()
    result_df["label"] = final_state["final_decisions"]
    explanations = final_state["explanations"]

    return result_df, explanations