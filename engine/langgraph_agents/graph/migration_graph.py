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
    df = pd.DataFrame(state["workloads"])
    votes = cpu_checker(df)
    return {"cpu_votes": votes}

def mem_node(state: dict) -> dict:
    df = pd.DataFrame(state["workloads"])
    votes = mem_checker(df)
    return {"mem_votes": votes}

def pending_node(state: dict) -> dict:
    df = pd.DataFrame(state["workloads"])
    votes = pending_checker(df)
    return {"pending_votes": votes}

def decision_node(state: dict) -> dict:
    df = pd.DataFrame(state["workloads"])
    explanations = explainer_agent(
        df,
        {
            "cpu": state["cpu_votes"],
            "mem": state["mem_votes"],
            "pending": state["pending_votes"]
        },
        state["final_decisions"]
    )
    return {"explanations": explanations}

def explainer_node(state: dict) -> dict:
    df = pd.DataFrame(state["workloads"])
    state["explanations"] = explainer_agent(
        df,
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

    # Parallel entry points
    graph.set_entry_point("split")
    graph.add_node("split", lambda state: state)

    # graph.set_entry_point("cpu")

    # graph.add_edge("cpu", "mem")
    # graph.add_edge("mem", "pending")
    # graph.add_edge("pending", "decision")

    graph.add_edge("split", "cpu")
    graph.add_edge("split", "mem")
    graph.add_edge("split", "pending")


    graph.add_edge("cpu", "decision")
    graph.add_edge("mem", "decision")
    graph.add_edge("pending", "decision")

    graph.add_edge("decision", "explainer")
    graph.add_edge("explainer", END)

    return graph.compile(checkpointer=MemorySaver())

# -----------------------
# Execução
# -----------------------
def run_migration_pipeline(workloads_df):
    if isinstance(workloads_df, list):
        workloads_df = pd.DataFrame(workloads_df)
    initial_state = {
        "workloads": workloads_df.to_dict(orient="records"),
        "cpu_votes": [],
        "mem_votes": [],
        "pending_votes": [],
        "final_decisions": [],
        "explanations": {}
    }
    graph = create_migration_graph()
    final_state = graph.invoke(initial_state)
    result_df = pd.DataFrame(final_state["workloads"]).copy()
    result_df["label"] = final_state["final_decisions"]
    explanations = final_state["explanations"]
    return result_df, explanations