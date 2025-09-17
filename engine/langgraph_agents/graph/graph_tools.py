import pandas as pd
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from ..tools.tool_percent_pending import pending_percentage

from ..nodes import (
    explanations
)

# -----------------------
# Functions for each node
# -----------------------

def explainer_node(state: dict) -> dict:
    """Generate explanations for the final decisions made.
    
    Args:
        state (dict): Current state containing workloads and final decisions.
    
    Returns:
        Dict: Updated state with explanations.
    """
    df = pd.DataFrame(state["workloads"])
    state["explanations"] = explanations(
        df,
        {
            "cpu": state["cpu_votes"],
            "mem": state["mem_votes"],
            "pending": state["pending_votes"]
        },
        state["final_decisions"]
    )
    return state

def prepare_pending_data(state: dict) -> dict:
    workloads = state.get("workloads", [])
    return {"data": {"latest": {"workloads": workloads}}}

# -----------------------
# Graph Creation
# -----------------------

def create_migration_graph():
    """Create a migration graph based on configuration.
    
    Returns:
        StateGraph: Configured state graph for migration analysis.
    """
    graph = StateGraph(dict)

    graph.add_node("prepare_data", prepare_pending_data)
    graph.add_node("pending_percentage", pending_percentage.invoke)
    graph.add_node("explanations", explanations)

    graph.add_edge(START, "prepare_data")
    graph.add_edge("prepare_data", "pending_percentage")
    graph.add_edge("pending_percentage", "explanations")
    graph.add_edge("explanations", END)
    
    return graph.compile(checkpointer=MemorySaver())