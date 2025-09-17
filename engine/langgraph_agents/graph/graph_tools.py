import pandas as pd

from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from ..tools.tool_percent_pending import pending_percentage

class MigrationState(TypedDict):
    workloads: List[Dict]
    explanations: Dict
    pending_tool_result: Dict

# -----------------------
# Functions for each node
# -----------------------

def pending_tool_node(state: dict) -> dict:
    """Node that uses tool pending_by_cluster to analyze workloads."""
    workloads = state.get("workloads", [])
    if not workloads:
        return {"pending_tool_result": {"error": "no workloads provided"}}

    result = pending_percentage.invoke({
        "input_dict": {"data": {"latest": {"workloads": workloads}}}
    })
    
    return {"pending_tool_result": result}


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
    graph = StateGraph(MigrationState)

    graph.add_node("pending_tool", pending_tool_node)
    graph.add_node("ai_explainer", explainer_agent_node)

    # Define as arestas
    graph.add_edge(START, "pending_tool")
    graph.add_edge("pending_tool", "ai_explainer")
    graph.add_edge("ai_explainer", END)
    
    return graph.compile(checkpointer=MemorySaver())