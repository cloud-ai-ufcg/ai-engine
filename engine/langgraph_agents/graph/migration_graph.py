import os
import pandas as pd
import yaml

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from ..nodes import (
    cpu_checker,
    mem_checker,
    pending_checker,
    decision_agent,
    explainer_agent
)

# -----------------------
# Functions for each node
# -----------------------
def cpu_node(state: dict) -> dict:
    """Process CPU information skipping workloads.
    
    Args:
        state (dict): Current state containing workloads.
    
    Returns:
        Dict: Updated state with CPU votes.
    """ 
    df = pd.DataFrame(state["workloads"])
    state["cpu_votes"] = cpu_checker(df)
    return state

def mem_node(state: dict) -> dict:
    """Process Memory information, skipping workloads.

    Args:
        state (dict): Current state containing workloads.

    Returns:
        Dict: Updated state with Memory votes.
    """
    df = pd.DataFrame(state["workloads"])
    state["mem_votes"] = mem_checker(df)
    return state

def pending_node(state: dict) -> dict:
    """Process Pending information, skipping workloads.

    Args:
        state (dict): Current state containing workloads.

    Returns:
        Dict: Updated state with Pending votes.
    """
    df = pd.DataFrame(state["workloads"])
    state["pending_votes"] = pending_checker(df)
    return state

def decision_node(state: dict) -> dict:
    """Make final decisions based on votes from previous nodes.
    
    Args:
        state (dict): Current state containing votes.
    
    Returns:
        Dict: Updated state with final decisions.
    """
    state["final_decisions"] = decision_agent(
        state["cpu_votes"],
        state["mem_votes"],
        state["pending_votes"]
    )
    return state

def explainer_node(state: dict) -> dict:
    """Generate explanations for the final decisions made.
    
    Args:
        state (dict): Current state containing workloads and final decisions.
    
    Returns:
        Dict: Updated state with explanations.
    """
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
# Graph Creation
# -----------------------

_node_function_map = {
    "cpu": cpu_node,
    "mem": mem_node,
    "pending": pending_node,
    "decision": decision_node,
    "explainer": explainer_node
}

def create_migration_graph():
    """Create a migration graph based on configuration.
    
    Returns:
        StateGraph: Configured state graph for migration analysis.
    """
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

def create_migration_graph_with_tools():
    graph = StateGraph(dict)
    graph.add_tools([percent_pending_tool, input_filter_tool, workload_pricing_tool])
    return graph

def load_config_nodes() -> tuple[bool, bool, bool]:
    """Load node execution configuration from YAML file.
    Returns:
        Tuple[bool, bool, bool]: Flags indicating whether to execute pending, cpu, and mem nodes.
    """
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml")
    )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    nodes = config.get("nodes", {})
    cpu = nodes.get("cpu", {}).get("execute", False)
    mem = nodes.get("memory", {}).get("execute", False)
    pending = nodes.get("pending", {}).get("execute", False)

    return pending, cpu, mem