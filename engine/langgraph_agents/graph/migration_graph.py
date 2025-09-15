import os
import pandas as pd
import yaml
from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from ..tools import pending_by_cluster


from ..nodes import (
    cpu_checker,
    mem_checker,
    pending_checker,
    decision_agent,
    explainer_agent
)

class MigrationState(TypedDict):
    workloads: List[Dict]
    cpu_votes: List[int]
    mem_votes: List[int]
    pending_votes: List[int]
    final_decisions: List[int]
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

    result = pending_by_cluster.invoke({
        "input_dict": {"data": {"latest": {"workloads": workloads}}}
    })

    return {"pending_tool_result": result}

def cpu_node(state: dict) -> dict:
    """Process CPU information skipping workloads.
    
    Args:
        state (dict): Current state containing workloads.
    
    Returns:
        Dict: Updated state with CPU votes.
    """ 
    df = pd.DataFrame(state["workloads"])
    return {"cpu_votes": cpu_checker(df)}

def mem_node(state: dict) -> dict:
    """Process Memory information, skipping workloads.

    Args:
        state (dict): Current state containing workloads.

    Returns:
        Dict: Updated state with Memory votes.
    """
    df = pd.DataFrame(state["workloads"])
    return {"mem_votes": mem_checker(df)}

def pending_node(state: dict) -> dict:
    """Process Pending information, skipping workloads.

    Args:
        state (dict): Current state containing workloads.

    Returns:
        Dict: Updated state with Pending votes.
    """
    df = pd.DataFrame(state["workloads"])
    return {"pending_votes": pending_checker(df)}


def decision_node(state: dict) -> dict:
    """Make final decisions based on votes from previous nodes.
    
    Args:
        state (dict): Current state containing votes.
    
    Returns:
        Dict: Updated state with final decisions.
    """
    return {
        "final_decisions": decision_agent(
            state.get("cpu_votes", []),
            state.get("mem_votes", []),
            state.get("pending_votes", [])
        )
    }

def explainer_node(state: dict) -> dict:
    """Generate explanations for the final decisions made.
    
    Args:
        state (dict): Current state containing workloads and final decisions.
    
    Returns:
        Dict: Updated state with explanations.
    """
    df = pd.DataFrame(state["workloads"])
    return {
        "explanations": explainer_agent(
            df,
            {
                "cpu": state.get("cpu_votes", []),
                "mem": state.get("mem_votes", []),
                "pending": state.get("pending_votes", [])
            },
            state.get("final_decisions", [])
        )
    }

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
    graph = StateGraph(MigrationState)

    
    graph.add_node("cpu", cpu_node)
    graph.add_node("mem", mem_node)
    graph.add_node("pending", pending_node)
    graph.add_node("decision", decision_node)
    graph.add_node("explainer", explainer_node)
    graph.add_node("pending_tool", pending_tool_node)

    graph.set_entry_point("split")
    graph.add_node("split", lambda state: state)

    graph.add_edge("split", "cpu")
    graph.add_edge("split", "mem")
    graph.add_edge("split", "pending")
    graph.add_edge("split", "pending_tool")
    graph.add_edge("cpu", "decision")
    graph.add_edge("mem", "decision")
    graph.add_edge("pending", "decision")
    graph.add_edge("pending_tool", "decision")

    graph.add_edge("decision", "explainer")
    graph.add_edge("explainer", END)

    return graph.compile(checkpointer=MemorySaver())

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

# -----------------------
# Execution Function
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
        "explanations": {},
        "pending_tool_result": {} 
    }
    graph = create_migration_graph()
    final_state = graph.invoke(initial_state)
    result_df = pd.DataFrame(final_state["workloads"]).copy()
    result_df["label"] = final_state["final_decisions"]
    explanations = final_state["explanations"]
    return result_df, explanations
