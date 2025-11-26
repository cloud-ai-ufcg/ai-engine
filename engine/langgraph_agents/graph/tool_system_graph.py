"""Graph definition for the tool-based migration recommendation system."""

from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# Import the tool lazily from the consolidated tools package
from ..tools import (
    pending_by_workload,
    input_filter,
    pending_by_cluster,
    workload_capacity,
    cluster_capacity,
    workload_pricing,
    infra_pricing,
)
from ..nodes import recommendations_node


class MigrationStateToolsGraph(TypedDict):
    """State container used by the migration tools LangGraph."""

    workloads: List[Dict]
    cluster_info: List[Dict]
    interval_duration: str
    decisions: List[int]
    explanations: Dict
    pending_tool_result: Dict


# -----------------------
# Functions for each node
# -----------------------


def input_filter_node(state: dict) -> dict:
    """Wrap the `input_filter` LangChain tool so that it plays nicely with LangGraph.

    LangGraph passes the full state dictionary as the *single* argument to a node, but
    `input_filter` is declared as a LangChain tool expecting a named argument
    ``data``.  This helper extracts the `workloads` list from the state and feeds it
    into the tool, then writes the normalised list back into the state under the
    same key.
    """

    workloads = state.get("workloads", [])
    normalised = input_filter.invoke({"data": workloads})

    return {"workloads": normalised}


def pending_by_workload_node(state: dict) -> dict:
    """Use the `pending_by_workload` tool to analyse workloads by job."""
    workloads = state.get("workloads", [])
    cluster_info = state.get("cluster_info", [])
    interval_duration = state.get("interval_duration", "30s")

    if not workloads:
        return {"pending_by_workload": {"error": "no workloads provided"}}

    result = pending_by_workload.invoke(
        {
            "data": {
                "latest": {
                    "workloads": workloads,
                    "cluster_info": cluster_info,
                    "interval_duration": interval_duration,
                }
            }
        }
    )

    return {"pending_by_workload": result}


def cluster_capacity_node(state: dict) -> dict:
    """Compute cluster capacity using the corresponding LangChain tool."""
    workloads = state.get("workloads", [])
    cluster_info = state.get("cluster_info", [])
    interval_duration = state.get("interval_duration", "30s")

    if not workloads:
        return {"cluster_capacity": {"error": "no workloads provided"}}

    result = cluster_capacity.invoke(
        {
            "data": {
                "latest": {
                    "workloads": workloads,
                    "cluster_info": cluster_info,
                    "interval_duration": interval_duration,
                }
            }
        }
    )
    return {"cluster_capacity": result}


def pending_by_cluster_node(state: dict) -> dict:
    """Compute pending workloads grouped by cluster using the tool."""
    workloads = state.get("workloads", [])
    cluster_info = state.get("cluster_info", [])
    interval_duration = state.get("interval_duration", "30s")

    if not workloads:
        return {"pending_by_cluster": {"error": "no workloads provided"}}

    result = pending_by_cluster.invoke(
        {
            "data": {
                "latest": {
                    "workloads": workloads,
                    "cluster_info": cluster_info,
                    "interval_duration": interval_duration,
                }
            }
        }
    )
    return {"pending_by_cluster": result}


def prepare_pending_data(state: dict) -> dict:
    """Prepare the minimal data structure expected by pending tools."""
    workloads = state.get("workloads", [])
    return {"data": {"latest": {"workloads": workloads}}}


def workload_pricing_node(state: dict) -> dict:
    """Calculate workload-level pricing information using the tool."""
    workloads = state.get("workloads", [])
    cluster_info = state.get("cluster_info", [])
    interval_duration = state.get("interval_duration", "30s")

    if not workloads:
        return {"workload_pricing": {"error": "no workloads provided"}}

    result = workload_pricing.invoke(
        {
            "data": {
                "latest": {
                    "workloads": workloads,
                    "cluster_info": cluster_info,
                    "interval_duration": interval_duration,
                }
            }
        }
    )
    return {"workload_pricing": result}


def workload_capacity_node(state: dict) -> dict:
    """Compute workload capacity per job using the LangChain tool."""
    workloads = state.get("workloads", [])
    cluster_info = state.get("cluster_info", [])
    interval_duration = state.get("interval_duration", "30s")

    if not workloads:
        return {"workload_capacity": {"error": "no workloads provided"}}

    result = workload_capacity.invoke(
        {
            "data": {
                "latest": {
                    "workloads": workloads,
                    "cluster_info": cluster_info,
                    "interval_duration": interval_duration,
                }
            }
        }
    )
    return {"workload_capacity": result}


def cluster_pricing_node(state: dict) -> dict:
    """Calculate infrastructure pricing information using the tool."""
    workloads = state.get("workloads", [])
    cluster_info = state.get("cluster_info", [])
    interval_duration = state.get("interval_duration", "30s")

    if not workloads:
        return {"infra_princing": {"error": "no workloads provided"}}

    result = infra_pricing.invoke(
        {
            "data": {
                "latest": {
                    "workloads": workloads,
                    "cluster_info": cluster_info,
                    "interval_duration": interval_duration,
                }
            }
        }
    )
    return {"infra_princing": result}


# -----------------------
# Graph Creation
# -----------------------
def create_tool_system_migration_graph():
    """Create a migration graph based on tool system nodes.

    Returns:
        StateGraph: Configured state graph for migration analysis.
    """
    graph = StateGraph(MigrationStateToolsGraph)

    graph.add_node("input_filter", input_filter_node)
    graph.add_node("cluster_capacity", cluster_capacity_node)
    graph.add_node("pending_by_cluster", pending_by_cluster_node)
    graph.add_node("pending_by_workload", pending_by_workload_node)
    graph.add_node("workload_pricing", workload_pricing_node)
    graph.add_node("workload_capacity", workload_capacity_node)
    graph.add_node("infra_pricing", cluster_pricing_node)
    graph.add_node("recommendations", recommendations_node)

    # Parallel nodes entry points
    graph.add_edge(START, "input_filter")

    graph.add_edge("input_filter", "pending_by_workload")
    graph.add_edge("input_filter", "cluster_capacity")
    graph.add_edge("input_filter", "pending_by_cluster")
    graph.add_edge("input_filter", "workload_capacity")
    graph.add_edge("input_filter", "workload_pricing")
    graph.add_edge("input_filter", "infra_pricing")

    graph.add_edge("pending_by_workload", "recommendations")
    graph.add_edge("cluster_capacity", "recommendations")
    graph.add_edge("pending_by_cluster", "recommendations")
    graph.add_edge("workload_capacity", "recommendations")
    graph.add_edge("workload_pricing", "recommendations")
    graph.add_edge("infra_pricing", "recommendations")

    graph.add_edge("recommendations", END)

    return graph.compile(checkpointer=MemorySaver())
