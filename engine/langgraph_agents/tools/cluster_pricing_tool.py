from typing import List, Dict, Any
from langchain_core.tools import tool

@tool("cluster_pricing", return_direct=False)
def cluster_pricing(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    LangGraph tool to calculate total pricing information for clusters.
    
    This function processes workload data from process_monitoring_data output
    and calculates running_price and pending_price for each cluster by summing
    the costs of all workloads in that cluster for the present timestamps.
    
    Args:
        data: Dict with timestamp as key and another dict with "workloads" list.
        Example: {"<timestamp>": {"workloads": [...]}} or {"latest": {"workloads": [...]}}
        
    Returns:
        Dict mapping cluster_label to pricing information (float)
        Example: {
            "private": {"running_price": 5.8, "total_price": 2.1}, 
            "public": {"running_price": 3.2, "total_price": 0.0}
        }
    """
    if not data:
        return {"error": "No workload data provided"}

    cluster_pricing_data = {
        "private": {"running_price": 0.0, "total_price": 0.0},
        "public": {"running_price": 0.0, "total_price": 0.0}
    }
    all_workloads = []

    for timestamp_data in data.values():
        all_workloads.extend(timestamp_data.get("workloads", []))

    if not all_workloads:
        return {"error": "No workloads found in the provided data"}
    
    for workload in all_workloads:
        cluster_label = workload.get("cluster_label")
        if not cluster_label:
            continue
        
        pricing = workload.get("pricing", {})
        if not pricing:
            return {"error": "No pricing information found in workload data"}

        interval_cost_total = pricing["interval_cost_total"] if "interval_cost_total" in pricing else 0.0
        interval_cost_running = pricing["interval_cost_running"] if "interval_cost_running" in pricing else 0.0
        
        cluster_pricing_data[cluster_label]["running_price"] += interval_cost_running    
        cluster_pricing_data[cluster_label]["total_price"] += interval_cost_total
    
    for cluster_label in cluster_pricing_data:
        cluster_pricing_data[cluster_label]["running_price"] = round(
            cluster_pricing_data[cluster_label]["running_price"], 4
        )
        cluster_pricing_data[cluster_label]["total_price"] = round(
            cluster_pricing_data[cluster_label]["total_price"], 4
        )
    
    return cluster_pricing_data