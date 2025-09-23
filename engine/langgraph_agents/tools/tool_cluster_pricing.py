from typing import List, Dict, Any
from langchain_core.tools import tool

@tool("calculate_cluster_pricing", return_direct=False)
def calculate_cluster_pricing(workloads: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    LangGraph tool to calculate total pricing information for clusters.
    
    This function processes workload data from process_monitoring_data output
    and calculates running_price and pending_price for each cluster by summing
    the costs of all workloads in that cluster for the present timestamps.
    
    Args:
        workloads: List of workload dictionaries from process_monitoring_data
        
    Returns:
        Dict mapping cluster_label to pricing information (float)
        Example: {
            "private": {"running_price": 5.8, "pending_price": 2.1}, 
            "public": {"running_price": 3.2, "pending_price": 0.0}
        }
    """
    if not workloads:
        return {"error": "No workload data provided"}

    cluster_pricing = {}
    
    for workload in workloads:
        cluster_label = workload.get("cluster_label")
        if not cluster_label:
            continue
            
        interval_cost_total = workload.get("interval_cost_total", 0.0)
        interval_cost_running = workload.get("interval_cost_running", 0.0)
        pods_pending = workload.get("pods_pending", 0)
        
        if cluster_label not in cluster_pricing:
            cluster_pricing[cluster_label] = {
                "running_price": 0.0,
                "pending_price": 0.0
            }
        
        cluster_pricing[cluster_label]["running_price"] += interval_cost_running    
        cluster_pricing[cluster_label]["pending_price"] += interval_cost_total
    
    for cluster_label in cluster_pricing:
        cluster_pricing[cluster_label]["running_price"] = round(
            cluster_pricing[cluster_label]["running_price"], 4
        )
        cluster_pricing[cluster_label]["pending_price"] = round(
            cluster_pricing[cluster_label]["pending_price"], 4
        )
    
    return cluster_pricing