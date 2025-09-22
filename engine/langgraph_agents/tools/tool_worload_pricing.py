from typing import List, Dict, Any
from langchain_core.tools import tool

@tool("calculate_workload_pricing", return_direct=False)
def calculate_workload_pricing(workloads: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    LangGraph tool to calculate total pricing information for workloads.
    
    This function processes workload data from process_monitoring_data output
    and sums the interval_cost_total and interval_cost_running for each workload across all timestamps where it appears.
    
    Args:
        workloads: List of workload dictionaries from process_monitoring_data
        
    Returns:
        Dict mapping workload_id to the two types of pricing (float)
        Example: {"workload1_id": {"interval_cost_total": 3.5, "interval_cost_running": 2.3}, 
                  "workload2_id": {"interval_cost_total": 5, "interval_cost_running": 3.4}}
    """
    if not workloads:
        return {"error": "No workload data provided"}

    pricing_data = {}
    
    for workload in workloads:
        workload_id = workload.get("workload_id")
        if not workload_id:
            continue
            
        # Extract cost from pricing information
        interval_cost_total = workload.get("interval_cost_total", 0.0)
        interval_cost_running = workload.get("interval_cost_running", 0.0)
        
        # Sum both costs for this workload across all timestamps
        if workload_id in pricing_data:
            pricing_data[workload_id] = {
                "interval_cost_total": round(pricing_data[workload_id]["interval_cost_total"] + interval_cost_total, 4),
                "interval_cost_running": round(pricing_data[workload_id]["interval_cost_running"] + interval_cost_running, 4)
            }
        else:
            pricing_data[workload_id] = {"interval_cost_total": round(interval_cost_total, 4), "interval_cost_running": round(interval_cost_running, 4)}
    
    return pricing_data
