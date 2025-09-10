from typing import List, Dict, Any
from langchain_core.tools import tool

@tool
def calculate_workload_pricing(workloads: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    LangGraph tool to calculate total pricing information for workloads.
    
    This function processes workload data from process_monitoring_data output
    and sums the interval_cost for each workload across all timestamps where it appears.
    
    Args:
        workloads: List of workload dictionaries from process_monitoring_data
        
    Returns:
        Dict mapping workload_id to total pricing (float)
        Example: {"workload1_id": 2.5, "workload2_id": 4.5}
    """
    if not workloads:
        return {"error": "No workload data provided"}

    pricing_data = {}
    
    for workload in workloads:
        workload_id = workload.get("workload_id")
        if not workload_id:
            continue
            
        # Extract interval_cost from pricing information
        interval_cost_total = workload.get("interval_cost_total", 0.0)
        interval_cost_running = workload.get("interval_cost_running", 0.0)
        
        # Sum the interval cost for this workload across all timestamps
        if workload_id in pricing_data:
            pricing_data[workload_id] = {
                "interval_cost_total": pricing_data[workload_id]["interval_cost_total"] + interval_cost_total,
                "interval_cost_running": pricing_data[workload_id]["interval_cost_running"] + interval_cost_running
            }
        else:
            pricing_data[workload_id] = {"interval_cost_total": interval_cost_total, "interval_cost_running": interval_cost_running}
    
    # Round to appropriate decimal places for currency precision
    for workload_id in pricing_data:
        pricing_data[workload_id] = {k: round(v, 4) for k, v in pricing_data[workload_id].items()}
    
    return pricing_data
