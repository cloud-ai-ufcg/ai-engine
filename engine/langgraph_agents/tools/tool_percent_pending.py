from langchain.tools import tool
from typing import Dict, Any

@tool("pending_percentage", return_direct=False)
def pending_percentage(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts the pending percentage for each workload from the provided data.
    
    Args:
        data: A dictionary containing workload data, with timestamps as keys.
              Example: {"<timestamp>": {"workloads": [...]}} or {"latest": {"workloads": [...]}}.
    
    Returns:
        A dictionary with timestamps as keys and another dictionary as value,
        containing workload IDs and their pending percentage.
        Example: {"<timestamp>": {"workload_id": "XX%"}}.
    """
    
    if not data:
        return {"error": "no data provided"}
    
    all_timestamps_results = {}

    for timestamp, timestamp_data in data.items():
        workloads = timestamp_data.get("workloads", [])
        
        if not workloads:
            continue

        workload_results = {}
        for wl in workloads:
            wl_id = wl.get("workload_id", "unknown")
            percent_pending = wl.get("percent_pending", 0)
            
            workload_results[wl_id] = f"{percent_pending}%"
        
        all_timestamps_results[timestamp] = workload_results

    return all_timestamps_results