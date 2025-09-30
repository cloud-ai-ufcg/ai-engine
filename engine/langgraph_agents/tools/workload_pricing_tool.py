from typing import List, Dict, Any
from langchain_core.tools import tool


@tool("workload_pricing", return_direct=False)
def workload_pricing(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    LangGraph tool to calculate total pricing information for workloads.

    This function processes workload data from process_monitoring_data output
    and sums the interval_cost_total and interval_cost_running for each workload across all timestamps where it appears.

    Args:
        data: Dict with timestamp as key and another dict with "workloads" list.
        Example: {"<timestamp>": {"workloads": [...]}} or {"latest": {"workloads": [...]}}

    Returns:
        Dict mapping workload_id to the two types of pricing (float)
        Example: {"workload1_id": {"interval_cost_total": 3.5, "interval_cost_running": 2.3},
                  "workload2_id": {"interval_cost_total": 5, "interval_cost_running": 3.4}}
    """
    if not data:
        return {"error": "No workload data provided"}

    pricing_data = {}
    all_workloads = []

    for timestamp_data in data.values():
        all_workloads.extend(timestamp_data.get("workloads", []))

    if not all_workloads:
        return {"error": "No workloads found in the provided data"}

    for workload in all_workloads:
        workload_id = workload.get("workload_id")
        if not workload_id:
            continue

        # Extract cost from pricing information
        pricing = workload.get("pricing", {})
        if not pricing:
            return {"error": "No pricing information found in workload data"}

        interval_cost_total = pricing["interval_cost_total"] if "interval_cost_total" in pricing else 0.0
        interval_cost_running = pricing["interval_cost_running"] if "interval_cost_running" in pricing else 0.0

        # Sum both costs for this workload across all timestamps
        if workload_id in pricing_data:
            pricing_data[workload_id] = {
                "interval_cost_total": round(
                    pricing_data[workload_id]["interval_cost_total"]
                    + interval_cost_total,
                    4,
                ),
                "interval_cost_running": round(
                    pricing_data[workload_id]["interval_cost_running"]
                    + interval_cost_running,
                    4,
                ),
            }
        else:
            pricing_data[workload_id] = {
                "interval_cost_total": round(interval_cost_total, 4),
                "interval_cost_running": round(interval_cost_running, 4),
            }

    return pricing_data
