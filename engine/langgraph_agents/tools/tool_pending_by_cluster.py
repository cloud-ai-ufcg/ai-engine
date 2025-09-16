from langchain.tools import tool
from typing import Dict, Any
import time


@tool("pending_by_cluster", return_direct=False)
def pending_by_cluster(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate the percentage of pending workloads per cluster and list workloads with pods pending in each cluster for ALL timestamps.
    Input:
        - dict {"data": {"<timestamp>": {"workloads": [...]}}}
        - OR {"data": {"latest": {"workloads": [...]}}}
    Output:
        - dict {
            "<timestamp>": {
                "cluster_label": {
                    "percent_pending_workloads": float,
                    "pending_workloads": [list of workload_ids with pods pending]
                }
            }
        }
    """
    data = input_dict.get("data", {})

    if not data:
        return {"error": "no data provided"}

    all_timestamps_results = {}

    for timestamp, timestamp_data in data.items():
        workloads = timestamp_data.get("workloads", [])
        if not workloads:
            all_timestamps_results[timestamp] = {"error": "no workloads provided"}
            continue

        clusters = {}
        for wl in workloads:
            cluster = wl.get("cluster_label", "unknown")
            wl_id = wl.get("workload_id", "unknown")
            pending_pods = wl.get("pods_pending", 0)

            if cluster not in clusters:
                clusters[cluster] = {"total": 0, "pending": 0, "pending_workloads": []}

            clusters[cluster]["total"] += 1
            if pending_pods > 0:
                clusters[cluster]["pending"] += 1
                clusters[cluster]["pending_workloads"].append(wl_id)

        timestamp_result = {}
        for cluster, stats in clusters.items():
            total = stats["total"]
            pending = stats["pending"]
            percent_pending = (pending / total * 100) if total > 0 else 0.0
            timestamp_result[cluster] = {
                "percent_pending_workloads": round(percent_pending, 2),
                "pending_workloads": stats["pending_workloads"]
        }

        all_timestamps_results[timestamp] = timestamp_result

    return all_timestamps_results