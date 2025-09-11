from langchain.tools import tool
import pandas as pd
import time


@tool("pending_by_cluster", return_direct=False)
def pending_by_cluster(input_dict: dict) -> dict:
    """
    Calculate the percentage of pending workloads per cluster and list workloads with pods pending in each cluster.
    Input:
        - dict {"data": {"<timestamp>": {"workloads": [...]}}}
        - OU {"data": {"latest": {"workloads": [...]}}}
    Output:
        - dict {cluster_label: {
            "percent_pending_workloads": float,
            "pending_workloads": [list of workload_ids with pods pending]
        }}
    """
    data = input_dict.get("data", {})

    if "latest" in data:
        data = {str(int(time.time())): data["latest"]}

    if not data:
        return {"error": "no data provided"}

    try:
        latest_ts = max(data.keys(), key=lambda x: int(x))
    except Exception:
        return {"error": "timestamps not in expected format"}

    workloads = data[latest_ts].get("workloads", [])
    if not workloads:
        return {"error": "not found workloads"}

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

    result = {}
    for cluster, stats in clusters.items():
        total = stats["total"]
        pending = stats["pending"]
        percent_pending = (pending / total * 100) if total > 0 else 0.0
        result[cluster] = {
            "percent_pending_workloads": round(percent_pending, 2),
            "pending_workloads": stats["pending_workloads"]
        }

    return result